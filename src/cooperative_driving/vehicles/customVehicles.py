import numpy as np
from highway_env.vehicle.behavior import IDMVehicle
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle

class IntentMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.intent = {
            # Positive = accelerate, negative = decelerate/brake, zero = maintain speed.
            "intent_acceleration": 0.0,
            # Low-level steering intent. Zero means no steering input. Positive is steer left and negative is steer right
            "intent_steering": 0.0,
            # Desired lane index minus current lane index. Zero means stay in lane.
            "intent_lane_change": 0.0,
        }

    def record_intent(self):
        self.intent.update({
            # A negative acceleration represents deceleration.
            "intent_acceleration": self.action.get("acceleration", 0.0),
            # The vehicle's current steering command.
            "intent_steering": self.action.get("steering", 0.0),
            # Positive/negative values indicate a target lane change direction.
            "intent_lane_change": (
                self.target_lane_index[2] - self.lane_index[2]
                if hasattr(self, "target_lane_index")
                else 0
            ),
        })

    def act(self, action=None):
        result = super().act(action)
        self.record_intent()
        return result

    def to_dict(self, origin_vehicle=None, observe_intentions=True):
        data = super().to_dict(origin_vehicle, observe_intentions)
        data.update(self.intent)
        data["vehicle_id"] = self.vehicle_id
        return data


class IntentIDMVehicle(IntentMixin, IDMVehicle):
    pass


class IntentControlledVehicle(IntentMixin, ControlledVehicle):
    pass



# ---------------------------------------------------------------------------
# Merge scenario: hidden driver types
# ---------------------------------------------------------------------------

YIELDING = "yielding"
AGGRESSIVE = "aggressive"
DRIVER_TYPES = (YIELDING, AGGRESSIVE)


class DriverTypeIDMVehicle(IntentMixin, IDMVehicle):
    """IDM car with a private driver type that only matters when someone merges.

    Both types drive identically (same IDM parameters, no lane changes) until a
    vehicle ahead of them starts changing into their lane. Then:

    - ``yielding`` treats the merging car as its new leader and brakes to open
      a gap;
    - ``aggressive`` ignores the merging car and accelerates to close the gap
      until the merging car has actually entered its lane.

    The type is therefore invisible in kinematics before the merge starts, and
    the only way for the ego vehicle to know it in advance is to ask.
    """

    # How far ahead (longitudinal metres) a merging car must be to trigger a
    # reaction. Cars further ahead are handled by normal IDM.
    REACTION_DISTANCE = 40.0
    # Dense-traffic headway so a tightly spaced platoon is stable instead of
    # braking into a self-made jam (IDM default: 10 m + 1.5 s).
    DISTANCE_WANTED = 3.0 + ControlledVehicle.LENGTH  # [m]
    TIME_WANTED = 0.8  # [s]
    AGGRESSIVE_ACCELERATION = 6.0  # [m/s2] while closing the gap
    MERGE_DONE_OFFSET = 1.0  # [m] lateral offset at which a merge is complete

    def __init__(self, *args, driver_type=YIELDING, **kwargs):
        super().__init__(*args, **kwargs)
        self.driver_type = driver_type
        self.enable_lane_change = False

    def act(self, action=None):
        if self.crashed:
            return
        # Normal IDM first (this may already include the merging car once it
        # overlaps our lane), then apply the driver-type reaction.
        IDMVehicle.act(self)
        merger = self._merging_vehicle()
        if merger is not None:
            acceleration = self.action["acceleration"]
            if self.driver_type == YIELDING:
                yield_acceleration = self.acceleration(
                    ego_vehicle=self, front_vehicle=merger
                )
                acceleration = min(acceleration, yield_acceleration)
            else:
                acceleration = self.AGGRESSIVE_ACCELERATION
            self.action["acceleration"] = float(
                np.clip(acceleration, -self.ACC_MAX, self.ACC_MAX)
            )
        self.record_intent()

    def _merging_vehicle(self):
        """Return the closest car ahead that is changing into our lane."""
        own_lane = self.lane_index[2]
        closest, closest_distance = None, None
        for vehicle in self.road.vehicles:
            if vehicle is self or not hasattr(vehicle, "target_lane_index"):
                continue
            # A car is still merging until it is (nearly) centred in our lane.
            lateral_offset = abs(self.lane.local_coordinates(vehicle.position)[1])
            changing_into_us = vehicle.target_lane_index[2] == own_lane and (
                vehicle.lane_index[2] != own_lane
                or lateral_offset > self.MERGE_DONE_OFFSET
            )
            if not changing_into_us:
                continue
            distance = vehicle.position[0] - self.position[0]
            if 0.0 < distance <= self.REACTION_DISTANCE and (
                closest_distance is None or distance < closest_distance
            ):
                closest, closest_distance = vehicle, distance
        return closest


class StalledVehicle(IntentMixin, Vehicle):
    """A stopped car that blocks the ego lane, forcing a merge before it."""

    driver_type = "stalled"

    def __init__(self, road, position, heading=0.0):
        super().__init__(road, position, heading, speed=0.0)

    def act(self, action=None):
        self.action = {"steering": 0.0, "acceleration": 0.0}
        self.record_intent()


class CruiseControlledVehicle(IntentControlledVehicle):
    """Ego vehicle whose meta-actions set the lane and desired speed, while a
    low-level adaptive cruise control (IDM gap term) keeps a safe distance to
    the car ahead in its current and target lanes.

    This removes trivial rear-end crashes, so the remaining risk comes from
    *other* drivers' reactions to the ego's lane changes.
    """

    COMFORT_ACC_MAX = IDMVehicle.COMFORT_ACC_MAX
    COMFORT_ACC_MIN = IDMVehicle.COMFORT_ACC_MIN
    ACC_MAX = IDMVehicle.ACC_MAX
    DISTANCE_WANTED = IDMVehicle.DISTANCE_WANTED
    TIME_WANTED = IDMVehicle.TIME_WANTED
    DELTA = IDMVehicle.DELTA
    acceleration = IDMVehicle.acceleration
    desired_gap = IDMVehicle.desired_gap
    EXIT_LANE_MIN_GAP = ControlledVehicle.LENGTH + 2.0  # [m]

    def act(self, action=None):
        super().act(action)  # sets steering + speed-control acceleration
        if self.crashed:
            return
        acceleration = self.action["acceleration"]
        changing_lane = self.lane_index != self.target_lane_index
        for lane_index in {self.lane_index, self.target_lane_index}:
            front_vehicle, _ = self.road.neighbour_vehicles(self, lane_index)
            if front_vehicle is None:
                continue
            # While leaving a lane, the car ahead in that lane only matters
            # in an emergency; otherwise a car stopped behind an obstacle
            # could never pull out of the blocked lane.
            if (
                changing_lane
                and lane_index == self.lane_index
                and self.lane_distance_to(front_vehicle) > self.EXIT_LANE_MIN_GAP
            ):
                continue
            acceleration = min(
                acceleration,
                self.acceleration(ego_vehicle=self, front_vehicle=front_vehicle),
            )
        self.action["acceleration"] = float(
            np.clip(acceleration, -self.ACC_MAX, self.ACC_MAX)
        )
