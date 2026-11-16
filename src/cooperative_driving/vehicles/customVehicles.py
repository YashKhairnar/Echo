"""Vehicles of the merge scenario.

- ``DriverTypeIDMVehicle``: a platoon car that is secretly yielding or aggressive.
- ``StalledVehicle``: a stopped car that blocks the ego lane.
- ``CruiseControlledVehicle``: the ego car.
"""

import numpy as np
from highway_env.vehicle.behavior import IDMVehicle
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle

YIELDING = "yielding"
AGGRESSIVE = "aggressive"


class IntentMixin:
    """Gives a car a stable ``vehicle_id`` in observations and remembers its
    current acceleration (its "intent"), which it can share in a message."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.intent = {"intent_acceleration": 0.0}

    def record_intent(self):
        self.intent["intent_acceleration"] = self.action.get("acceleration", 0.0)

    def act(self, action=None):
        super().act(action)
        self.record_intent()

    def to_dict(self, origin_vehicle=None, observe_intentions=True):
        # The observation reads its features from here, including vehicle_id.
        data = super().to_dict(origin_vehicle, observe_intentions)
        data["vehicle_id"] = self.vehicle_id
        return data


class DriverTypeIDMVehicle(IntentMixin, IDMVehicle):
    """Platoon car with a hidden driver type.

    Both types drive the same until a car ahead starts changing into their
    lane. Then a ``yielding`` car brakes to let it in, and an ``aggressive``
    car speeds up to close the gap. So the type cannot be seen before the
    merge; the only way to know it in advance is to ask.
    """

    REACTION_DISTANCE = 40.0        # [m] react to merging cars this close ahead
    DISTANCE_WANTED = 3.0 + ControlledVehicle.LENGTH  # [m] dense-platoon headway
    TIME_WANTED = 0.8               # [s]
    AGGRESSIVE_ACCELERATION = 6.0   # [m/s2]
    MERGE_DONE_OFFSET = 1.0         # [m] a merging car is "in" once this centred

    def __init__(self, *args, driver_type=YIELDING, **kwargs):
        super().__init__(*args, **kwargs)
        self.driver_type = driver_type
        self.enable_lane_change = False

    def act(self, action=None):
        if self.crashed:
            return
        IDMVehicle.act(self)  # normal driving first
        merger = self._merging_vehicle()
        if merger is not None:
            if self.driver_type == YIELDING:
                # Treat the merging car as the new car ahead and brake for it.
                acceleration = min(
                    self.action["acceleration"],
                    self.acceleration(ego_vehicle=self, front_vehicle=merger),
                )
            else:
                acceleration = self.AGGRESSIVE_ACCELERATION
            self.action["acceleration"] = float(np.clip(acceleration, -self.ACC_MAX, self.ACC_MAX))
        self.record_intent()

    def _merging_vehicle(self):
        """The closest car ahead (within reaction distance) changing into our lane."""
        own_lane = self.lane_index[2]
        merging = []
        for vehicle in self.road.vehicles:
            if vehicle is self or not hasattr(vehicle, "target_lane_index"):
                continue
            lateral_offset = abs(self.lane.local_coordinates(vehicle.position)[1])
            still_merging = vehicle.target_lane_index[2] == own_lane and (
                vehicle.lane_index[2] != own_lane or lateral_offset > self.MERGE_DONE_OFFSET
            )
            distance = vehicle.position[0] - self.position[0]
            if still_merging and 0.0 < distance <= self.REACTION_DISTANCE:
                merging.append((distance, vehicle))
        return min(merging, key=lambda item: item[0])[1] if merging else None


class StalledVehicle(IntentMixin, Vehicle):
    """A stopped car that blocks the ego lane, forcing a merge before it."""

    driver_type = "stalled"

    def __init__(self, road, position, heading=0.0):
        super().__init__(road, position, heading, speed=0.0)

    def act(self, action=None):
        self.action = {"steering": 0.0, "acceleration": 0.0}
        self.record_intent()


class CruiseControlledVehicle(IntentMixin, ControlledVehicle):
    """The ego car.

    Its actions (change lane, faster, slower, ...) work as usual, and an
    adaptive cruise control brakes for the car ahead in its current and target
    lane. That removes silly rear-end crashes, so the only real risk is how
    the *other* drivers react to the ego's lane change.
    """

    # Borrow IDM's gap-keeping formula for the cruise control.
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
        super().act(action)
        if self.crashed:
            return
        acceleration = self.action["acceleration"]
        changing_lane = self.lane_index != self.target_lane_index
        for lane_index in {self.lane_index, self.target_lane_index}:
            front_vehicle, _ = self.road.neighbour_vehicles(self, lane_index)
            if front_vehicle is None:
                continue
            # While leaving a lane, ignore the car ahead in it unless it is
            # very close; otherwise the ego could never pull out from behind
            # the stalled car.
            leaving_this_lane = changing_lane and lane_index == self.lane_index
            if leaving_this_lane and self.lane_distance_to(front_vehicle) > self.EXIT_LANE_MIN_GAP:
                continue
            acceleration = min(
                acceleration, self.acceleration(ego_vehicle=self, front_vehicle=front_vehicle)
            )
        self.action["acceleration"] = float(np.clip(acceleration, -self.ACC_MAX, self.ACC_MAX))
