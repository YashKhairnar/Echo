from highway_env.envs.highway_env import HighwayEnv
from highway_env.utils import near_split
from highway_env.vehicle.kinematics import Vehicle
import numpy as np
from highway_env import utils

from ..vehicles.customVehicles import (
    AGGRESSIVE,
    CruiseControlledVehicle,
    YIELDING,
    DriverTypeIDMVehicle,
    IntentControlledVehicle,
    IntentIDMVehicle,
    StalledVehicle,
)


class CustomHighwayEnv(HighwayEnv):
    @classmethod
    def default_config(cls):
        # The observation is a fixed 10-row kinematics vector: presence,
        # relative position/velocity, and a stable vehicle ID. Private
        # information (intent, driver type) is only available via messages.
        config = super().default_config()
        config.update({
            "vehicles_count":10,
            "controlled_vehicles":1,
                "observation": {
                    "type": "Kinematics",
                    "vehicles_count": 10,
                    "features": [
                        "presence", "x", "y", "vx", "vy", "vehicle_id",
                    ],
                    "see_behind": True,
                    "normalize": True
                }
            })
        return config

    def _create_vehicles(self):
        # IDs are assigned when cars are created and remain stable for the
        # lifetime of the episode. Observation slots can still change.
        self.next_vehicle_id = 0

        """Create vehicles that broadcast their current low-level intent."""
        other_per_controlled = near_split(
            self.config["vehicles_count"],
            num_bins=self.config["controlled_vehicles"],
        )

        self.controlled_vehicles = []
        for others in other_per_controlled:
            vehicle = Vehicle.create_random(
                self.road,
                speed=25.0,
                lane_id=self.config["initial_lane_id"],
                spacing=self.config["ego_spacing"],
            )
            vehicle = IntentControlledVehicle(
                self.road, vehicle.position, vehicle.heading, vehicle.speed
            )
            vehicle.vehicle_id = self.next_vehicle_id
            self.next_vehicle_id += 1
            self.controlled_vehicles.append(vehicle)
            self.road.vehicles.append(vehicle)

            for _ in range(others):
                vehicle = IntentIDMVehicle.create_random(
                    self.road,
                    spacing=1 / self.config["vehicles_density"],
                )
                vehicle.randomize_behavior()
                vehicle.vehicle_id = self.next_vehicle_id
                self.next_vehicle_id += 1
                self.road.vehicles.append(vehicle)


TARGET_LANE = 0  # upper lane: the lane the ego must merge into
EGO_LANE = 1     # lower lane: blocked by a stalled car ahead


class CooperativeMergeEnv(CustomHighwayEnv):
    """Forced-merge scenario where a neighbour's hidden driver type matters.

    Layout (two lanes, ego starts in the lower lane):

    - The ego lane is blocked by a stalled car somewhere ahead, so the ego
      must merge into the target lane before reaching it.
    - The target lane carries a platoon of ``DriverTypeIDMVehicle`` cars.
      Each car is secretly ``yielding`` or ``aggressive``. Both types drive
      the same until the ego starts merging in front of them; then a yielding
      car brakes to open a gap and an aggressive car speeds up to close it.

    Merging early in front of a yielding car is safe and fast; doing the same
    in front of an aggressive car usually causes a crash. Without asking, the
    ego has to either gamble or slow down and wait for a large gap.
    """

    @classmethod
    def default_config(cls):
        config = super().default_config()
        config.update({
            "lanes_count": 2,
            "vehicles_count": 9,          # platoon cars in the target lane
            "duration": 20,               # [s] at 1 decision per second
            "ego_speed": 25.0,
            "ego_start_x": 100.0,
            "platoon_start_offset": [-90.0, -50.0],  # first car vs ego
            "platoon_spacing": [25.0, 45.0],          # centre-to-centre [m]
            "platoon_speed": [23.0, 26.0],
            "stalled_distance": [250.0, 350.0],       # ahead of ego [m]
            "yielding_probability": 0.5,
            # Reward terms (not normalised).
            "collision_reward": -2.0,
            "high_speed_reward": 0.4,
            "target_lane_reward": 0.6,
            "reward_speed_range": [20, 30],
            "normalize_reward": False,
        })
        return config

    def _create_vehicles(self):
        rng = self.np_random
        cfg = self.config
        self.next_vehicle_id = 0
        lane_of = lambda index: self.road.network.get_lane(("0", "1", index))

        ego_x = cfg["ego_start_x"]
        ego_lane = lane_of(EGO_LANE)
        ego = CruiseControlledVehicle(
            self.road,
            ego_lane.position(ego_x, 0.0),
            ego_lane.heading_at(ego_x),
            cfg["ego_speed"],
        )
        ego.vehicle_id = self._new_id()
        self.controlled_vehicles = [ego]
        self.road.vehicles.append(ego)

        target_lane = lane_of(TARGET_LANE)
        x = ego_x + rng.uniform(*cfg["platoon_start_offset"])
        platoon_speed = rng.uniform(*cfg["platoon_speed"])
        for _ in range(cfg["vehicles_count"]):
            driver_type = (
                YIELDING if rng.random() < cfg["yielding_probability"]
                else AGGRESSIVE
            )
            vehicle = DriverTypeIDMVehicle(
                self.road,
                target_lane.position(x, 0.0),
                target_lane.heading_at(x),
                platoon_speed,
                # Slightly above cruise speed so IDM settles near cruise.
                target_speed=platoon_speed + 2.0,
                driver_type=driver_type,
            )
            vehicle.vehicle_id = self._new_id()
            self.road.vehicles.append(vehicle)
            x += rng.uniform(*cfg["platoon_spacing"])

        stalled_x = ego_x + rng.uniform(*cfg["stalled_distance"])
        stalled = StalledVehicle(
            self.road, ego_lane.position(stalled_x, 0.0), ego_lane.heading_at(stalled_x)
        )
        stalled.vehicle_id = self._new_id()
        self.road.vehicles.append(stalled)
        self.stalled_vehicle = stalled
        self.merge_time = None

    def _new_id(self):
        vehicle_id = self.next_vehicle_id
        self.next_vehicle_id += 1
        return vehicle_id

    def _rewards(self, action):
        forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
        scaled_speed = utils.lmap(
            forward_speed, self.config["reward_speed_range"], [0, 1]
        )
        return {
            "collision_reward": float(self.vehicle.crashed),
            "high_speed_reward": float(np.clip(scaled_speed, 0, 1)),
            "target_lane_reward": float(self.in_target_lane()),
            "on_road_reward": float(self.vehicle.on_road),
        }

    def _reward(self, action):
        rewards = self._rewards(action)
        if self.vehicle.crashed:
            return self.config["collision_reward"]
        reward = (
            self.config["high_speed_reward"] * rewards["high_speed_reward"]
            + self.config["target_lane_reward"] * rewards["target_lane_reward"]
        )
        return reward * rewards["on_road_reward"]

    def target_lane_neighbours(self):
        """Return (front, rear) cars in the target lane relative to the ego."""
        ego = self.vehicle
        candidates = [
            vehicle for vehicle in self.road.vehicles
            if vehicle is not ego
            and vehicle is not self.stalled_vehicle
            and vehicle.lane_index[2] == TARGET_LANE
        ]
        ahead = [v for v in candidates if v.position[0] >= ego.position[0]]
        behind = [v for v in candidates if v.position[0] < ego.position[0]]
        front = min(ahead, key=lambda v: v.position[0], default=None)
        rear = max(behind, key=lambda v: v.position[0], default=None)
        return front, rear

    def in_target_lane(self):
        return self.vehicle.lane_index[2] == TARGET_LANE

    def _info(self, obs, action=None):
        info = super()._info(obs, action)
        if self.merge_time is None and self.in_target_lane() and not self.vehicle.crashed:
            self.merge_time = self.time
        info.update({
            "merged": self.merge_time is not None,
            "merge_time": self.merge_time,
            "crash_with_stalled": bool(
                self.vehicle.crashed and self.stalled_vehicle.crashed
            ),
        })
        return info


ENVIRONMENTS = {
    "merge": CooperativeMergeEnv,
    "highway": CustomHighwayEnv,
}


def make_environment(name="merge", **kwargs):
    """Create one of the project environments by name."""
    return ENVIRONMENTS[name](**kwargs)
