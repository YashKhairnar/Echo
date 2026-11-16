"""The cooperative merge environment.

Two lanes. The ego starts in the lower lane (``EGO_LANE``), which is blocked
by a stalled car 250-350 m ahead, so it must merge into the upper lane
(``TARGET_LANE``). The upper lane carries a platoon of nine cars; each is
secretly yielding or aggressive (see ``DriverTypeIDMVehicle``).

Merging in front of a yielding car is safe; in front of an aggressive car it
usually ends in a crash. Without asking, the ego must gamble or slow down and
wait for a big gap.
"""

import numpy as np
from highway_env import utils
from highway_env.envs.highway_env import HighwayEnv

from ..vehicles.customVehicles import (
    AGGRESSIVE,
    YIELDING,
    CruiseControlledVehicle,
    DriverTypeIDMVehicle,
    StalledVehicle,
)

TARGET_LANE = 0  # upper lane: the lane the ego must merge into
EGO_LANE = 1     # lower lane: blocked by the stalled car


class CooperativeMergeEnv(HighwayEnv):

    @classmethod
    def default_config(cls):
        config = super().default_config()
        config.update({
            # Observation: 10 rows (ego + 9 nearest cars) of
            # [presence, x, y, vx, vy, vehicle_id]. Driver types are NOT in
            # here; they are only available by asking (messages).
            "observation": {
                "type": "Kinematics",
                "vehicles_count": 10,
                "features": ["presence", "x", "y", "vx", "vy", "vehicle_id"],
                "see_behind": True,
                "normalize": True,
            },
            "controlled_vehicles": 1,
            "lanes_count": 2,
            "vehicles_count": 9,                      # platoon cars
            "duration": 20,                           # steps (1 per second)
            "ego_speed": 25.0,
            "ego_start_x": 100.0,
            "platoon_start_offset": [-90.0, -50.0],   # first platoon car vs ego [m]
            "platoon_spacing": [25.0, 45.0],          # between platoon cars [m]
            "platoon_speed": [23.0, 26.0],            # [m/s]
            "stalled_distance": [250.0, 350.0],       # stalled car ahead of ego [m]
            "yielding_probability": 0.5,
            # Reward per step: 0.4 * speed + 0.6 * in target lane; -2 on a crash.
            "collision_reward": -2.0,
            "high_speed_reward": 0.4,
            "target_lane_reward": 0.6,
            "reward_speed_range": [20, 30],
            "normalize_reward": False,
        })
        return config

    def _create_vehicles(self):
        rng, cfg = self.np_random, self.config
        self.next_vehicle_id = 0
        ego_lane = self.road.network.get_lane(("0", "1", EGO_LANE))
        target_lane = self.road.network.get_lane(("0", "1", TARGET_LANE))

        # Ego car.
        ego_x = cfg["ego_start_x"]
        ego = CruiseControlledVehicle(
            self.road, ego_lane.position(ego_x, 0.0), ego_lane.heading_at(ego_x), cfg["ego_speed"]
        )
        self._add(ego)
        self.controlled_vehicles = [ego]

        # Platoon in the target lane, each car yielding or aggressive at random.
        x = ego_x + rng.uniform(*cfg["platoon_start_offset"])
        platoon_speed = rng.uniform(*cfg["platoon_speed"])
        for _ in range(cfg["vehicles_count"]):
            driver_type = YIELDING if rng.random() < cfg["yielding_probability"] else AGGRESSIVE
            self._add(DriverTypeIDMVehicle(
                self.road,
                target_lane.position(x, 0.0),
                target_lane.heading_at(x),
                platoon_speed,
                target_speed=platoon_speed + 2.0,  # IDM then settles near cruise speed
                driver_type=driver_type,
            ))
            x += rng.uniform(*cfg["platoon_spacing"])

        # Stalled car blocking the ego lane.
        stalled_x = ego_x + rng.uniform(*cfg["stalled_distance"])
        self.stalled_vehicle = StalledVehicle(
            self.road, ego_lane.position(stalled_x, 0.0), ego_lane.heading_at(stalled_x)
        )
        self._add(self.stalled_vehicle)
        self.merge_time = None

    def _add(self, vehicle):
        """Add a car to the road with the next stable ID."""
        vehicle.vehicle_id = self.next_vehicle_id
        self.next_vehicle_id += 1
        self.road.vehicles.append(vehicle)

    def _rewards(self, action):
        forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
        scaled_speed = utils.lmap(forward_speed, self.config["reward_speed_range"], [0, 1])
        return {
            "collision_reward": float(self.vehicle.crashed),
            "high_speed_reward": float(np.clip(scaled_speed, 0, 1)),
            "target_lane_reward": float(self.in_target_lane()),
            "on_road_reward": float(self.vehicle.on_road),
        }

    def _reward(self, action):
        if self.vehicle.crashed:
            return self.config["collision_reward"]
        rewards = self._rewards(action)
        reward = (
            self.config["high_speed_reward"] * rewards["high_speed_reward"]
            + self.config["target_lane_reward"] * rewards["target_lane_reward"]
        )
        return reward * rewards["on_road_reward"]

    def in_target_lane(self):
        return self.vehicle.lane_index[2] == TARGET_LANE

    def target_lane_neighbours(self):
        """Return the (front, rear) cars in the target lane next to the ego."""
        ego_x = self.vehicle.position[0]
        platoon = [
            vehicle for vehicle in self.road.vehicles
            if vehicle is not self.vehicle
            and vehicle is not self.stalled_vehicle
            and vehicle.lane_index[2] == TARGET_LANE
        ]
        ahead = [v for v in platoon if v.position[0] >= ego_x]
        behind = [v for v in platoon if v.position[0] < ego_x]
        front = min(ahead, key=lambda v: v.position[0], default=None)
        rear = max(behind, key=lambda v: v.position[0], default=None)
        return front, rear

    def _info(self, obs, action=None):
        info = super()._info(obs, action)
        if self.merge_time is None and self.in_target_lane() and not self.vehicle.crashed:
            self.merge_time = self.time
        info["merged"] = self.merge_time is not None
        info["merge_time"] = self.merge_time
        info["crash_with_stalled"] = bool(self.vehicle.crashed and self.stalled_vehicle.crashed)
        return info


def make_environment(**kwargs):
    """Create the merge environment used by all scripts."""
    return CooperativeMergeEnv(**kwargs)
