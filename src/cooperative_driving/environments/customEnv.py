from highway_env.envs.highway_env import HighwayEnv
from highway_env.utils import near_split
from highway_env.vehicle.kinematics import Vehicle
from ..vehicles.customVehicles import (
    IntentControlledVehicle,
    IntentIDMVehicle,
)


class CustomHighwayEnv(HighwayEnv):
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config.update({
            "observation": {
                "type": "TupleObservation",
                "observation_configs": [
                    {
                        "type": "Kinematics",
                        "vehicles_count": 10,
                        "features": [
                            "presence", "x", "y", "vx", "vy",
                            # Neighbor action intent: acceleration, steering, lane change.
                            "intent_acceleration", "intent_steering",
                            "intent_lane_change",
                        ],
                        "see_behind": True,
                    },
                    {
                        "type": "LidarObservation",
                        "cells": 128,
                    },
                ],
            }
        })
        return config

    def _create_vehicles(self):
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
            self.controlled_vehicles.append(vehicle)
            self.road.vehicles.append(vehicle)

            for _ in range(others):
                vehicle = IntentIDMVehicle.create_random(
                    self.road,
                    spacing=1 / self.config["vehicles_density"],
                )
                vehicle.randomize_behavior()
                self.road.vehicles.append(vehicle)

