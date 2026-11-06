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
            "vehicles_count":10,
            "controlled_vehicles":1,
                "observation": {
                    "type": "Kinematics",
                    "vehicles_count": 10,
                    "features": [
                        "presence", "x", "y", "vx", "vy",
                        "intent_acceleration",
                        "intent_steering",
                        "intent_lane_change",
                        "vehicle_id",
                    ],
                    "see_behind": True,
                    "normalize": True
                }
            })
        return config

    def _create_vehicles(self):
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
        

    

