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
        # The observation is a fixed 10-row kinematics vector. Each row also
        # contains the neighboring car's intent and vehicle ID.
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
        

    

