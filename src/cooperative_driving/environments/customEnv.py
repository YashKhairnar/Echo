import copy
import numpy as np
import gymnasium as gym

from highway_env.envs.highway_env import HighwayEnv
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.behavior import (
    IDMVehicle,
    AggressiveVehicle,
    DefensiveVehicle,
)


class CustomEnv(HighwayEnv):
    @classmethod
    def default_config(cls):
        config = super().default_config()

        config.update(
            {
                "observation": {
                    "type": "Kinematics",
                    # Ego + 10 neighboring vehicles
                    "vehicles_count": 11,
                    "features": [
                        "presence",
                        "x",
                        "y",
                        "vx",
                        "vy",
                    ],
                    "see_behind": True,
                    "normalize": True,
                },
                "action": {
                    "type": "DiscreteMetaAction",
                },
                "lanes_count": 4,
                "vehicles_count": 10,
                "aggressive_vehicles": 5,
                "defensive_vehicles": 5,
                "idm_vehicles": 0,
                "controlled_vehicles": 1,
                "ego_target_speed": 27.0,
                "goal_lane_id": None,
                "initial_lane_id": None,
                "duration": 100,
                "ego_spacing": 2,
                "vehicles_density": 1,
                "collision_reward": -1,
                "right_lane_reward": 0.1,
                "high_speed_reward": 0.4,
                "lane_change_reward": 0,
                "reward_speed_range": [20, 30],
                "normalize_reward": True,
                "offroad_terminal": False,
                "simulation_frequency": 15,
                "policy_frequency": 1,
                "other_vehicles_type": (
                    "highway_env.vehicle.behavior.IDMVehicle"
                ),
                "screen_width": 600,
                "screen_height": 150,
                "centering_position": [0.3, 0.5],
                "scaling": 5.5,
                "show_trajectories": False,
                "render_agent": True,
                "offscreen_rendering": None,
            }
        )

        return config

    def _create_vehicles(self):
        """
        Create the ego vehicle and all background vehicles.
        This method is called by HighwayEnv during reset().
        """

        self.next_vehicle_id = 0
        self.vehicle_types = {}
        self.vehicles_by_id = {}

        self.goal_lane_id = self.config["goal_lane_id"]

        # Randomize which lane has the closest lead vehicle. This creates
        # both left-escape and right-escape lane-change situations.
        self.lane_lead_offsets = [
            int(offset)
            for offset in self.np_random.permutation([30.0, 45.0, 60.0, 75.0])
        ]

        # ------------------------------------------------------------------
        # Create ego vehicle
        # ------------------------------------------------------------------

        configured_initial_lane = self.config["initial_lane_id"]
        ego_lane_id = (
            int(self.np_random.choice([0, 1, 2, 3]))
            if configured_initial_lane is None
            else int(configured_initial_lane)
        )
        ego_lane = self.road.network.get_lane(
            ("0", "1", ego_lane_id)
        )

        ego = ControlledVehicle(
            road=self.road,
            position=ego_lane.position(0.0, 0.0),
            heading=ego_lane.heading_at(0.0),
            speed=25.0,
            # Start at the current speed; the rule can issue FASTER once
            # to move toward ego_target_speed.
            target_speed=25.0,
        )
        # Make the controlled ego visually distinct in HighwayEnv's viewer.
        ego.color = (50, 200, 0)  # green

        self._add_vehicle(ego, vehicle_type="ego")

        # HighwayEnv uses the first controlled vehicle as self.vehicle
        self.controlled_vehicles = [ego]

        # ------------------------------------------------------------------
        # Create background traffic types
        # ------------------------------------------------------------------

        vehicle_types = (
            ["aggressive"] * self.config["aggressive_vehicles"]
            + ["defensive"] * self.config["defensive_vehicles"]
            + ["idm"] * self.config["idm_vehicles"]
        )

        if len(vehicle_types) != self.config["vehicles_count"]:
            raise ValueError(
                "aggressive_vehicles + defensive_vehicles + "
                "idm_vehicles must equal vehicles_count"
            )

        self.np_random.shuffle(vehicle_types)

        # ------------------------------------------------------------------
        # Create background vehicles
        # ------------------------------------------------------------------

        for index, driver_type in enumerate(vehicle_types):
            lane_id = index % self.config["lanes_count"]

            lane = self.road.network.get_lane(
                ("0", "1", lane_id)
            )

            # Stagger vehicles by lane and randomize the lead offsets per
            # episode, so the safe escape side is not always the same.
            x_position = (
                self.lane_lead_offsets[lane_id]
                + (index // self.config["lanes_count"]) * 60.0
            )

            if driver_type == "aggressive":
                vehicle_class = AggressiveVehicle
            elif driver_type == "defensive":
                vehicle_class = DefensiveVehicle
            elif driver_type == "idm":
                vehicle_class = IDMVehicle
            else:
                raise ValueError(
                    f"Unknown driver type: {driver_type}"
                )

            vehicle = vehicle_class(
                road=self.road,
                position=lane.position(x_position, 0.0),
                heading=lane.heading_at(x_position),
                speed=25.0,
                target_speed=27.0,
            )

            # HighwayEnv gives LinearVehicle subclasses the same default
            # color, so explicitly distinguish the hidden behavior types.
            vehicle.color = {
                "aggressive": (255, 165, 0),  # orange
                "defensive": (100, 200, 255),  # blue
                "idm": (200, 200, 0),  # yellow
            }[driver_type]

            # Randomize behavior parameters if desired.
            #
            # For AggressiveVehicle and DefensiveVehicle this randomizes
            # controller parameters. Remove this line if you want every
            # vehicle of the same type to behave identically.
            vehicle.randomize_behavior()

            self._add_vehicle(
                vehicle,
                vehicle_type=driver_type,
            )

    def _add_vehicle(self, vehicle, vehicle_type):
        """
        Assign a stable ID and add the vehicle to the road.
        """

        vehicle.vehicle_id = self.next_vehicle_id
        self.next_vehicle_id += 1

        self.road.vehicles.append(vehicle)

        self.vehicles_by_id[vehicle.vehicle_id] = vehicle
        self.vehicle_types[vehicle.vehicle_id] = vehicle_type

    def _update_vehicle_colors(self):
        """Keep behavior colors, except that every crashed vehicle is red."""
        for vehicle in self.road.vehicles:
            if vehicle.crashed:
                vehicle.color = (255, 100, 100)
                continue
            vehicle_type = self.vehicle_types.get(
                getattr(vehicle, "vehicle_id", None), "idm"
            )
            vehicle.color = {
                "ego": (50, 200, 0),
                "aggressive": (255, 165, 0),
                "defensive": (100, 200, 255),
                "idm": (200, 200, 0),
            }.get(vehicle_type, (200, 200, 0))

    def step(self, action):
        result = super().step(action)
        self._update_vehicle_colors()
        return result

    def get_vehicle(self, vehicle_id):
        """
        Retrieve a vehicle using its stable ID.
        """

        if vehicle_id not in self.vehicles_by_id:
            raise ValueError(
                f"Unknown vehicle_id: {vehicle_id}"
            )

        return self.vehicles_by_id[vehicle_id]

    def get_behavior_type(self, vehicle):
        """
        Return the behavior category of a vehicle.
        """

        if isinstance(vehicle, AggressiveVehicle):
            return "aggressive"

        if isinstance(vehicle, DefensiveVehicle):
            return "defensive"

        if isinstance(vehicle, IDMVehicle):
            return "idm"

        if vehicle is self.vehicle:
            return "ego"

        return vehicle.__class__.__name__

    def preview_next_action(self, vehicle_id):
        """
        Compute a neighbor's next controller action without modifying
        the real environment.

        The copied vehicle runs its actual IDM/MOBIL or
        aggressive/defensive policy.
        """

        vehicle = self.get_vehicle(vehicle_id)

        if vehicle is self.vehicle:
            raise ValueError(
                "preview_next_action() should be used for a neighbor, "
                "not the ego vehicle."
            )

        if vehicle.crashed:
            return {
                "vehicle_id": vehicle_id,
                "behavior_type": self.get_behavior_type(vehicle),
                "crashed": True,
            }

        # Copy the complete road and all vehicles.
        road_copy = copy.deepcopy(self.road)

        # Find the corresponding copied vehicle.
        copied_vehicle = next(
            copied_vehicle
            for copied_vehicle in road_copy.vehicles
            if copied_vehicle.vehicle_id == vehicle_id
        )

        # This computes the action on the copied scene.
        #
        # It performs:
        # 1. MOBIL lane-change decision
        # 2. Steering computation
        # 3. IDM/LinearVehicle acceleration computation
        copied_vehicle.act()

        return {
            "vehicle_id": copied_vehicle.vehicle_id,
            "behavior_type": self.vehicle_types[vehicle_id],
            "crashed": copied_vehicle.crashed,
            "target_lane": copied_vehicle.target_lane_index,
            "target_speed": copied_vehicle.target_speed,
            "acceleration": copied_vehicle.action["acceleration"],
            "steering": copied_vehicle.action["steering"],
        }

    def get_vehicle_state(self, vehicle):
        """
        Return the current state and controller information
        for one vehicle.
        """

        return {
            "vehicle_id": vehicle.vehicle_id,
            "behavior_type": self.get_behavior_type(vehicle),
            "position": vehicle.position.copy(),
            "velocity": vehicle.velocity.copy(),
            "speed": float(vehicle.speed),
            "heading": float(vehicle.heading),
            "lane_index": vehicle.lane_index,
            "target_lane": vehicle.target_lane_index,
            "target_speed": float(
                vehicle.target_speed
            )
            if vehicle.target_speed is not None
            else None,
            "acceleration": float(
                vehicle.action["acceleration"]
            ),
            "steering": float(
                vehicle.action["steering"]
            ),
            "crashed": bool(vehicle.crashed),
        }

    def get_all_vehicle_states(self):
        """
        Return states for every vehicle in the scene.
        """

        return [
            self.get_vehicle_state(vehicle)
            for vehicle in self.road.vehicles
        ]

    def request_neighbor_action(self, vehicle_id):
        """
        Public communication API.

        The ego requests the next controller action of a neighbor.
        """

        return self.preview_next_action(vehicle_id)
