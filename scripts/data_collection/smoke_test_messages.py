"""Smoke test neighbor-intent messages.

Run from the project root with:

    .venv/bin/python -m scripts.data_collection.smoke_test_messages
"""

import argparse
from pprint import pprint

import numpy as np

from src.cooperative_driving.environments.customEnv import CustomEnv


def main():
    parser = argparse.ArgumentParser(
        description="Query and print preview messages from neighboring vehicles."
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    environment = CustomEnv()
    environment.reset(seed=args.seed)

    ego = environment.vehicle
    neighbors = [
        vehicle for vehicle in environment.road.vehicles
        if vehicle is not ego
    ]

    print(f"Ego vehicle id: {ego.vehicle_id}")
    print(f"Neighbor count: {len(neighbors)}")

    # Save positions to confirm previewing is non-mutating.
    positions_before = {
        vehicle.vehicle_id: vehicle.position.copy()
        for vehicle in environment.road.vehicles
    }

    for vehicle in neighbors:
        print(
            f"\nRequesting message from vehicle {vehicle.vehicle_id} "
            f"({environment.get_behavior_type(vehicle)})"
        )
        message = environment.request_neighbor_action(vehicle.vehicle_id)
        pprint(message)

    for vehicle in environment.road.vehicles:
        np.testing.assert_array_equal(
            vehicle.position,
            positions_before[vehicle.vehicle_id],
            err_msg="preview_next_action changed the real environment",
        )

    print("\nSmoke test passed: preview messages were returned and the real scene was unchanged.")


if __name__ == "__main__":
    main()
