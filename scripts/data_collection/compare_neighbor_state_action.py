"""Compare current neighbor state with its predicted next action.

Run from the project root with:

    .venv/bin/python -m scripts.data_collection.compare_neighbor_state_action
"""

import argparse

from src.cooperative_driving.environments.customEnv import CustomEnv
from scripts.data_collection.show_visible_neighbors import visible_neighbors


def main():
    parser = argparse.ArgumentParser(
        description="Compare current state and previewed action for visible neighbors."
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--visibility", type=float, default=60.0)
    parser.add_argument("--side-window", type=float, default=10.0)
    args = parser.parse_args()

    environment = CustomEnv()
    environment.reset(seed=args.seed)
    visible = visible_neighbors(
        environment,
        visibility_distance=args.visibility,
        side_window=args.side_window,
    )

    print(f"Comparing {len(visible)} visible neighbor(s) within {args.visibility:.1f} m")

    for slot, candidate in sorted(visible.items()):
        vehicle = candidate["vehicle"]
        current = environment.get_vehicle_state(vehicle)
        predicted = environment.preview_next_action(vehicle.vehicle_id)

        print(
            f"\n{slot[0]} / {slot[1]} — "
            f"vehicle {vehicle.vehicle_id} ({current['behavior_type']})"
        )
        print(f"  distance:          {candidate['distance']:.2f} m")
        print(f"  current lane:      {current['lane_index']}")
        print(f"  current speed:     {current['speed']:.2f} m/s")
        print(f"  current target:    {current['target_lane']}")
        print(f"  current accel:     {current['acceleration']:.3f}")
        print(f"  current steering:  {current['steering']:.3f}")
        print(f"  preview target:    {predicted['target_lane']}")
        print(f"  preview accel:     {predicted['acceleration']:.3f}")
        print(f"  preview steering:  {predicted['steering']:.3f}")


if __name__ == "__main__":
    main()
