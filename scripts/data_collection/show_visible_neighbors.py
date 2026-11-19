"""Show the vehicles visible to the ego vehicle.

The script considers the ego lane and its immediate left/right lanes. It keeps
at most one vehicle in each relative slot and ignores vehicles farther than
the visibility radius.

Run from the project root with:

    .venv/bin/python -m scripts.data_collection.show_visible_neighbors
"""

import argparse

import numpy as np

from src.cooperative_driving.environments.customEnv import CustomEnv


SLOT_ORDER = [
    ("left", "front"),
    ("left", "beside"),
    ("left", "rear"),
    ("ego_lane", "front"),
    ("ego_lane", "rear"),
    ("right", "front"),
    ("right", "beside"),
    ("right", "rear"),
]


def relative_slot(ego, vehicle, side_window):
    """Return (lane_name, longitudinal_slot), or None if out of scope."""
    ego_lane = ego.lane_index[2]
    vehicle_lane = vehicle.lane_index[2]
    lane_delta = vehicle_lane - ego_lane

    if lane_delta == -1:
        lane_name = "left"
    elif lane_delta == 0:
        lane_name = "ego_lane"
    elif lane_delta == 1:
        lane_name = "right"
    else:
        return None

    longitudinal_delta = vehicle.position[0] - ego.position[0]

    if abs(longitudinal_delta) <= side_window and lane_name != "ego_lane":
        longitudinal_slot = "beside"
    elif longitudinal_delta > side_window:
        longitudinal_slot = "front"
    elif longitudinal_delta < -side_window:
        longitudinal_slot = "rear"
    elif longitudinal_delta > 0:
        longitudinal_slot = "front"
    else:
        longitudinal_slot = "rear"

    # The ego lane can contain a vehicle in front or behind, but not beside.
    if lane_name == "ego_lane" and longitudinal_slot == "beside":
        longitudinal_slot = "front" if longitudinal_delta > 0 else "rear"

    return lane_name, longitudinal_slot


def visible_neighbors(environment, visibility_distance, side_window):
    ego = environment.vehicle
    candidates = {}

    for vehicle in environment.road.vehicles:
        if vehicle is ego:
            continue

        distance = float(np.linalg.norm(vehicle.position - ego.position))
        if distance > visibility_distance:
            continue

        slot = relative_slot(ego, vehicle, side_window)
        if slot is None:
            continue

        # Keep the closest vehicle for each lane/relative-position slot.
        if slot not in candidates or distance < candidates[slot]["distance"]:
            candidates[slot] = {
                "vehicle": vehicle,
                "distance": distance,
            }

    return candidates


def ordered_visible_ids(visible):
    """Return visible vehicle IDs in the fixed spatial slot order."""
    return [
        visible[slot]["vehicle"].vehicle_id
        for slot in SLOT_ORDER
        if slot in visible
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Show visible front, side, and rear neighboring vehicles."
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--visibility", type=float, default=60.0)
    parser.add_argument(
        "--side-window",
        type=float,
        default=10.0,
        help="Longitudinal distance used to classify a vehicle as beside the ego.",
    )
    args = parser.parse_args()

    environment = CustomEnv()
    environment.reset(seed=args.seed)
    visible = visible_neighbors(
        environment,
        visibility_distance=args.visibility,
        side_window=args.side_window,
    )

    print(f"Visibility radius: {args.visibility:.1f} m")
    print(f"Visible vehicles: {len(visible)} / maximum 8")

    for slot in SLOT_ORDER:
        candidate = visible.get(slot)
        if candidate is None:
            print(f"{slot[0]:8s} {slot[1]:6s}: None")
            continue

        vehicle = candidate["vehicle"]
        print(
            f"{slot[0]:8s} {slot[1]:6s}: "
            f"id={vehicle.vehicle_id}, "
            f"type={environment.get_behavior_type(vehicle)}, "
            f"distance={candidate['distance']:.2f} m, "
            f"position={vehicle.position}"
        )


if __name__ == "__main__":
    main()
