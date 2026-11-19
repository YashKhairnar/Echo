"""Rule-based ego policy using neighboring vehicles' preview messages.

Run from the project root with:

    .venv/bin/python -m scripts.policy.intent_aware_policy
"""

import argparse

from scripts.policy.exploration import FASTER, IDLE, LANE_LEFT, LANE_RIGHT, SLOWER
from scripts.data_collection.show_visible_neighbors import visible_neighbors
from src.cooperative_driving.environments.customEnv import CustomEnv


ACTION_NAMES = {
    IDLE: "IDLE",
    LANE_LEFT: "LANE_LEFT",
    LANE_RIGHT: "LANE_RIGHT",
    FASTER: "FASTER",
    SLOWER: "SLOWER",
}


def time_to_collision(ego_speed, other_speed, gap):
    """Return seconds until collision, or infinity if not closing."""
    closing_speed = ego_speed - other_speed
    if closing_speed <= 0 or gap <= 0:
        return float("inf")
    return gap / closing_speed


def front_vehicle_within_caution_distance(
    environment, visible, caution_distance=40.0, lane_id=None
):
    """Return whether a front vehicle is close enough to stop accelerating."""
    ego = environment.vehicle
    lane_id = ego.lane_index[2] if lane_id is None else lane_id
    for candidate in visible.values():
        vehicle = candidate["vehicle"]
        delta_x = vehicle.position[0] - ego.position[0]
        if (
            vehicle.lane_index[2] == lane_id
            and delta_x > 10.0
            and delta_x < caution_distance
        ):
            return True
    return False


def lane_risk(environment, lane_id, visible, side_window=10.0, messages=None):
    """Estimate collision risk for one lane using state and intent messages."""
    ego = environment.vehicle
    risk = 0.0
    reasons = []

    for candidate in visible.values():
        vehicle = candidate["vehicle"]
        state = environment.get_vehicle_state(vehicle)
        if messages is None:
            message = environment.preview_next_action(vehicle.vehicle_id)
        else:
            # An unrequested vehicle contributes geometry, but not hidden intent.
            message = messages.get(vehicle.vehicle_id, {})
        delta_x = vehicle.position[0] - ego.position[0]

        # A neighbor planning to enter this lane is a possible lane-change conflict.
        target_lane = message.get("target_lane")
        target_lane_id = target_lane[2] if target_lane is not None else None
        entering_lane = (
            target_lane_id == lane_id
            and vehicle.lane_index[2] != lane_id
        )
        if entering_lane and abs(delta_x) <= 30.0:
            risk += 5.0
            reasons.append(f"vehicle {vehicle.vehicle_id} intends to enter lane {lane_id}")

        if vehicle.lane_index[2] != lane_id:
            continue

        # A vehicle beside the ego blocks an immediate lane change.
        if abs(delta_x) <= side_window:
            risk += 10.0
            reasons.append(f"vehicle {vehicle.vehicle_id} is beside ego")
            continue

        if delta_x > side_window:
            gap = delta_x
            ttc = time_to_collision(ego.speed, vehicle.speed, gap)
            if gap < 25.0 or ttc < 2.0:
                risk += 6.0
                reasons.append(
                    f"vehicle {vehicle.vehicle_id} is too close ahead (TTC={ttc:.1f}s)"
                )
            if message.get("acceleration", 0.0) < -1.5:
                risk += 3.0
                reasons.append(f"vehicle {vehicle.vehicle_id} is braking")
        else:
            gap = -delta_x
            ttc = time_to_collision(vehicle.speed, ego.speed, gap)
            if gap < 25.0 or ttc < 2.0:
                risk += 7.0
                reasons.append(
                    f"vehicle {vehicle.vehicle_id} is too close behind (TTC={ttc:.1f}s)"
                )
            if (
                message.get("behavior_type") == "aggressive"
                and message.get("acceleration", 0.0) > 0.5
            ):
                risk += 3.0
                reasons.append(f"aggressive vehicle {vehicle.vehicle_id} is accelerating")

    return risk, reasons


def choose_action(
    environment,
    visibility_distance=60.0,
    side_window=10.0,
    messages=None,
    goal_lane_id=None,
):
    """Choose a conservative ego action using neighbor intent messages."""
    visible = visible_neighbors(
        environment,
        visibility_distance=visibility_distance,
        side_window=side_window,
    )
    ego = environment.vehicle
    ego_lane = ego.lane_index[2]

    current_risk, current_reasons = lane_risk(
        environment, ego_lane, visible, side_window, messages
    )

    front_danger = any(
        "ahead" in reason or "braking" in reason
        for reason in current_reasons
    )

    front_caution = front_vehicle_within_caution_distance(
        environment, visible
    )

    # When a front vehicle enters the caution range, first try a safe adjacent
    # lane. This creates natural lane changes as the ego catches slower traffic.
    if front_danger or front_caution:
        lane_options = []
        for lane_id, action in (
            (ego_lane - 1, LANE_LEFT),
            (ego_lane + 1, LANE_RIGHT),
        ):
            if 0 <= lane_id < environment.config["lanes_count"]:
                risk, reasons = lane_risk(
                    environment, lane_id, visible, side_window, messages
                )
                lane_options.append((
                    risk,
                    action,
                    reasons,
                    front_vehicle_within_caution_distance(
                        environment, visible, lane_id=lane_id
                    ),
                ))

        safe_options = [
            option for option in lane_options
            if option[0] == 0 and not option[3]
        ]
        if safe_options:
            risk, action, reasons, _ = min(
                safe_options, key=lambda option: option[0]
            )
            return action, {
                "visible_count": len(visible),
                "current_lane_risk": current_risk,
                "current_reasons": current_reasons,
                "selected_lane_risk": risk,
                "selected_reasons": reasons or [
                    "front vehicle is close and adjacent lane is safe"
                ],
            }

        return SLOWER, {
            "visible_count": len(visible),
            "current_lane_risk": current_risk,
            "current_reasons": current_reasons,
            "selected_lane_risk": None,
            "selected_reasons": [
                "no safe adjacent lane: slow down"
            ],
        }

    # If the current lane is dangerous, move to the safest available neighbor lane.
    if current_risk > 0:
        lane_options = []
        for lane_id, action in (
            (ego_lane - 1, LANE_LEFT),
            (ego_lane + 1, LANE_RIGHT),
        ):
            if 0 <= lane_id < environment.config["lanes_count"]:
                risk, reasons = lane_risk(
                    environment, lane_id, visible, side_window, messages
                )
                lane_options.append((risk, action, reasons))

        safe_options = [option for option in lane_options if option[0] == 0]
        if safe_options:
            risk, action, reasons = min(safe_options, key=lambda option: option[0])
            return action, {
                "visible_count": len(visible),
                "current_lane_risk": current_risk,
                "current_reasons": current_reasons,
                "selected_lane_risk": risk,
                "selected_reasons": reasons,
            }

        # No safe lane change: reduce speed and create more distance.
        return SLOWER, {
            "visible_count": len(visible),
            "current_lane_risk": current_risk,
            "current_reasons": current_reasons,
            "selected_lane_risk": None,
            "selected_reasons": ["no safe adjacent lane"],
        }

    # When the lane is safe, accelerate toward the configured target speed.
    desired_speed = environment.config.get("ego_target_speed", 27.0)
    if (
        ego.speed < desired_speed - 0.5
        and ego.target_speed <= desired_speed
    ):
        return FASTER, {
            "visible_count": len(visible),
            "current_lane_risk": current_risk,
            "current_reasons": ["current lane is safe"],
            "selected_lane_risk": current_risk,
            "selected_reasons": ["ego is below target speed"],
        }

    return IDLE, {
        "visible_count": len(visible),
        "current_lane_risk": current_risk,
        "current_reasons": ["current lane is safe"],
        "selected_lane_risk": current_risk,
        "selected_reasons": [],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--visibility", type=float, default=60.0)
    parser.add_argument("--side-window", type=float, default=10.0)
    args = parser.parse_args()

    environment = CustomEnv()
    environment.reset(seed=args.seed)

    for timestep in range(args.steps):
        action, explanation = choose_action(
            environment,
            visibility_distance=args.visibility,
            side_window=args.side_window,
        )
        visible = visible_neighbors(
            environment,
            visibility_distance=args.visibility,
            side_window=args.side_window,
        )

        print(
            f"step={timestep:02d} action={ACTION_NAMES[action]:10s} "
            f"visible={explanation['visible_count']} "
            f"current_risk={explanation['current_lane_risk']:.1f}"
        )
        for slot, candidate in sorted(visible.items()):
            vehicle = candidate["vehicle"]
            state = environment.get_vehicle_state(vehicle)
            message = environment.preview_next_action(vehicle.vehicle_id)
            print(
                f"  vehicle {vehicle.vehicle_id}: "
                f"slot={slot[0]}/{slot[1]}, "
                f"type={state['behavior_type']}, "
                f"distance={candidate['distance']:.1f}m, "
                f"speed={state['speed']:.1f}m/s, "
                f"lane={state['lane_index'][2]}, "
                f"intends_lane={message.get('target_lane')}, "
                f"intends_accel={message.get('acceleration', 0.0):.2f}"
            )
        for reason in explanation["current_reasons"] + explanation["selected_reasons"]:
            print(f"  - {reason}")

        _, reward, terminated, truncated, _ = environment.step(action)
        print(f"  reward={reward:.3f}\n")

        if terminated or truncated:
            print("Episode ended.")
            break


if __name__ == "__main__":
    main()
