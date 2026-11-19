"""Enumerate request subsets and the rule-based ego action for each subset.

For n visible vehicles, this evaluates every non-empty request subset from
size 1 through n. With eight visible vehicles, that is 2**8 - 1 = 255 cases.

Run from the project root with:

    .venv/bin/python -m scripts.data_collection.enumerate_message_requests --steps 3
"""

import argparse
import copy
import itertools
import json

from scripts.policy.intent_aware_policy import ACTION_NAMES, choose_action
from scripts.data_collection.show_visible_neighbors import ordered_visible_ids, visible_neighbors
from src.cooperative_driving.environments.customEnv import CustomEnv


def jsonable(value):
    """Convert numpy values and tuples into JSON-compatible values."""
    if hasattr(value, "tolist"):
        return jsonable(value.tolist())
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def scenario_rows(environment, visibility_distance, side_window, include_none=False):
    visible = visible_neighbors(
        environment,
        visibility_distance=visibility_distance,
        side_window=side_window,
    )
    visible_ids = ordered_visible_ids(visible)
    all_messages = {
        vehicle_id: environment.request_neighbor_action(vehicle_id)
        for vehicle_id in visible_ids
    }

    rows = []
    request_sizes = range(1, len(visible_ids) + 1)
    if include_none:
        rows.append((tuple(), {}))

    for request_size in request_sizes:
        for requested_ids in itertools.combinations(visible_ids, request_size):
            messages = {
                vehicle_id: all_messages[vehicle_id]
                for vehicle_id in requested_ids
            }
            rows.append((requested_ids, messages))

    scenarios = []
    for requested_ids, messages in rows:
        action, explanation = choose_action(
            environment,
            visibility_distance=visibility_distance,
            side_window=side_window,
            messages=messages,
        )
        # Evaluate this request/action scenario without changing the real rollout.
        scenario_environment = copy.deepcopy(environment)
        next_observation, reward, terminated, truncated, info = (
            scenario_environment.step(action)
        )

        scenarios.append({
            "requested_vehicle_ids": list(requested_ids),
            "requested_messages": messages,
            "action": action,
            "action_name": ACTION_NAMES[action],
            "explanation": explanation,
            "next_observation": next_observation,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "info": info,
        })

    return visible_ids, scenarios


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--visibility", type=float, default=60.0)
    parser.add_argument("--side-window", type=float, default=10.0)
    parser.add_argument("--include-none", action="store_true")
    parser.add_argument("--output", default=None, help="optional JSON output path")
    args = parser.parse_args()

    environment = CustomEnv()
    environment.reset(seed=args.seed)
    dataset = []

    for timestep in range(args.steps):
        visible_ids, scenarios = scenario_rows(
            environment,
            visibility_distance=args.visibility,
            side_window=args.side_window,
            include_none=args.include_none,
        )

        action_counts = {}
        for scenario in scenarios:
            name = scenario["action_name"]
            action_counts[name] = action_counts.get(name, 0) + 1

        print(
            f"step={timestep:02d} visible={len(visible_ids)} "
            f"scenarios={len(scenarios)} actions={action_counts}"
        )
        for scenario in scenarios:
            print(
                f"  request={scenario['requested_vehicle_ids']} "
                f"-> {scenario['action_name']}"
            )

        # Advance one real rollout using all available messages.
        full_messages = {
            vehicle_id: environment.request_neighbor_action(vehicle_id)
            for vehicle_id in visible_ids
        }
        full_action, _ = choose_action(
            environment,
            visibility_distance=args.visibility,
            side_window=args.side_window,
            messages=full_messages,
        )
        _, reward, terminated, truncated, _ = environment.step(full_action)

        dataset.append({
            "timestep": timestep,
            "visible_vehicle_ids": visible_ids,
            "visible_vehicle_slots": visible_ids,
            "scenarios": scenarios,
            "rollout_action": full_action,
            "rollout_reward": reward,
            "terminated": terminated,
            "truncated": truncated,
        })

        if terminated or truncated:
            break

    if args.output:
        with open(args.output, "w") as file:
            json.dump(jsonable(dataset), file, indent=2)
        print(f"Wrote {len(dataset)} timestep(s) to {args.output}")


if __name__ == "__main__":
    main()
