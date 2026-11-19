"""Collect rule-based informed-action transitions.

Each record has the form:

    state + requested messages + ego action
        -> next state + reward + terminated/truncated

For every timestep, all non-empty request subsets are evaluated on copied
environments. Use ``--include-none`` to add the uninformed scenario too.

Run from the project root with:

        .venv/bin/python -m scripts.data_collection.collect_intent_aware_dataset \
        --episodes 10 --output data/intent_aware_dataset.jsonl
"""

import argparse
import json
import os

from scripts.data_collection.enumerate_message_requests import jsonable, scenario_rows
from scripts.policy.intent_aware_policy import choose_action
from src.cooperative_driving.environments.customEnv import CustomEnv


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--visibility", type=float, default=60.0)
    parser.add_argument("--side-window", type=float, default=10.0)
    parser.add_argument("--include-none", action="store_true")
    parser.add_argument(
        "--output",
        default="data/intent_aware_dataset.jsonl",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    environment = CustomEnv()
    total_records = 0
    total_crashes = 0
    total_terminated = 0
    total_truncated = 0
    actual_episodes_terminated = 0
    actual_episodes_truncated = 0
    actual_rollout_steps = 0

    with open(args.output, "w") as file:
        for episode in range(args.episodes):
            observation, _ = environment.reset(seed=args.seed + episode)
            max_steps = args.steps or environment.config["duration"]
            episode_terminated = False
            episode_truncated = False

            for timestep in range(max_steps):
                visible_ids, scenarios = scenario_rows(
                    environment,
                    visibility_distance=args.visibility,
                    side_window=args.side_window,
                    include_none=args.include_none,
                )

                for scenario in scenarios:
                    record = {
                        "dataset_source": "intent_aware",
                        "episode": episode,
                        "timestep": timestep,
                        "goal_lane_id": environment.goal_lane_id,
                        "state": observation,
                        "visible_vehicle_ids": visible_ids,
                        "visible_vehicle_slots": visible_ids,
                        "requested_vehicle_ids": scenario["requested_vehicle_ids"],
                        "messages": (
                            scenario["requested_messages"]
                            if scenario["requested_vehicle_ids"]
                            else None
                        ),
                        "ego_action": scenario["action"],
                        "ego_action_name": scenario["action_name"],
                        "next_state": scenario["next_observation"],
                        "reward": scenario["reward"],
                        "terminated": scenario["terminated"],
                        "truncated": scenario["truncated"],
                        "info": scenario["info"],
                    }
                    file.write(json.dumps(jsonable(record)) + "\n")

                    total_records += 1
                    total_crashes += int(scenario["terminated"])
                    total_terminated += int(scenario["terminated"])
                    total_truncated += int(scenario["truncated"])

                # Advance the real rollout using all currently visible messages.
                full_messages = {
                    vehicle_id: environment.request_neighbor_action(vehicle_id)
                    for vehicle_id in visible_ids
                }
                action, _ = choose_action(
                    environment,
                    visibility_distance=args.visibility,
                    side_window=args.side_window,
                    messages=full_messages,
                )
                observation, _, terminated, truncated, _ = environment.step(action)
                actual_rollout_steps += 1
                episode_terminated = terminated
                episode_truncated = truncated

                if terminated or truncated:
                    break

            actual_episodes_terminated += int(episode_terminated)
            actual_episodes_truncated += int(episode_truncated)

    crash_rate = total_crashes / total_records if total_records else 0.0
    print(f"Wrote {total_records} records to {args.output}")
    print(f"Terminated/crashed scenarios: {total_terminated}")
    print(f"Truncated scenarios: {total_truncated}")
    print(f"Counterfactual scenario crash rate: {crash_rate:.2%}")
    print(f"Actual rollout steps: {actual_rollout_steps}")
    print(f"Actual rollout crashes: {actual_episodes_terminated}")
    print(f"Actual rollout truncations: {actual_episodes_truncated}")
    print(
        "Actual rollout crash rate: "
        f"{actual_episodes_terminated / args.episodes:.2%}"
    )


if __name__ == "__main__":
    main()
