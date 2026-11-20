"""Collect uninformed random-action World Model transitions.

The ego requests no vehicles and takes a random discrete action. Episodes are
recorded through crashes or time-limit truncation, so the dataset includes
negative examples for learning unsafe outcomes.

Run from the project root with:

        .venv/bin/python -m scripts.data_collection.collect_random_no_message_dataset \
        --episodes 1000 \
        --experiment 1
"""

import argparse
import json
import os
import random
from collections import Counter

from scripts.data_collection.enumerate_message_requests import jsonable
from scripts.data_collection.show_visible_neighbors import ordered_visible_ids, visible_neighbors
from scripts.policy.exploration import FASTER, IDLE, LANE_LEFT, LANE_RIGHT, SLOWER
from scripts.utils.experiment_utils import add_experiment_argument, experiment_data_path
from src.cooperative_driving.environments.customEnv import CustomEnv


ACTION_NAMES = {
    LANE_LEFT: "LANE_LEFT",
    IDLE: "IDLE",
    LANE_RIGHT: "LANE_RIGHT",
    FASTER: "FASTER",
    SLOWER: "SLOWER",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_experiment_argument(parser)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--visibility", type=float, default=60.0)
    parser.add_argument("--side-window", type=float, default=10.0)
    parser.add_argument(
        "--output",
        default=None,
    )
    args = parser.parse_args()
    args.output = args.output or str(
        experiment_data_path(args.experiment, "random_no_message_dataset.jsonl")
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    environment = CustomEnv()
    rng = random.Random(args.seed)
    action_counts = Counter()
    total_records = 0
    total_crashes = 0
    total_truncated = 0

    with open(args.output, "w") as file:
        for episode in range(args.episodes):
            observation, _ = environment.reset(seed=args.seed + episode)

            for timestep in range(environment.config["duration"]):
                visible = visible_neighbors(
                    environment,
                    visibility_distance=args.visibility,
                    side_window=args.side_window,
                )
                visible_ids = ordered_visible_ids(visible)

                action = rng.randrange(environment.action_space.n)
                next_observation, reward, terminated, truncated, info = (
                    environment.step(action)
                )

                record = {
                    "dataset_source": "random_no_message",
                    "episode": episode,
                    "timestep": timestep,
                    "goal_lane_id": environment.goal_lane_id,
                    "state": observation,
                    "visible_vehicle_ids": visible_ids,
                    "visible_vehicle_slots": visible_ids,
                    "requested_vehicle_ids": [],
                    "messages": None,
                    "ego_action": action,
                    "ego_action_name": ACTION_NAMES.get(action, str(action)),
                    "next_state": next_observation,
                    "reward": reward,
                    "terminated": terminated,
                    "truncated": truncated,
                    "info": info,
                }
                file.write(json.dumps(jsonable(record)) + "\n")

                total_records += 1
                total_crashes += int(terminated)
                total_truncated += int(truncated)
                action_counts[ACTION_NAMES.get(action, str(action))] += 1
                observation = next_observation

                if terminated or truncated:
                    break

    print(f"Wrote {total_records} records to {args.output}")
    print(f"Crashes: {total_crashes}")
    print(f"Truncations: {total_truncated}")
    print(f"Crash rate: {total_crashes / total_records:.2%}")
    print("Actions:")
    for action, count in action_counts.most_common():
        print(f"  {action:10s} {count:6d} ({count / total_records:.2%})")


if __name__ == "__main__":
    main()
