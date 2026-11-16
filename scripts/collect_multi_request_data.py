import csv
import argparse
import json
import os
import random

from scripts.exploration import MergeExploration
from src.cooperative_driving.communication.request_utils import (
    message_bundle,
    visible_neighbor_ids,
)
from src.cooperative_driving.environments.customEnv import make_environment


COMMUNICATION_COST = 0.05
OUTPUT_PATH = "data/mergeMultiRequestData.csv"

HEADERS = [
    "episode_id",
    "timestep",
    "observation_t",
    "request_action_t",
    "action_t",
    "message_t",
    "observation_t+1",
    "driving_reward_t+1",
    "communication_cost_t+1",
    "reward_t+1",
    "terminated",
    "truncated",
    "done",
]


def to_json(value):
    return json.dumps(value, default=lambda item: item.tolist())


def next_episode_id(filepath):
    if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
        return 0

    episode_ids = []
    with open(filepath, newline="") as file:
        for row in csv.DictReader(file):
            try:
                episode_ids.append(int(row["episode_id"]))
            except (KeyError, TypeError, ValueError):
                continue
    return max(episode_ids, default=-1) + 1


def choose_request_set(environment, observation):
    """Randomly request zero, one, or several visible neighbouring cars.

    Each visible car is asked independently with probability 0.5, so the
    world model sees the same kinds of situations with and without each
    car's message.
    """
    visible_count = len(visible_neighbor_ids(observation))
    requested_slots = [slot for slot in range(visible_count) if random.random() < 0.5]
    return requested_slots, message_bundle(environment, observation, requested_slots)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--output", default=OUTPUT_PATH)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    output_path = args.output
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    first_episode_id = next_episode_id(output_path)
    file_has_data = os.path.exists(output_path) and os.path.getsize(output_path) > 0

    environment = make_environment(render_mode="rgb_array")
    explorer = MergeExploration(environment.action_space.n, rng=random.Random(args.seed))
    with open(output_path, "a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=HEADERS)
        if not file_has_data:
            writer.writeheader()

        for episode in range(args.episodes):
            episode_id = first_episode_id + episode
            observation, _ = environment.reset(seed=episode_id)
            explorer.reset()

            for timestep in range(environment.config["duration"]):
                request_slots, message_data = choose_request_set(
                    environment,
                    observation,
                )
                action = explorer.act(timestep)
                next_observation, driving_reward, terminated, truncated, _ = (
                    environment.step(action)
                )
                communication_cost = len(request_slots) * COMMUNICATION_COST

                # Store both reward components and the communication-aware total.
                writer.writerow({
                    "episode_id": episode_id,
                    "timestep": timestep,
                    "observation_t": to_json(observation),
                    "request_action_t": to_json(request_slots),
                    "action_t": to_json(action),
                    "message_t": to_json(message_data),
                    "observation_t+1": to_json(next_observation),
                    "driving_reward_t+1": driving_reward,
                    "communication_cost_t+1": communication_cost,
                    "reward_t+1": driving_reward - communication_cost,
                    "terminated": terminated,
                    "truncated": truncated,
                    "done": terminated or truncated,
                })

                observation = next_observation
                if terminated or truncated:
                    break

            file.flush()

    environment.close()
    print(f"Saved multi-request data to {output_path}")


if __name__ == "__main__":
    main()
