import csv
import json
import os
import random

from src.cooperative_driving.communication.getMessage import getMessage
from src.cooperative_driving.communication.request_utils import (
    requested_vehicle,
    valid_request_actions,
)
from src.cooperative_driving.environments.customEnv import CustomHighwayEnv


HEADERS = [
    "episode_id",
    "timestep",
    "observation_t",
    "request_action_t",
    "action_t",
    "message_t",
    "observation_t+1",
    "reward_t+1",
    "terminated",
    "truncated",
    "done",
]

def to_json(value):
    """Convert NumPy arrays and vehicle data into CSV-safe JSON."""
    return json.dumps(value, default=lambda item: item.tolist())


def choose_request(env, observation):
    """Choose a random visible-neighbor slot or no request."""
    request_action = random.choice(valid_request_actions(observation))
    return request_action, requested_vehicle(env, observation, request_action)


def next_episode_id(filepath):
    """Continue episode numbering when appending to an existing dataset."""
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


def main():
    filepath = "data/randomPolicyData_requestPolicy.csv"
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    env = CustomHighwayEnv(render_mode="rgb_array")
    num_episodes = 1000
    first_episode_id = next_episode_id(filepath)
    file_has_data = os.path.exists(filepath) and os.path.getsize(filepath) > 0

    with open(filepath, "a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=HEADERS)
        if not file_has_data:
            writer.writeheader()

        for episode in range(num_episodes):
            episode_id = first_episode_id + episode
            observation_t, info = env.reset(seed=episode_id)

            for timestep in range(env.config["duration"]):
                request_action_t, message_vehicle = choose_request(
                    env,
                    observation_t,
                )
                message_t = (
                    getMessage(message_vehicle)
                    if message_vehicle is not None
                    else None
                )
                action_t = env.action_space.sample()

                observation_t_1, reward_t_1, terminated, truncated, info = env.step(
                    action_t
                )

                writer.writerow({
                    "episode_id": episode_id,
                    "timestep": timestep,
                    "request_action_t": request_action_t,
                    "observation_t": to_json(observation_t),
                    "action_t": to_json(action_t),
                    "message_t": to_json(message_t),
                    "observation_t+1": to_json(observation_t_1),
                    "reward_t+1": reward_t_1,
                    "terminated": terminated,
                    "truncated": truncated,
                    "done": terminated or truncated,
                })

                observation_t = observation_t_1

                if terminated or truncated:
                    break

            file.flush()

    env.close()
    print(f"Saved data to {filepath}")


if __name__ == "__main__":
    main()
