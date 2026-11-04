import csv
import json
import os
import random

from src.cooperative_driving.communication.getMessage import getMessage
from src.cooperative_driving.environments.customEnv import CustomHighwayEnv


HEADERS = [
    "episode_id",
    "timestep",
    "observation_t",
    "action_t",
    "message_t",
    "observation_t+1",
    "reward_t+1",
]


def to_json(value):
    """Convert NumPy arrays and vehicle data into CSV-safe JSON."""
    return json.dumps(value, default=lambda item: item.tolist())


def choose_message_vehicle(env):
    """Choose one neighboring vehicle to communicate with."""
    cars = [car for car in env.road.vehicles if car is not env.vehicle]
    return random.choice(cars) if cars else None


def main():
    filepath = "data/randomPolicyData.csv"
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    env = CustomHighwayEnv(render_mode="rgb_array")
    num_episodes = 10

    with open(filepath, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=HEADERS)
        writer.writeheader()

        for episode in range(num_episodes):
            observation_t, info = env.reset(seed=episode)

            for timestep in range(env.config["duration"]):
                action_t = env.action_space.sample()

                message_vehicle = choose_message_vehicle(env)
                message_t = (
                    getMessage(message_vehicle)
                    if message_vehicle is not None
                    else None
                )

                observation_t_1, reward_t_1, terminated, truncated, info = env.step(
                    action_t
                )

                writer.writerow({
                    "episode_id": episode,
                    "timestep": timestep,
                    "observation_t": to_json(observation_t),
                    "action_t": to_json(action_t),
                    "message_t": to_json(message_t),
                    "observation_t+1": to_json(observation_t_1),
                    "reward_t+1": reward_t_1,
                })

                observation_t = observation_t_1

                if terminated or truncated:
                    break

    env.close()
    print(f"Saved data to {filepath}")


if __name__ == "__main__":
    main()
