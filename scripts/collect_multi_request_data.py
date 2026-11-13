import csv
import argparse
import json
import os
import random

from src.cooperative_driving.communication.getMessage import getMessage
from src.cooperative_driving.communication.request_utils import visible_neighbor_ids
from src.cooperative_driving.environments.customEnv import CustomHighwayEnv


COMMUNICATION_COST = 0.1
OUTPUT_PATH = "data/randomMultiRequestData_ttc.csv"
MAX_REQUEST_SLOTS = 10

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


def vehicle_by_id(environment, vehicle_id):
    return next(
        (
            vehicle
            for vehicle in environment.road.vehicles
            if vehicle.vehicle_id == vehicle_id
        ),
        None,
    )


def choose_request_set(environment, observation):
    """Randomly request zero, one, or several visible neighboring cars."""
    # Independently sample visible cars so one transition can contain many.
    visible_ids = visible_neighbor_ids(observation)
    requested_slots = [
        slot
        for slot in range(len(visible_ids))
        if random.random() < 0.5
    ]

    request_mask = [0] * MAX_REQUEST_SLOTS
    messages = [[0.0] * 4 for _ in range(MAX_REQUEST_SLOTS)]

    for slot in requested_slots:
        request_mask[slot] = 1
        vehicle = vehicle_by_id(environment, visible_ids[slot])
        if vehicle is not None:
            message = getMessage(vehicle)
            messages[slot] = [
                message["intent"]["intent_acceleration"],
                message["intent"]["intent_steering"],
                message["target_lane"],
                message["time_to_collision"],
            ]

    message_data = {
        "request_mask": request_mask,
        "messages": messages,
    }
    return requested_slots, message_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--output", default=OUTPUT_PATH)
    args = parser.parse_args()

    output_path = args.output
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    first_episode_id = next_episode_id(output_path)
    file_has_data = os.path.exists(output_path) and os.path.getsize(output_path) > 0

    environment = CustomHighwayEnv(render_mode="rgb_array")
    with open(output_path, "a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=HEADERS)
        if not file_has_data:
            writer.writeheader()

        for episode in range(args.episodes):
            episode_id = first_episode_id + episode
            observation, _ = environment.reset(seed=episode_id)

            for timestep in range(environment.config["duration"]):
                request_slots, message_data = choose_request_set(
                    environment,
                    observation,
                )
                action = environment.action_space.sample()
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
