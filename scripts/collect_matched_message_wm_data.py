"""Collect matched message-conditioned world-model transitions.

For every physical state and driving action, this creates several rows with
different request sets. The action and simulator prefix are identical across
branches, so the only changed input is the communication message.
"""

import argparse
import csv
import json
import os
import random

from src.cooperative_driving.communication.getMessage import encode_message, getMessage
from src.cooperative_driving.communication.request_utils import visible_neighbor_ids
from scripts.exploration import make_exploration
from src.cooperative_driving.environments.customEnv import ENVIRONMENTS, make_environment


COMMUNICATION_COST = 0.05
MAX_REQUEST_SLOTS = 10
DEFAULT_OUTPUTS = {
    "merge": "data/matchedMergeWMData.csv",
    "highway": "data/matchedMessageWMData.csv",
}

HEADERS = [
    "episode_id",
    "timestep",
    "branch",
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
        (vehicle for vehicle in environment.road.vehicles
         if vehicle.vehicle_id == vehicle_id),
        None,
    )


def message_data(environment, observation, request_slots):
    visible_ids = visible_neighbor_ids(observation)
    request_mask = [0] * MAX_REQUEST_SLOTS
    messages = [[0.0] * 4 for _ in range(MAX_REQUEST_SLOTS)]

    for slot in request_slots:
        if slot >= len(visible_ids):
            continue
        vehicle = vehicle_by_id(environment, visible_ids[slot])
        if vehicle is None:
            continue
        request_mask[slot] = 1
        messages[slot] = encode_message(vehicle)

    return {
        "request_mask": request_mask,
        "messages": messages,
    }


def urgent_slot(environment, observation):
    candidates = []
    for slot, vehicle_id in enumerate(visible_neighbor_ids(observation)):
        vehicle = vehicle_by_id(environment, vehicle_id)
        if vehicle is not None:
            candidates.append((getMessage(vehicle)["time_to_collision"], slot))
    return min(candidates)[1] if candidates else None


def rear_target_lane_slot(environment, observation):
    """Slot of the car behind the ego in the merge target lane, if visible."""
    if not hasattr(environment, "target_lane_neighbours"):
        return None
    _, rear = environment.target_lane_neighbours()
    if rear is None:
        return None
    visible_ids = visible_neighbor_ids(observation)
    return visible_ids.index(rear.vehicle_id) if rear.vehicle_id in visible_ids else None


def replay_prefix(environment, seed, prefix_actions):
    """Recreate exactly the same physical state in a branch environment."""
    observation, _ = environment.reset(seed=seed)
    for action in prefix_actions:
        observation, _, terminated, truncated, _ = environment.step(action)
        if terminated or truncated:
            return None
    return observation


def request_branches(environment, observation, rng):
    visible_count = len(visible_neighbor_ids(observation))
    visible_slots = list(range(visible_count))
    urgent = urgent_slot(environment, observation)
    random_slots = []
    if visible_slots:
        random_slots = sorted(
            rng.sample(visible_slots, rng.randint(1, len(visible_slots)))
        )

    rear = rear_target_lane_slot(environment, observation)
    branches = [
        ("none", []),
        ("urgent", [] if urgent is None else [urgent]),
        ("rear", [] if rear is None else [rear]),
        ("all", visible_slots),
        ("random", random_slots),
    ]
    unique = []
    for name, slots in branches:
        if slots not in [values for _, values in unique]:
            unique.append((name, slots))
    return unique


def main():
    parser = argparse.ArgumentParser(
        description="Collect matched message-conditioned WM transitions."
    )
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument(
        "--states-per-episode",
        type=int,
        default=5,
        help="Number of matched physical states sampled from each episode.",
    )
    parser.add_argument("--env", choices=sorted(ENVIRONMENTS), default="merge")
    parser.add_argument("--output", default=None)
    parser.add_argument("--communication-cost", type=float, default=COMMUNICATION_COST)
    parser.add_argument("--seed", type=int, default=20261114)
    args = parser.parse_args()
    args.output = args.output or DEFAULT_OUTPUTS[args.env]

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    first_episode_id = next_episode_id(args.output)
    file_has_data = os.path.exists(args.output) and os.path.getsize(args.output) > 0
    rng = random.Random(args.seed)

    main_environment = make_environment(args.env, render_mode="rgb_array")
    explorer = make_exploration(args.env, main_environment.action_space.n, rng)
    collected = 0

    try:
        with open(args.output, "a", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=HEADERS)
            if not file_has_data:
                writer.writeheader()

            for episode_offset in range(args.episodes):
                episode_id = first_episode_id + episode_offset
                observation, _ = main_environment.reset(seed=episode_id)
                explorer.reset()
                prefix_actions = []
                duration = main_environment.config["duration"]
                sample_count = min(args.states_per_episode, duration)
                sampled_timesteps = set(
                    rng.sample(range(duration), sample_count)
                )
                # Always include the step where the merge is attempted: it is
                # the decision the messages are meant to inform.
                if hasattr(explorer, "merge_step"):
                    sampled_timesteps.add(explorer.merge_step)

                for timestep in range(main_environment.config["duration"]):
                    driving_action = explorer.act(timestep)

                    branch_messages = []
                    if timestep in sampled_timesteps:
                        branches = request_branches(
                            main_environment, observation, rng
                        )
                        branch_messages = [
                            (
                                branch_name,
                                request_slots,
                                message_data(
                                    main_environment,
                                    observation,
                                    request_slots,
                                ),
                            )
                            for branch_name, request_slots in branches
                        ]

                    # Communication does not alter the current simulator
                    # physics, so one shared step supplies every branch's
                    # identical next state and driving reward.
                    next_observation, driving_reward, terminated, truncated, _ = (
                        main_environment.step(driving_action)
                    )

                    for branch_name, request_slots, message in branch_messages:
                        communication_cost = (
                            len(request_slots) * args.communication_cost
                        )

                        writer.writerow({
                            "episode_id": episode_id,
                            "timestep": timestep,
                            "branch": branch_name,
                            "observation_t": to_json(observation),
                            "request_action_t": to_json(request_slots),
                            "action_t": to_json(driving_action),
                            "message_t": to_json(message),
                            "observation_t+1": to_json(next_observation),
                            "driving_reward_t+1": driving_reward,
                            "communication_cost_t+1": communication_cost,
                            "reward_t+1": driving_reward - communication_cost,
                            "terminated": terminated,
                            "truncated": truncated,
                            "done": terminated or truncated,
                        })
                        collected += 1

                    observation = next_observation
                    prefix_actions.append(driving_action)
                    if terminated or truncated:
                        break

                file.flush()
                if (episode_offset + 1) % 25 == 0:
                    print(
                        f"Processed {episode_offset + 1}/{args.episodes} episodes "
                        f"({collected} matched rows)"
                    )
    finally:
        main_environment.close()

    print(f"Saved {collected} matched WM rows to {args.output}")


if __name__ == "__main__":
    main()
