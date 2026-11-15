"""Compare MLPx five-step predictions with real HighwayEnv rollouts."""

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from src.cooperative_driving.communication.getMessage import getMessage
from src.cooperative_driving.communication.request_utils import visible_neighbor_ids
from src.cooperative_driving.environments.customEnv import CustomHighwayEnv
from src.cooperative_driving.models.MLPx.mlpx import MLPxWorldModel


MAX_REQUEST_SLOTS = 10
ACTION_NAMES = {
    0: "LANE_LEFT",
    1: "IDLE",
    2: "LANE_RIGHT",
    3: "FASTER",
    4: "SLOWER",
}


def load_model(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = MLPxWorldModel()
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    model.eval()
    return model


def vehicle_by_id(environment, vehicle_id):
    return next(
        (vehicle for vehicle in environment.road.vehicles
         if vehicle.vehicle_id == vehicle_id),
        None,
    )


def message_bundle(environment, observation, mode):
    visible_ids = visible_neighbor_ids(observation)
    if mode == "none":
        slots = []
    elif mode == "all":
        slots = list(range(len(visible_ids)))
    else:
        candidates = []
        for slot, vehicle_id in enumerate(visible_ids):
            vehicle = vehicle_by_id(environment, vehicle_id)
            if vehicle is not None:
                candidates.append((getMessage(vehicle)["time_to_collision"], slot))
        slots = [] if not candidates else [min(candidates)[1]]

    request_mask = [0] * MAX_REQUEST_SLOTS
    messages = [[0.0] * 4 for _ in range(MAX_REQUEST_SLOTS)]
    for slot in slots:
        vehicle = vehicle_by_id(environment, visible_ids[slot])
        if vehicle is None:
            continue
        message = getMessage(vehicle)
        request_mask[slot] = 1
        messages[slot] = [
            message["intent"]["intent_acceleration"],
            message["intent"]["intent_steering"],
            message["target_lane"],
            message["time_to_collision"],
        ]
    return request_mask, messages


@torch.no_grad()
def predicted_rollout(model, observation, action, request_mask, messages, horizon):
    current = np.asarray(observation, dtype=np.float32).reshape(-1)
    total_reward = 0.0
    for _ in range(horizon):
        current, reward = model.predict_step(
            current, action, request_mask, messages
        )
        current = current.cpu().numpy()
        total_reward += float(reward.item())
    return current, total_reward


def real_rollout(seed, prefix_actions, action, horizon, communication_cost, request_count):
    environment = CustomHighwayEnv(render_mode="rgb_array")
    try:
        observation, _ = environment.reset(seed=seed)
        for prefix_action in prefix_actions:
            observation, _, terminated, truncated, _ = environment.step(prefix_action)
            if terminated or truncated:
                return None

        total_reward = 0.0
        for _ in range(horizon):
            observation, driving_reward, terminated, truncated, _ = environment.step(action)
            total_reward += float(driving_reward) - communication_cost * request_count
            if terminated or truncated:
                break
        return observation, total_reward
    finally:
        environment.close()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate MLPx five-step rollouts against the real environment."
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--states-per-episode", type=int, default=3)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--communication-cost", type=float, default=0.05)
    parser.add_argument("--message-mode", choices=["none", "urgent", "all"], default="none")
    parser.add_argument("--checkpoint", default="checkpoints/MLPx/best.pt")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    model = load_model(args.checkpoint)
    rng = random.Random(args.seed)
    errors = []
    ranking_matches = 0
    ranking_total = 0

    for episode in range(args.episodes):
        environment = CustomHighwayEnv(render_mode="rgb_array")
        try:
            observation, _ = environment.reset(seed=args.seed + episode)
            duration = environment.config["duration"]
            sample_steps = set(rng.sample(
                range(duration), min(args.states_per_episode, duration)
            ))
            prefix_actions = []

            for timestep in range(duration):
                if timestep in sample_steps:
                    request_mask, messages = message_bundle(
                        environment, observation, args.message_mode
                    )
                    request_count = sum(request_mask)
                    predicted_returns = []
                    real_returns = []
                    state_errors = []
                    reward_errors = []

                    for action in range(5):
                        predicted_state, predicted_reward = predicted_rollout(
                            model, observation, action, request_mask, messages,
                            args.horizon,
                        )
                        real_result = real_rollout(
                            args.seed + episode, prefix_actions, action,
                            args.horizon, args.communication_cost, request_count,
                        )
                        if real_result is None:
                            continue
                        real_state, real_reward = real_result
                        predicted_returns.append(predicted_reward)
                        real_returns.append(real_reward)
                        state_errors.append(float(np.mean(np.abs(
                            predicted_state - np.asarray(real_state).reshape(-1)
                        ))))
                        reward_errors.append(abs(predicted_reward - real_reward))

                    if len(real_returns) == 5:
                        predicted_best = int(np.argmax(predicted_returns))
                        real_best = int(np.argmax(real_returns))
                        ranking_matches += int(predicted_best == real_best)
                        ranking_total += 1
                        errors.append({
                            "episode": episode + 1,
                            "timestep": timestep,
                            "state_mae": float(np.mean(state_errors)),
                            "reward_mae": float(np.mean(reward_errors)),
                            "predicted_best_action": predicted_best,
                            "real_best_action": real_best,
                            "predicted_best_return": predicted_returns[predicted_best],
                            "real_best_return": real_returns[real_best],
                        })

                action = environment.action_space.sample()
                observation, _, terminated, truncated, _ = environment.step(action)
                prefix_actions.append(action)
                if terminated or truncated:
                    break
        finally:
            environment.close()

    if not errors:
        raise RuntimeError("No complete five-step rollouts were collected.")

    mean_state_mae = np.mean([row["state_mae"] for row in errors])
    mean_reward_mae = np.mean([row["reward_mae"] for row in errors])
    print(f"Evaluated {len(errors)} matched {args.horizon}-step states")
    print(f"Message mode: {args.message_mode}")
    print(f"Mean next-state MAE: {mean_state_mae:.6f}")
    print(f"Mean {args.horizon}-step reward MAE: {mean_reward_mae:.6f}")
    print(
        f"Action-ranking agreement: {ranking_matches}/{ranking_total} "
        f"({ranking_matches / max(ranking_total, 1):.1%})"
    )
    print("Predicted best-action counts:", Counter(
        ACTION_NAMES[row["predicted_best_action"]] for row in errors
    ))
    print("Real best-action counts:", Counter(
        ACTION_NAMES[row["real_best_action"]] for row in errors
    ))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(errors[0]))
            writer.writeheader()
            writer.writerows(errors)
        print(f"Detailed results saved to {output_path}")


if __name__ == "__main__":
    main()
