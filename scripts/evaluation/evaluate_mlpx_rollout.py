"""Compare MLPx rollouts with real HighwayEnv rollouts.

The evaluator tests every discrete ego action from the same sampled state,
using constant actions for three and five steps. Results are grouped by
message mode and saved inside the selected experiment folder.

Run from the project root with:

    .venv/bin/python -m scripts.evaluation.evaluate_mlpx_rollout \
        --experiment 1 --episodes 10 --states-per-episode 3
"""

import argparse
import copy
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from scripts.data_collection.show_visible_neighbors import SLOT_ORDER, visible_neighbors
from scripts.utils.experiment_utils import (
    add_experiment_argument,
    experiment_checkpoint_dir,
    experiment_results_dir,
)
from src.cooperative_driving.environments.customEnv import CustomEnv
from src.cooperative_driving.models.MLPx.mlpx import MLPxWorldModel


def message_bundle(environment, visible, mode):
    if mode == "none":
        return None

    messages = [None] * len(SLOT_ORDER)
    for index, slot in enumerate(SLOT_ORDER):
        candidate = visible.get(slot)
        if candidate is not None:
            vehicle_id = candidate["vehicle"].vehicle_id
            messages[index] = environment.request_neighbor_action(vehicle_id)
    return {"messages_by_slot": messages}


def evaluate_state(model, environment, observation, message, horizon):
    """Evaluate all ego actions from one state and return comparison metrics."""
    action_results = []
    for action in range(environment.action_space.n):
        predicted_observation = np.asarray(observation, dtype=np.float32).reshape(-1)
        predicted_reward = 0.0
        real_reward = 0.0
        state_errors = []
        real_environment = copy.deepcopy(environment)
        real_observation = np.asarray(observation, dtype=np.float32).reshape(-1)
        complete = True

        for step in range(horizon):
            predicted_observation, step_reward = model.predict_step(
                predicted_observation, action, message
            )
            predicted_observation = predicted_observation.numpy()
            predicted_reward += float(step_reward)

            real_observation, reward, terminated, truncated, _ = (
                real_environment.step(action)
            )
            real_observation = np.asarray(real_observation, dtype=np.float32).reshape(-1)
            real_reward += float(reward)
            state_errors.append(
                float(np.mean(np.abs(predicted_observation - real_observation)))
            )

            if (terminated or truncated) and step + 1 < horizon:
                complete = False
                break

        real_environment.close()
        if not complete:
            return None

        action_results.append(
            {
                "predicted_reward": predicted_reward,
                "real_reward": real_reward,
                "state_mae": float(np.mean(state_errors)),
            }
        )

    predicted_rewards = [item["predicted_reward"] for item in action_results]
    real_rewards = [item["real_reward"] for item in action_results]
    return {
        "state_mae": float(np.mean([item["state_mae"] for item in action_results])),
        "reward_mae": float(np.mean(np.abs(np.asarray(predicted_rewards) - real_rewards))),
        "action_ranking_agreement": int(np.argmax(predicted_rewards) == np.argmax(real_rewards)),
        "predicted_best_action": int(np.argmax(predicted_rewards)),
        "real_best_action": int(np.argmax(real_rewards)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_experiment_argument(parser)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--states-per-episode", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--visibility", type=float, default=60.0)
    parser.add_argument("--side-window", type=float, default=10.0)
    parser.add_argument("--horizons", type=int, nargs="+", default=[3, 5])
    parser.add_argument(
        "--message-modes",
        nargs="+",
        choices=["none", "all"],
        default=["none", "all"],
    )
    parser.add_argument("--mlpx-checkpoint", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    checkpoint = args.mlpx_checkpoint or str(
        experiment_checkpoint_dir(args.experiment, "MLPx") / "best.pt"
    )
    output_dir = Path(args.output_dir or experiment_results_dir(args.experiment, "evaluation") / "mlpx_rollout")
    output_dir.mkdir(parents=True, exist_ok=True)
    model = MLPxWorldModel.load(checkpoint)
    rows = []
    skipped = 0

    for episode in range(args.episodes):
        environment = CustomEnv()
        observation, _ = environment.reset(seed=args.seed + episode)
        sampled_states = 0

        for timestep in range(environment.config["duration"]):
            if sampled_states >= args.states_per_episode:
                break

            visible = visible_neighbors(environment, args.visibility, args.side_window)
            for mode in args.message_modes:
                message = message_bundle(environment, visible, mode)
                for horizon in args.horizons:
                    metrics = evaluate_state(model, environment, observation, message, horizon)
                    if metrics is None:
                        skipped += 1
                        continue
                    rows.append(
                        {
                            "episode": episode,
                            "timestep": timestep,
                            "message_mode": mode,
                            "horizon": horizon,
                            **metrics,
                        }
                    )

            sampled_states += 1
            action = environment.action_space.sample()
            observation, _, terminated, truncated, _ = environment.step(action)
            if terminated or truncated:
                break
        environment.close()

    csv_path = output_dir / "mlpx_rollout_evaluation.csv"
    with csv_path.open("w", newline="") as file:
        fieldnames = list(rows[0]) if rows else [
            "episode", "timestep", "message_mode", "horizon",
            "state_mae", "reward_mae", "action_ranking_agreement",
            "predicted_best_action", "real_best_action",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["horizon"], row["message_mode"])].append(row)
    summary = {}
    for (horizon, mode), group in sorted(grouped.items()):
        key = f"{horizon}_steps_{mode}"
        summary[key] = {
            "samples": len(group),
            "state_mae": float(np.mean([row["state_mae"] for row in group])),
            "reward_mae": float(np.mean([row["reward_mae"] for row in group])),
            "action_ranking_agreement": float(
                np.mean([row["action_ranking_agreement"] for row in group])
            ),
        }

    (output_dir / "mlpx_rollout_summary.json").write_text(
        json.dumps(
            {
                "experiment": args.experiment,
                "checkpoint": checkpoint,
                "episodes": args.episodes,
                "states_per_episode": args.states_per_episode,
                "skipped_incomplete_states": skipped,
                "summary": summary,
            },
            indent=2,
        )
    )
    print(f"Saved MLPx rollout metrics to {output_dir}")
    for key, metrics in summary.items():
        print(
            f"{key}: state_mae={metrics['state_mae']:.4f} "
            f"reward_mae={metrics['reward_mae']:.4f} "
            f"agreement={metrics['action_ranking_agreement']:.2%}"
        )


if __name__ == "__main__":
    main()
