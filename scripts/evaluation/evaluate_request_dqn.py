"""Evaluate the Request DQN against communication baselines.

All policies use the same fixed environment seeds. The DQN chooses one visible
vehicle or asks nobody; the planner then chooses the ego driving action. The
evaluation reports driving reward, communication-adjusted return, requests,
crashes, and truncations.

Run from the project root with:

    .venv/bin/python -m scripts.evaluation.evaluate_request_dqn \
        --experiment 1 --episodes 200
"""

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from scripts.data_collection.show_visible_neighbors import SLOT_ORDER, ordered_visible_ids, visible_neighbors
from scripts.training.train_request_dqn import (
    ASK_NOBODY,
    request_state,
    requested_message,
    select_action,
    valid_actions,
)
from scripts.utils.experiment_utils import (
    add_experiment_argument,
    experiment_checkpoint_dir,
    experiment_results_dir,
)
from src.cooperative_driving.environments.customEnv import CustomEnv
from src.cooperative_driving.models.MLPx.mlpx import MLPxWorldModel
from src.cooperative_driving.planner.MCP import MCPPlanner
from src.cooperative_driving.policies.request_dqn import RequestDQN


def all_messages(environment, visible):
    messages = [None] * len(SLOT_ORDER)
    for index, slot in enumerate(SLOT_ORDER):
        candidate = visible.get(slot)
        if candidate is not None:
            messages[index] = environment.request_neighbor_action(
                candidate["vehicle"].vehicle_id
            )
    return {"messages_by_slot": messages}


def choose_request(policy, dqn, environment, visible, rng):
    visible_ids = ordered_visible_ids(visible)
    if policy == "none" or not visible_ids:
        return ASK_NOBODY, None
    if policy == "all":
        return len(visible_ids), all_messages(environment, visible)
    if policy == "random":
        action = rng.choice([a for a in valid_actions(visible) if a != ASK_NOBODY])
        return action, requested_message(environment, visible, action)

    state = request_state(environment, visible)
    encoding = dqn.request_action_encoding
    action = select_action(dqn, state, valid_actions(visible, encoding), epsilon=0.0, rng=rng)
    return action, requested_message(environment, visible, action, encoding)


def evaluate_policy(policy, dqn, world_model, args):
    rows = []
    policy_offsets = {"dqn": 0, "none": 1000, "random": 2000, "all": 3000}
    rng = random.Random(args.seed + policy_offsets[policy])
    for episode in range(args.episodes):
        environment = CustomEnv()
        observation, _ = environment.reset(seed=args.seed + episode)
        planner = MCPPlanner(
            world_model,
            action_size=environment.action_space.n,
            planning_horizon=args.planning_horizon,
        )
        total_return = 0.0
        driving_reward = 0.0
        requests = 0
        terminated = False
        truncated = False

        for _ in range(environment.config["duration"]):
            visible = visible_neighbors(
                environment,
                args.visibility,
                args.side_window,
            )
            request_action, message = choose_request(
                policy, dqn, environment, visible, rng
            )
            requested_count = (
                len(ordered_visible_ids(visible))
                if policy == "all"
                else int(request_action != ASK_NOBODY)
            )
            driving_action = planner.plan(observation, message)
            observation, reward, terminated, truncated, _ = environment.step(
                driving_action
            )
            driving_reward += float(reward)
            requests += requested_count
            total_return += float(reward) - args.communication_cost * requested_count
            if terminated or truncated:
                break

        rows.append(
            {
                "policy": policy,
                "episode": episode,
                "return": total_return,
                "driving_reward": driving_reward,
                "requests": requests,
                "crashed": int(terminated),
                "truncated": int(truncated),
            }
        )
        environment.close()
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_experiment_argument(parser)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=100000)
    parser.add_argument("--planning-horizon", type=int, default=3)
    parser.add_argument("--communication-cost", type=float, default=0.05)
    parser.add_argument("--visibility", type=float, default=60.0)
    parser.add_argument("--side-window", type=float, default=10.0)
    parser.add_argument(
        "--policies",
        nargs="+",
        choices=["dqn", "none", "random", "all"],
        default=["dqn", "none", "random", "all"],
    )
    parser.add_argument("--dqn-checkpoint", default=None)
    parser.add_argument("--mlpx-checkpoint", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    dqn_checkpoint = args.dqn_checkpoint or str(
        experiment_checkpoint_dir(args.experiment, "RequestDQN") / "best.pt"
    )
    mlpx_checkpoint = args.mlpx_checkpoint or str(
        experiment_checkpoint_dir(args.experiment, "MLPx") / "best.pt"
    )
    output_dir = Path(args.output_dir or experiment_results_dir(args.experiment, "evaluation") / "request_dqn")
    output_dir.mkdir(parents=True, exist_ok=True)
    dqn = RequestDQN.load(dqn_checkpoint)
    world_model = MLPxWorldModel.load(mlpx_checkpoint)

    rows = []
    for policy in args.policies:
        rows.extend(evaluate_policy(policy, dqn, world_model, args))

    csv_path = output_dir / "request_dqn_evaluation.csv"
    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["policy"]].append(row)
    summary = {}
    for policy, group in sorted(grouped.items()):
        summary[policy] = {
            "episodes": len(group),
            "mean_return": float(np.mean([row["return"] for row in group])),
            "mean_driving_reward": float(
                np.mean([row["driving_reward"] for row in group])
            ),
            "mean_requests": float(np.mean([row["requests"] for row in group])),
            "crashes": int(sum(row["crashed"] for row in group)),
            "crash_rate": float(np.mean([row["crashed"] for row in group])),
            "truncations": int(sum(row["truncated"] for row in group)),
        }

    (output_dir / "request_dqn_summary.json").write_text(
        json.dumps(
            {
                "experiment": args.experiment,
                "dqn_checkpoint": dqn_checkpoint,
                "request_action_encoding": dqn.request_action_encoding,
                "mlpx_checkpoint": mlpx_checkpoint,
                "episodes": args.episodes,
                "seed_start": args.seed,
                "communication_cost": args.communication_cost,
                "summary": summary,
            },
            indent=2,
        )
    )
    print(f"Saved Request DQN evaluation to {output_dir}")
    for policy, metrics in summary.items():
        print(
            f"{policy:7s}: return={metrics['mean_return']:.3f} "
            f"driving={metrics['mean_driving_reward']:.3f} "
            f"requests={metrics['mean_requests']:.2f} "
            f"crash_rate={metrics['crash_rate']:.2%}"
        )


if __name__ == "__main__":
    main()
