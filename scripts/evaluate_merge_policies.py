"""Compare request policies in the merge scenario with the MLPx planner driving.

Every policy drives the same evaluation seeds (not used in training). The
MLPx + MCP planner chooses the driving action in all cases; only *who is
asked* differs:

- none:           never ask
- all:            ask every visible car every step (communication upper bound)
- random:         ask one random visible car every step
- random_matched: ask one random visible car with the DQN's request rate
                  (same communication budget as the DQN)
- rear:           hand-written rule: ask the rear car in the target lane
                  while the ego has not merged yet
- dqn:            the trained request policy

Example:
    .venv/bin/python -m scripts.evaluate_merge_policies \
        --dqn-checkpoint checkpoints/DQN/best.pt --communication-cost 0.05
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from src.cooperative_driving.planner.MCP import build_planner
from src.cooperative_driving.communication.request_utils import (
    MESSAGE_FEATURES,
    NO_REQUEST_ACTION,
    message_bundle,
    slot_of_vehicle,
    valid_request_actions,
    visible_neighbor_ids,
)
from src.cooperative_driving.environments.customEnv import TARGET_LANE, make_environment
from src.cooperative_driving.policies.DQN import RequestDQNPolicy

EVALUATION_SEED_OFFSET = 100_000
BITS_PER_MESSAGE = MESSAGE_FEATURES * 32  # four float32 values


class RequestRule:
    name = "none"

    def reset(self):
        pass

    def slots(self, environment, observation):
        return []


class AskAll(RequestRule):
    name = "all"

    def slots(self, environment, observation):
        return list(range(len(visible_neighbor_ids(observation))))


class AskRandom(RequestRule):
    def __init__(self, probability=1.0, name="random", seed=0):
        self.probability = probability
        self.name = name
        self.rng = random.Random(seed)

    def slots(self, environment, observation):
        visible = len(visible_neighbor_ids(observation))
        if visible and self.rng.random() < self.probability:
            return [self.rng.randrange(visible)]
        return []


class AskRear(RequestRule):
    name = "rear"

    def slots(self, environment, observation):
        if environment.vehicle.lane_index[2] == TARGET_LANE:
            return []
        _, rear = environment.target_lane_neighbours()
        slot = slot_of_vehicle(observation, rear)
        return [] if slot is None else [slot]


class AskDQN(RequestRule):
    name = "dqn"

    def __init__(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.policy = RequestDQNPolicy()
        self.policy.load_state_dict(checkpoint["online_network"])
        self.policy.eval()

    def slots(self, environment, observation):
        action = self.policy.act(observation, valid_request_actions(observation))
        return [] if action == NO_REQUEST_ACTION else [action]


def run(environment, planner, rule, episodes, cost):
    episodes_out = []
    for index in range(episodes):
        observation, _ = environment.reset(seed=EVALUATION_SEED_OFFSET + index)
        rule.reset()
        driving, requests, steps, info = 0.0, 0, 0, {}
        asked_rear = 0
        for _ in range(environment.config["duration"]):
            slots = rule.slots(environment, observation)
            _, rear = environment.target_lane_neighbours()
            asked_rear += int(slot_of_vehicle(observation, rear) in slots)
            action = planner.plan(observation, message_bundle(environment, observation, slots))
            observation, reward, terminated, truncated, info = environment.step(action)
            driving += reward
            requests += len(slots)
            steps += 1
            if terminated or truncated:
                break
        crashed = bool(info.get("crashed", False))
        episodes_out.append({
            "driving_reward": driving,
            "communication_cost": cost * requests,
            "total_reward": driving - cost * requests,
            "requests": requests,
            "rear_requests": asked_rear,
            "steps": steps,
            "crashed": crashed,
            "merged": bool(info.get("merged", False)) and not crashed,
            "merge_time": info.get("merge_time"),
        })
    return episodes_out


def summarise(episodes_out):
    def mean(key):
        return float(np.mean([e[key] for e in episodes_out]))

    totals = np.array([e["total_reward"] for e in episodes_out])
    merge_times = [e["merge_time"] for e in episodes_out if e["merged"] and e["merge_time"] is not None]
    requests = sum(e["requests"] for e in episodes_out)
    return {
        "total_reward": float(totals.mean()),
        "total_reward_se": float(totals.std(ddof=1) / np.sqrt(len(totals))),
        "driving_reward": mean("driving_reward"),
        "communication_cost": mean("communication_cost"),
        "crash_rate": mean("crashed"),
        "merge_rate": mean("merged"),
        "merge_time_s": float(np.mean(merge_times)) if merge_times else float("nan"),
        "requests_per_episode": mean("requests"),
        "bits_per_episode": mean("requests") * BITS_PER_MESSAGE,
        "share_of_requests_to_rear_car": (
            sum(e["rear_requests"] for e in episodes_out) / requests if requests else float("nan")
        ),
        "request_rate_per_step": requests / sum(e["steps"] for e in episodes_out),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--communication-cost", type=float, default=0.05)
    parser.add_argument("--dqn-checkpoint", default="checkpoints/DQN/best.pt")
    parser.add_argument("--mlpx-checkpoint", default="checkpoints/MLPx/best.pt")
    parser.add_argument("--output", default=None, help="JSON summary path")
    args = parser.parse_args()

    environment = make_environment()
    planner = build_planner(args.mlpx_checkpoint, environment)

    rules = [RequestRule(), AskAll(), AskRandom(), AskRear()]
    has_dqn = Path(args.dqn_checkpoint).exists()
    if has_dqn:
        rules.append(AskDQN(args.dqn_checkpoint))
    else:
        print(f"No DQN checkpoint at {args.dqn_checkpoint}; evaluating baselines only.")

    summary = {}
    for rule in rules:
        summary[rule.name] = summarise(run(environment, planner, rule, args.episodes, args.communication_cost))
    if has_dqn:
        matched = AskRandom(summary["dqn"]["request_rate_per_step"], "random_matched", seed=1)
        summary[matched.name] = summarise(
            run(environment, planner, matched, args.episodes, args.communication_cost)
        )

    print(
        f"\n{args.episodes} evaluation seeds, communication cost {args.communication_cost} per request"
    )
    print(
        f"  {'policy':<15} {'total':>13} {'driving':>8} {'crash':>6} {'merged':>7}"
        f" {'merge t':>8} {'asks/ep':>8} {'to rear':>8}"
    )
    for name, s in summary.items():
        print(
            f"  {name:<15} {s['total_reward']:6.2f} ± {s['total_reward_se']:4.2f}"
            f" {s['driving_reward']:8.2f} {s['crash_rate']:6.0%} {s['merge_rate']:7.0%}"
            f" {s['merge_time_s']:7.1f}s {s['requests_per_episode']:8.2f}"
            f" {s['share_of_requests_to_rear_car']:8.0%}"
        )

    output = Path(args.output) if args.output else Path("results/merge_policies") / (
        f"cost{args.communication_cost:g}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "episodes": args.episodes,
        "communication_cost": args.communication_cost,
        "dqn_checkpoint": args.dqn_checkpoint if has_dqn else None,
        "mlpx_checkpoint": args.mlpx_checkpoint,
        "policies": summary,
    }, indent=2))
    print(f"Saved summary to {output}")
    environment.close()


if __name__ == "__main__":
    main()
