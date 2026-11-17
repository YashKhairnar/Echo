"""Where does the request DQN ask, and does it ask at the moments that matter?

Drives the test roads with a trained DQN and the MLPx planner, and puts every
step in one of four situations:

    decisive        not merged yet, and the planner would choose a different
                    driving action with the rear car's message than without
                    it: asking the rear car changes what the ego does
    not decisive    not merged yet, but the rear car's message would not
                    change the driving action
    no rear car     not merged yet, and no rear car is visible to ask
    after merge     the ego is already in the target lane

For each situation it reports what the DQN chose: ask nobody, ask the rear
car, ask another target-lane car, or ask a car elsewhere.

    .venv/bin/python -m scripts.check_dqn_requests --dqn-checkpoint checkpoints/DQN/best.pt
"""

import argparse
import json
from collections import Counter, defaultdict

from scripts.evaluate_merge_policies import EVALUATION_SEED_OFFSET, AskDQN
from src.cooperative_driving.communication.request_utils import (
    message_bundle,
    slot_of_vehicle,
    vehicle_by_id,
    visible_neighbor_ids,
)
from src.cooperative_driving.environments.customEnv import TARGET_LANE, make_environment
from src.cooperative_driving.planner.MCP import build_planner

SITUATIONS = ["decisive", "not decisive", "no rear car", "after merge"]
CHOICES = ["nobody", "rear car", "front car", "other target-lane car", "car elsewhere"]


def situation_of(environment, planner, observation, rear_slot):
    if environment.vehicle.lane_index[2] == TARGET_LANE:
        return "after merge"
    if rear_slot is None:
        return "no rear car"
    without = planner.plan(observation, None)
    with_rear = planner.plan(observation, message_bundle(environment, observation, [rear_slot]))
    return "decisive" if without != with_rear else "not decisive"


def choice_of(environment, observation, slots, rear_slot, front_slot):
    if not slots:
        return "nobody"
    if slots[0] == rear_slot:
        return "rear car"
    if slots[0] == front_slot:
        return "front car"
    vehicle = vehicle_by_id(environment, visible_neighbor_ids(observation)[slots[0]])
    if vehicle is not None and vehicle.lane_index[2] == TARGET_LANE:
        return "other target-lane car"
    return "car elsewhere"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dqn-checkpoint", default="checkpoints/DQN/best.pt")
    parser.add_argument("--mlpx-checkpoint", default="checkpoints/MLPx/best.pt")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--output", default=None, help="optional JSON path")
    args = parser.parse_args()

    environment = make_environment()
    planner = build_planner(args.mlpx_checkpoint, environment)
    policy = AskDQN(args.dqn_checkpoint)
    counts = defaultdict(Counter)

    for index in range(args.episodes):
        observation, _ = environment.reset(seed=EVALUATION_SEED_OFFSET + index)
        for _ in range(environment.config["duration"]):
            front, rear = environment.target_lane_neighbours()
            rear_slot = slot_of_vehicle(observation, rear)
            front_slot = slot_of_vehicle(observation, front)
            situation = situation_of(environment, planner, observation, rear_slot)
            slots = policy.slots(environment, observation)
            counts[situation][choice_of(environment, observation, slots, rear_slot, front_slot)] += 1

            action = planner.plan(observation, message_bundle(environment, observation, slots))
            observation, _, crashed, time_up, _ = environment.step(action)
            if crashed or time_up:
                break

    total_steps = sum(sum(c.values()) for c in counts.values())
    total_questions = sum(sum(v for k, v in c.items() if k != "nobody") for c in counts.values())
    print(f"\n{args.episodes} test roads, {args.dqn_checkpoint}")
    print(f"  {'situation':<14} {'steps':>6} {'share':>6}  " + "  ".join(f"{c:>21}" for c in CHOICES)
          + f"  {'questions':>9}")
    print("  (after the merge, 'rear car' and 'front car' are the ego's own neighbours in that lane)")
    for situation in SITUATIONS:
        c = counts[situation]
        steps = sum(c.values())
        if not steps:
            continue
        questions = steps - c["nobody"]
        print(f"  {situation:<14} {steps:6d} {steps / total_steps:6.0%}  "
              + "  ".join(f"{c[choice] / steps:21.0%}" for choice in CHOICES)
              + f"  {questions / total_questions:9.0%}")
    print("  ('questions' = share of all questions asked that fall in this situation)")

    if args.output:
        with open(args.output, "w") as file:
            json.dump({"dqn_checkpoint": args.dqn_checkpoint, "episodes": args.episodes,
                       "counts": {s: dict(counts[s]) for s in SITUATIONS}}, file, indent=2)
    environment.close()


if __name__ == "__main__":
    main()
