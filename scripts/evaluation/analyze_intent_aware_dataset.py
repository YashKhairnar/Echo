"""Summarize an intent-aware JSONL dataset.

Run from the project root with:

        .venv/bin/python -m scripts.evaluation.analyze_intent_aware_dataset \
        --input data/intent_aware_dataset.jsonl
"""

import argparse
import json
from collections import Counter, defaultdict


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--input",
        default="data/intent_aware_dataset.jsonl",
    )
    args = parser.parse_args()

    records = 0
    rewards = []
    actions = Counter()
    request_sizes = Counter()
    behavior_types = Counter()
    crashes_by_action = Counter()
    crashes_by_request_size = Counter()
    rewards_by_action = defaultdict(list)

    with open(args.input) as file:
        for line in file:
            if not line.strip():
                continue

            record = json.loads(line)
            records += 1

            action = record["ego_action_name"]
            request_size = len(record["requested_vehicle_ids"])
            reward = float(record["reward"])
            crashed = bool(record["terminated"])

            actions[action] += 1
            request_sizes[request_size] += 1
            rewards.append(reward)
            rewards_by_action[action].append(reward)

            if crashed:
                crashes_by_action[action] += 1
                crashes_by_request_size[request_size] += 1

            messages = record.get("messages") or {}
            for message in messages.values():
                behavior_types[message.get("behavior_type", "unknown")] += 1

    print(f"Records: {records}")
    print(f"Mean reward: {sum(rewards) / records:.3f}" if records else "Mean reward: n/a")
    print(f"Terminated/crashed records: {sum(crashes_by_action.values())}")
    print(
        "Truncated records: "
        f"{sum(1 for line in open(args.input) if json.loads(line).get('truncated', False))}"
    )

    print("\nEgo actions:")
    for action, count in actions.most_common():
        mean_reward = sum(rewards_by_action[action]) / count
        print(f"  {action:10s} {count:5d} ({count / records:6.2%}) mean_reward={mean_reward:.3f}")

    print("\nRequested vehicle count:")
    for count, frequency in sorted(request_sizes.items()):
        print(f"  {count:2d} messages: {frequency:5d} ({frequency / records:6.2%})")

    print("\nMessage behavior types:")
    for behavior_type, count in behavior_types.most_common():
        print(f"  {behavior_type:10s} {count:5d}")

    print("\nCrashes by action:")
    if crashes_by_action:
        for action, count in crashes_by_action.most_common():
            print(f"  {action:10s} {count:5d}")
    else:
        print("  none")

    print("\nCrashes by request count:")
    if crashes_by_request_size:
        for count, frequency in sorted(crashes_by_request_size.items()):
            print(f"  {count:2d} messages: {frequency:5d}")
    else:
        print("  none")


if __name__ == "__main__":
    main()
