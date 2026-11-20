"""Export matching informed-source charts for an experiment without loading all records."""

import argparse
import json
from collections import Counter
from pathlib import Path

from scripts.evaluation.create_intent_aware_dashboard import ACTION_ORDER, save_bar_chart
from scripts.utils.experiment_utils import add_experiment_argument


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_experiment_argument(parser)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    output = root / "docs" / "dataset_dashboard_charts" / f"experiment_{args.experiment}"
    output.mkdir(parents=True, exist_ok=True)
    actions, requests, behaviors, rewards = (Counter() for _ in range(4))
    with (root / "data" / f"experiment_{args.experiment}" / "intent_aware_dataset.jsonl").open() as file:
        for line in file:
            if not line.strip():
                continue
            row = json.loads(line)
            action = row["ego_action_name"]
            actions[action] += 1
            rewards[action] += float(row["reward"])
            requests[len(row["requested_vehicle_ids"])] += 1
            behaviors.update(message.get("behavior_type", "unknown")
                             for message in (row.get("messages") or {}).values())
    labels = [action for action in ACTION_ORDER if actions[action]]
    save_bar_chart(output / "actions.png", "Ego action distribution", labels,
                   [actions[action] for action in labels], color="#2563eb")
    counts = sorted(requests)
    save_bar_chart(output / "requests.png", "Requested vehicles per scenario",
                   [str(count) for count in counts], [requests[count] for count in counts],
                   color="#7c3aed", ylabel="Scenarios")
    types = sorted(behaviors)
    save_bar_chart(output / "behaviors.png", "Behavior types in requested messages",
                   types, [behaviors[kind] for kind in types], color="#059669")
    save_bar_chart(output / "rewards.png", "Mean reward by ego action", labels,
                   [rewards[action] / actions[action] for action in labels],
                   color="#d97706", ylabel="Mean reward")
    print(f"Exported Experiment {args.experiment} informed-source charts to {output}")


if __name__ == "__main__":
    main()
