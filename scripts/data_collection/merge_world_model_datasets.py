"""Merge intent-aware and random no-message JSONL datasets.

Run from the project root with:

    .venv/bin/python -m scripts.data_collection.merge_world_model_datasets --experiment 1
"""

import argparse
import json
import os
from collections import Counter

from scripts.utils.experiment_utils import add_experiment_argument, experiment_data_path


def read_records(path, fallback_source):
    with open(path) as file:
        for line in file:
            if not line.strip():
                continue
            record = json.loads(line)
            record.setdefault("dataset_source", fallback_source)
            yield record


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_experiment_argument(parser)
    parser.add_argument(
        "--intent",
        default=None,
    )
    parser.add_argument(
        "--random",
        dest="random_dataset",
        default=None,
    )
    parser.add_argument(
        "--output",
        default=None,
    )
    args = parser.parse_args()
    args.intent = args.intent or str(
        experiment_data_path(args.experiment, "intent_aware_dataset.jsonl")
    )
    args.random_dataset = args.random_dataset or str(
        experiment_data_path(args.experiment, "random_no_message_dataset.jsonl")
    )
    args.output = args.output or str(
        experiment_data_path(args.experiment, "combined_world_model_dataset.jsonl")
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    source_counts = Counter()
    action_counts = Counter()
    terminated = Counter()
    truncated = Counter()
    total = 0
    global_episode_offset = 0

    with open(args.output, "w") as output:
        for path, fallback_source in (
            (args.intent, "intent_aware"),
            (args.random_dataset, "random_no_message"),
        ):
            source_episodes = set()
            records = []
            for record in read_records(path, fallback_source):
                source_episodes.add(record.get("episode"))
                records.append(record)

            episode_map = {
                episode: global_episode_offset + index
                for index, episode in enumerate(sorted(source_episodes))
            }
            global_episode_offset += len(episode_map)

            for record in records:
                source = record["dataset_source"]
                record["source_episode"] = record.get("episode")
                record["global_episode"] = episode_map.get(
                    record.get("episode"), global_episode_offset
                )
                record["record_id"] = total
                output.write(json.dumps(record) + "\n")

                total += 1
                source_counts[source] += 1
                action_counts[record.get("ego_action_name", "unknown")] += 1
                terminated[source] += int(bool(record.get("terminated")))
                truncated[source] += int(bool(record.get("truncated")))

    print(f"Wrote {total} records to {args.output}")
    print("Records by source:")
    for source, count in source_counts.items():
        print(f"  {source:20s} {count:6d}")
    print("Actions:")
    for action, count in action_counts.most_common():
        print(f"  {action:12s} {count:6d}")
    print("Terminated/crashed records by source:")
    for source, count in terminated.items():
        print(f"  {source:20s} {count:6d}")
    print("Truncated records by source:")
    for source, count in truncated.items():
        print(f"  {source:20s} {count:6d}")


if __name__ == "__main__":
    main()
