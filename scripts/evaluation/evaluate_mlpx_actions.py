"""Evaluate one-step WM errors by action, message presence, and crash outcome.

Streams JSONL in batches and reproduces the trainer's episode-group holdout.
"""

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from data.combined_world_model import encode_messages, episode_group, heldout_episode_groups
from scripts.policy.intent_aware_policy import ACTION_NAMES
from scripts.utils.experiment_utils import (
    add_experiment_argument, create_run_dir, experiment_checkpoint_dir,
    experiment_data_path, experiment_results_dir, save_run_metadata,
)
from src.cooperative_driving.models.MLPx.mlpx import MLPxWorldModel


def intent_message_probe(model, source, test_groups):
    """Compare genuine, missing, and swapped responses on held-out paired roots."""
    paired = defaultdict(dict)
    with source.open() as file:
        for line in file:
            row = json.loads(line)
            if (row.get('dataset_source') == 'matched_intent' and row.get('depth') == 0
                    and row.get('messages') and episode_group(row) in test_groups):
                paired[(row['episode'], row['ego_action'])][row['hidden_mode']] = row
    records, alternatives = [], []
    for modes in paired.values():
        if set(modes) != {'stay', 'merge'}:
            raise ValueError('Paired root is incomplete')
        np.testing.assert_array_equal(modes['stay']['state'], modes['merge']['state'])
        for mode, opposite in [('stay', 'merge'), ('merge', 'stay')]:
            records.append(modes[mode])
            alternatives.append(modes[opposite])
    if not records:
        return None
    report = {'records': len(records), 'groups': len(paired) // 5, 'conditions': {}}
    target = np.asarray([row['next_state'][:55] for row in records])
    for condition in ['genuine', 'none', 'swapped']:
        inputs = []
        for row, opposite in zip(records, alternatives):
            message = encode_messages(row if condition == 'genuine' else opposite) if condition != 'none' else [0.] * 56
            inputs.append(np.concatenate([np.asarray(row['state']).reshape(-1)[:model.STATE_SIZE],
                                          np.eye(5)[row['ego_action']], message]))
        with torch.no_grad():
            predictions, rewards = model(torch.tensor(np.asarray(inputs), dtype=torch.float32))
        report['conditions'][condition] = {
            'kinematic_state_mae': float(np.abs(predictions.numpy()[:, :55] - target).mean()),
            'reward_mae': float(np.abs(rewards.numpy().reshape(-1) - [r['reward'] for r in records]).mean()),
        }
    return report


def accumulate(model, records, totals):
    state_size = getattr(model, 'STATE_SIZE', 55)
    inputs = [np.concatenate((np.asarray(row["state"]).reshape(-1)[:state_size],
                              np.eye(5)[int(row["ego_action"])], encode_messages(row))) for row in records]
    with torch.no_grad():
        states, rewards = model(torch.tensor(np.asarray(inputs), dtype=torch.float32))
    for index, row in enumerate(records):
        error = states[index].numpy() - np.asarray(row["next_state"]).reshape(-1)[:state_size]
        reward_error = float(rewards[index, 0]) - float(row["reward"])
        mode = "with_messages" if row.get("messages") else "no_messages"
        outcome = "crashed" if row.get("terminated") else "noncrashed"
        for label in ("all", outcome):
            total = totals[(int(row["ego_action"]), mode, label)]
            total["records"] += 1
            total["state_mae_sum"] += float(np.abs(error).mean())
            total["state_mse_sum"] += float(np.square(error).mean())
            total["reward_mae_sum"] += abs(reward_error)
            total["reward_mse_sum"] += reward_error ** 2
            total["reward_bias_sum"] += reward_error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_experiment_argument(parser)
    parser.add_argument("--input")
    parser.add_argument("--mlpx-checkpoint")
    parser.add_argument("--split", choices=["heldout", "all"], default="heldout")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--output-root")
    parser.add_argument('--intent-probe', action='store_true')
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    source = Path(args.input or experiment_data_path(args.experiment, "combined_world_model_dataset.jsonl"))
    checkpoint = Path(args.mlpx_checkpoint or experiment_checkpoint_dir(args.experiment, "MLPx") / "best.pt")
    groups, source_hash = set(), hashlib.sha256()
    with source.open("rb") as file:
        for line in file:
            source_hash.update(line)
            if line.strip():
                groups.add(episode_group(json.loads(line)))
    test_groups = heldout_episode_groups(groups)
    model = MLPxWorldModel.load(checkpoint)
    totals = defaultdict(lambda: defaultdict(float))
    batch = []
    with source.open() as file:
        for line in file:
            if not line.strip():
                continue
            row = json.loads(line)
            if args.split == "heldout" and episode_group(row) not in test_groups:
                continue
            batch.append(row)
            if len(batch) == args.batch_size:
                accumulate(model, batch, totals)
                batch = []
        if batch:
            accumulate(model, batch, totals)
    rows = []
    for action in range(5):
        for mode in ("no_messages", "with_messages"):
            for outcome in ("all", "crashed", "noncrashed"):
                total = totals[(action, mode, outcome)]
                count = int(total["records"])
                rows.append({"action": ACTION_NAMES[action], "message_mode": mode,
                             "outcome": outcome, "records": count,
                             **{key: (total[key + "_sum"] / count if count else None)
                                for key in ("state_mae", "reward_mae", "reward_bias")},
                             **{key: (float(np.sqrt(total[key.replace("rmse", "mse") + "_sum"] / count)) if count else None)
                                for key in ("state_rmse", "reward_rmse")}})
    output = create_run_dir(Path(args.output_root or experiment_results_dir(args.experiment, "evaluation") / "mlpx_actions"), "actions")
    save_run_metadata(output, {**vars(args), "input": str(source), "data_sha256": source_hash.hexdigest(),
                              "checkpoint": str(checkpoint), "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                              "episode_groups": len(groups), "heldout_groups": len(test_groups),
                              "note": "Heldout reconstruction requires the original unchanged training dataset; observations are normalized"})
    with (output / "action_metrics.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "action_metrics.json").write_text(json.dumps(rows, indent=2))
    if args.intent_probe:
        probe = intent_message_probe(model, source, test_groups)
        (output / 'intent_message_probe.json').write_text(json.dumps(probe, indent=2))
        print('Held-out intent probe:', json.dumps(probe), flush=True)
    print(f"Saved per-action evaluation to {output}")
    for row in rows:
        if row["outcome"] == "all" and row["records"]:
            print(f"{row['action']:10s} {row['message_mode']:13s} n={row['records']:6d} "
                  f"state_mae={row['state_mae']:.4f} reward_mae={row['reward_mae']:.4f}")


if __name__ == "__main__":
    main()
