"""Train MLPx, the world model.

    .venv/bin/python -m scripts.training.train_mlpx_wm --epochs 50

Saves the best model (lowest test loss) to
checkpoints/experiment_N/MLPx/best.pt and the run's log and plots to
results/experiment_N/MLPx/experiments/<time>_mlpx/.
"""

import argparse
import csv
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from data.combined_world_model import CombinedWorldModelDataset, heldout_episode_groups
from scripts.utils.experiment_utils import (
    add_experiment_argument,
    create_run_dir,
    experiment_checkpoint_dir,
    experiment_data_path,
    experiment_results_dir,
    save_run_metadata,
)
from src.cooperative_driving.models.MLPx.mlpx import MLPxWorldModel

LOG_FIELDS = ["epoch", "train_loss", "test_loss", "state_loss", "reward_loss"]


def balanced_sampler(records, train_indices, balance_by, seed):
    """Equal expected sampling mass per observed action(/message) stratum.

    Uses only training indices; the held-out distribution is unchanged.
    """
    if balance_by == "none":
        return None
    if balance_by not in ("action", "action-message", "action-message-stage"):
        raise ValueError("Unknown balancing mode")
    labels = [(int(records[index]["ego_action"]),
               bool(records[index].get("messages")) if balance_by.startswith("action-message") else False)
              for index in train_indices]
    if balance_by == 'action-message-stage':
        labels = [(*label, records[index].get('dataset_source', 'unknown'),
                   records[index].get('depth', 0) if records[index].get('dataset_source') == 'matched_intent' else 'ordinary')
                  for label, index in zip(labels, train_indices)]
    counts = Counter(labels)
    weights = torch.tensor([1. / counts[label] for label in labels], dtype=torch.double)
    return WeightedRandomSampler(weights, len(weights), replacement=True,
                                 generator=torch.Generator().manual_seed(seed))


def compute_losses(model, batch):
    """Return next-state and reward losses for one batch."""
    inputs, next_states, rewards = batch
    predicted_states, predicted_rewards = model(inputs)
    losses = {
        "state_loss": F.mse_loss(predicted_states, next_states),
        "reward_loss": F.mse_loss(predicted_rewards, rewards),
    }
    return losses


@torch.no_grad()
def evaluate(model, loader):
    """Return average test losses."""
    model.eval()
    sums, batches = {}, 0
    for batch in loader:
        losses = compute_losses(model, batch)
        for name, value in losses.items():
            sums[name] = sums.get(name, 0.0) + value.item()
        batches += 1
    metrics = {name: value / batches for name, value in sums.items()}
    metrics["test_loss"] = sum(metrics.values())
    return metrics


def save_plot(log_path):
    rows = list(csv.DictReader(log_path.open()))
    epochs = [int(row["epoch"]) for row in rows]
    plt.figure(figsize=(10, 6))
    for column in ["train_loss", "test_loss", "state_loss", "reward_loss"]:
        plt.plot(epochs, [float(row[column]) for row in rows], label=column)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("MLPx training")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(log_path.with_name("mlpx_training_losses.png"), dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_experiment_argument(parser)
    parser.add_argument("--data", default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--balance-by", choices=["none", "action", "action-message", "action-message-stage"], default="none",
                        help="Inverse-frequency sampling on training records only")
    args = parser.parse_args()

    if args.data is None:
        args.data = str(
            experiment_data_path(args.experiment, "combined_world_model_dataset.jsonl")
        )
    results_root = experiment_results_dir(args.experiment, "MLPx") / "experiments"
    checkpoint_dir = experiment_checkpoint_dir(args.experiment, "MLPx")

    dataset = CombinedWorldModelDataset(args.data, compact_records=True)
    test_groups = heldout_episode_groups(dataset.groups)
    train_indices = [index for index, group in enumerate(dataset.groups) if group not in test_groups]
    test_indices = [index for index, group in enumerate(dataset.groups) if group in test_groups]
    train_data = Subset(dataset, train_indices)
    test_data = Subset(dataset, test_indices)
    torch.manual_seed(args.seed)
    sampler = balanced_sampler(dataset.records, train_indices, args.balance_by, args.seed)
    train_loader = DataLoader(train_data, batch_size=args.batch_size,
                              shuffle=sampler is None, sampler=sampler)
    test_loader = DataLoader(test_data, batch_size=args.batch_size)

    run_dir = create_run_dir(results_root, "mlpx")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_run_metadata(run_dir, {"model": "MLPx (per-vehicle)",
                                "message_slot_encoding": "spatial_v1",
                                "state_size": dataset.state_size,
                                "state_encoding": 'kinematics_ego_control_v1' if dataset.state_size == 57 else 'kinematics_v1',
                                "train_records": len(train_indices), "heldout_records": len(test_indices),
                                "heldout_groups": sorted(test_groups, key=str), **vars(args)})

    model = MLPxWorldModel(state_size=dataset.state_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_test_loss = float("inf")
    log_path = run_dir / "mlpx_training.csv"
    with log_path.open("w", newline="") as log_file:
        log = csv.DictWriter(log_file, fieldnames=LOG_FIELDS)
        log.writeheader()

        for epoch in range(1, args.epochs + 1):
            model.train()
            train_loss = 0.0
            for batch in train_loader:
                losses = compute_losses(model, batch)
                loss = sum(losses.values())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)

            metrics = evaluate(model, test_loader)
            log.writerow({"epoch": epoch, "train_loss": train_loss, **metrics})
            log_file.flush()
            print(f"Epoch {epoch}/{args.epochs} | train={train_loss:.4f} | test={metrics['test_loss']:.4f} | "
                  f"state_mse={metrics['state_loss']:.4f} | reward_mse={metrics['reward_loss']:.4f}")

            checkpoint = {"epoch": epoch, "model_state_dict": model.state_dict(),
                          "test_loss": metrics["test_loss"], "run_dir": str(run_dir),
                          "data": args.data, "message_slot_encoding": "spatial_v1"}
            checkpoint['state_size'] = model.STATE_SIZE
            checkpoint['state_encoding'] = model.state_encoding
            torch.save(checkpoint, run_dir / "latest.pt")
            if metrics["test_loss"] < best_test_loss:
                best_test_loss = metrics["test_loss"]
                torch.save(checkpoint, run_dir / "best.pt")
                torch.save(checkpoint, checkpoint_dir / "best.pt")

    save_plot(log_path)
    print(f"Saved run to {run_dir}")


if __name__ == "__main__":
    main()
