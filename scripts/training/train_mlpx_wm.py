"""Train MLPx, the world model.

    .venv/bin/python -m scripts.training.train_mlpx_wm --epochs 50

Saves the best model (lowest test loss) to
checkpoints/experiment_N/MLPx/best.pt and the run's log and plots to
results/experiment_N/MLPx/experiments/<time>_mlpx/.
"""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from data.combined_world_model import CombinedWorldModelDataset
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
    args = parser.parse_args()

    if args.data is None:
        args.data = str(
            experiment_data_path(args.experiment, "combined_world_model_dataset.jsonl")
        )
    results_root = experiment_results_dir(args.experiment, "MLPx") / "experiments"
    checkpoint_dir = experiment_checkpoint_dir(args.experiment, "MLPx")

    dataset = CombinedWorldModelDataset(args.data)
    groups = sorted(set(dataset.groups), key=str)
    test_groups = set(groups[::5])
    train_indices = [index for index, group in enumerate(dataset.groups) if group not in test_groups]
    test_indices = [index for index, group in enumerate(dataset.groups) if group in test_groups]
    train_data = Subset(dataset, train_indices)
    test_data = Subset(dataset, test_indices)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size)

    run_dir = create_run_dir(results_root, "mlpx")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_run_metadata(run_dir, {"model": "MLPx (per-vehicle)", **vars(args)})

    model = MLPxWorldModel()
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
                          "data": args.data}
            torch.save(checkpoint, run_dir / "latest.pt")
            if metrics["test_loss"] < best_test_loss:
                best_test_loss = metrics["test_loss"]
                torch.save(checkpoint, run_dir / "best.pt")
                torch.save(checkpoint, checkpoint_dir / "best.pt")

    save_plot(log_path)
    print(f"Saved run to {run_dir}")


if __name__ == "__main__":
    main()
