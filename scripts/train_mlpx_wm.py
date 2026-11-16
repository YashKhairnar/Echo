"""Train MLPx, the world model.

    .venv/bin/python -m scripts.train_mlpx_wm --epochs 50

Saves the best model (lowest test loss) to checkpoints/MLPx/best.pt and the
run's log and plots to results/MLPx/experiments/<time>_mlpx/.
"""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import roc_auc_score
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split

from data.dataset import MultiRequestWorldModelDataset
from scripts.experiment_utils import create_run_dir, save_run_metadata
from src.cooperative_driving.models.MLPx.mlpx import MLPxWorldModel

RESULTS_ROOT = Path("results/MLPx/experiments")
CHECKPOINT_DIR = Path("checkpoints/MLPx")
LOG_FIELDS = ["epoch", "train_loss", "test_loss", "state_loss", "reward_loss",
              "crash_loss", "crash_soon_loss", "crash_auc", "crash_soon_auc"]


def compute_losses(model, batch):
    """Return the four losses and the two crash logits for one batch.

    Next state and reward are only learned on steps without a crash: after a
    crash there is no meaningful next state, and the crash heads handle it.
    """
    inputs, next_states, rewards, crashes, crash_soon = batch
    predicted_states, predicted_rewards, crash_logits, crash_soon_logits = model(inputs)
    survived = crashes.squeeze(-1) < 0.5
    losses = {
        "state_loss": F.mse_loss(predicted_states[survived], next_states[survived]),
        "reward_loss": F.mse_loss(predicted_rewards[survived], rewards[survived]),
        "crash_loss": F.binary_cross_entropy_with_logits(crash_logits, crashes),
        "crash_soon_loss": F.binary_cross_entropy_with_logits(crash_soon_logits, crash_soon),
    }
    return losses, crash_logits, crash_soon_logits


@torch.no_grad()
def evaluate(model, loader):
    """Average test losses, plus AUC of both crash heads
    (0.5 = guessing, 1.0 = every crash scored above every safe step)."""
    model.eval()
    sums, batches = {}, 0
    crash_scores, crash_soon_scores, crash_labels, crash_soon_labels = [], [], [], []
    for batch in loader:
        losses, crash_logits, crash_soon_logits = compute_losses(model, batch)
        for name, value in losses.items():
            sums[name] = sums.get(name, 0.0) + value.item()
        batches += 1
        crash_scores += crash_logits.squeeze(-1).tolist()
        crash_soon_scores += crash_soon_logits.squeeze(-1).tolist()
        crash_labels += batch[3].squeeze(-1).tolist()
        crash_soon_labels += batch[4].squeeze(-1).tolist()
    metrics = {name: value / batches for name, value in sums.items()}
    metrics["test_loss"] = sum(metrics.values())
    metrics["crash_auc"] = roc_auc_score(crash_labels, crash_scores)
    metrics["crash_soon_auc"] = roc_auc_score(crash_soon_labels, crash_soon_scores)
    return metrics


def save_plot(log_path):
    rows = list(csv.DictReader(log_path.open()))
    epochs = [int(row["epoch"]) for row in rows]
    plt.figure(figsize=(10, 6))
    for column in ["train_loss", "test_loss", "state_loss", "reward_loss", "crash_loss", "crash_soon_loss"]:
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
    parser.add_argument("--data", default="data/mergeMultiRequestData.csv")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--crash-horizon", type=int, default=3,
                        help="steps ahead for the crash-soon label (this step included)")
    args = parser.parse_args()

    dataset = MultiRequestWorldModelDataset(args.data, crash_horizon=args.crash_horizon)
    train_size = int(0.8 * len(dataset))
    train_data, test_data = random_split(
        dataset, [train_size, len(dataset) - train_size], generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size)

    run_dir = create_run_dir(RESULTS_ROOT, "mlpx")
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
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
                losses, _, _ = compute_losses(model, batch)
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
                  f"crash_auc={metrics['crash_auc']:.3f} | crash_soon_auc={metrics['crash_soon_auc']:.3f}")

            checkpoint = {"epoch": epoch, "model_state_dict": model.state_dict(),
                          "test_loss": metrics["test_loss"], "crash_horizon": args.crash_horizon,
                          "run_dir": str(run_dir)}
            torch.save(checkpoint, run_dir / "latest.pt")
            if metrics["test_loss"] < best_test_loss:
                best_test_loss = metrics["test_loss"]
                torch.save(checkpoint, run_dir / "best.pt")
                torch.save(checkpoint, CHECKPOINT_DIR / "best.pt")

    save_plot(log_path)
    print(f"Saved run to {run_dir}")


if __name__ == "__main__":
    main()
