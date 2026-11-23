"""Train MLPx, the world model.

    .venv/bin/python -m scripts.training.train_mlpx_wm --epochs 50

Use ``--architecture per_vehicle`` to train PerVehicleMLPx (one shared
encoder per car, car-matched messages and a crash head) instead of the flat
MLP. Use ``--checkpoint-name`` to keep the old best.pt.

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
from src.cooperative_driving.models.MLPx.per_vehicle import ARCHITECTURES

LOG_FIELDS = ["epoch", "train_loss", "test_loss", "state_loss", "reward_loss", "crash_loss"]


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
    """Return next-state, reward and (if the model has one) crash losses.

    The crash loss is binary cross-entropy on the "terminated" label. It is
    reported apart, so ``test_loss`` (state + reward) stays comparable
    between architectures.
    """
    inputs, next_states, rewards = batch[:3]
    if hasattr(model, "forward_all"):
        predicted_states, predicted_rewards, crash_logits = model.forward_all(inputs)
    else:
        (predicted_states, predicted_rewards), crash_logits = model(inputs), None
    losses = {
        "state_loss": F.mse_loss(predicted_states, next_states),
        "reward_loss": F.mse_loss(predicted_rewards, rewards),
    }
    if crash_logits is not None and len(batch) > 3:
        losses["crash_loss"] = F.binary_cross_entropy_with_logits(crash_logits, batch[3])
    return losses


def training_loss(losses, crash_weight):
    return losses["state_loss"] + losses["reward_loss"] + crash_weight * losses.get("crash_loss", 0.0)


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
    metrics["test_loss"] = metrics["state_loss"] + metrics["reward_loss"]
    metrics.setdefault("crash_loss", "")
    return metrics


def save_plot(log_path):
    rows = list(csv.DictReader(log_path.open()))
    epochs = [int(row["epoch"]) for row in rows]
    plt.figure(figsize=(10, 6))
    for column in ["train_loss", "test_loss", "state_loss", "reward_loss", "crash_loss"]:
        if rows and rows[0].get(column) not in (None, ""):
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
    parser.add_argument("--architecture", choices=sorted(ARCHITECTURES), default="flat",
                        help="flat: MLPx; per_vehicle: PerVehicleMLPx")
    parser.add_argument("--hidden-size", type=int, default=None,
                        help="Hidden width (default: 256 for flat, 128 for per_vehicle)")
    parser.add_argument("--crash-weight", type=float, default=0.1,
                        help="Weight of the crash loss (per_vehicle only)")
    parser.add_argument("--checkpoint-name", default="best.pt",
                        help="File name in checkpoints/experiment_N/MLPx/")
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
    save_run_metadata(run_dir, {"model": ARCHITECTURES[args.architecture].__name__,
                                "message_slot_encoding": "spatial_v1",
                                "state_size": dataset.state_size,
                                "state_encoding": 'kinematics_ego_control_v1' if dataset.state_size == 57 else 'kinematics_v1',
                                "train_records": len(train_indices), "heldout_records": len(test_indices),
                                "heldout_groups": sorted(test_groups, key=str), **vars(args)})

    model_kwargs = {"state_size": dataset.state_size}
    if args.hidden_size is not None:
        model_kwargs["hidden_size"] = args.hidden_size
    model = ARCHITECTURES[args.architecture](**model_kwargs)
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
                loss = training_loss(losses, args.crash_weight)
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
            checkpoint['architecture'] = args.architecture
            if args.architecture == "per_vehicle":
                checkpoint['hidden_size'] = model.hidden_size
            torch.save(checkpoint, run_dir / "latest.pt")
            if metrics["test_loss"] < best_test_loss:
                best_test_loss = metrics["test_loss"]
                torch.save(checkpoint, run_dir / "best.pt")
                torch.save(checkpoint, checkpoint_dir / args.checkpoint_name)

    save_plot(log_path)
    print(f"Saved run to {run_dir}")


if __name__ == "__main__":
    main()
