import argparse
import csv
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from data.dataset import WorldModelDataset
from src.cooperative_driving.models.MLP.mlp import MLP_worldModel
from scripts.experiment_utils import create_run_dir, save_run_metadata


MLP_RESULTS_DIR = Path("results/MLP")
DEFAULT_MLP_LOG = MLP_RESULTS_DIR / "mlp_training.csv"


def get_run_log_path(requested_path, resume):
    """Keep old training logs when starting a new run."""
    path = Path(requested_path)

    if resume and path.exists():
        return path

    if not path.exists():
        return path

    version = 1
    while True:
        versioned_path = path.with_name(
            f"{path.stem}_{version}{path.suffix}"
        )
        if not versioned_path.exists():
            return versioned_path
        version += 1


def save_training_plots(log_path):
    """Save loss plots using all metrics recorded in the run log."""
    epochs = []
    metrics = {
        "train_loss": [],
        "test_loss": [],
        "test_state_loss": [],
        "test_reward_loss": [],
    }

    with log_path.open(newline="") as file:
        for row in csv.DictReader(file):
            epochs.append(int(row["epoch"]))
            for name in metrics:
                metrics[name].append(float(row[name]))

    if not epochs:
        return []

    loss_plot = log_path.with_name(f"{log_path.stem}_losses.png")
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, metrics["train_loss"], label="Train loss")
    plt.plot(epochs, metrics["test_loss"], label="Test loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("MLP World Model Training Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(loss_plot, dpi=300)
    plt.close()

    components_plot = log_path.with_name(
        f"{log_path.stem}_loss_components.png"
    )
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, metrics["test_state_loss"], label="Next-state loss")
    plt.plot(epochs, metrics["test_reward_loss"], label="Reward loss")
    plt.xlabel("Epoch")
    plt.ylabel("Test loss")
    plt.title("MLP World Model Test Loss Components")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(components_plot, dpi=300)
    plt.close()

    return [loss_plot, components_plot]


def evaluate(model, loader, state_loss_fn, reward_loss_fn, device):
    model.eval()
    total_loss = 0.0
    total_state_loss = 0.0
    total_reward_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for inputs, target_next_state, target_reward in loader:
            inputs = inputs.to(device)
            target_next_state = target_next_state.to(device)
            target_reward = target_reward.to(device)

            predicted_state, predicted_reward = model(inputs)
            state_loss = state_loss_fn(predicted_state, target_next_state)
            reward_loss = reward_loss_fn(predicted_reward, target_reward)

            total_loss += (state_loss + reward_loss).item()
            total_state_loss += state_loss.item()
            total_reward_loss += reward_loss.item()
            total_batches += 1

    total_batches = max(total_batches, 1)
    return (
        total_loss / total_batches,
        total_state_loss / total_batches,
        total_reward_loss / total_batches,
    )


def save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    train_loss,
    test_loss,
    best_test_loss,
):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "test_loss": test_loss,
            "best_test_loss": best_test_loss,
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default="data/randomPolicyData_requestPolicy.csv",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--log", default=None)
    parser.add_argument("--experiment-dir", default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir = Path("checkpoints/mlp")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    run_dir = (
        Path(args.experiment_dir)
        if args.experiment_dir
        else Path(args.log).parent
        if args.log
        else create_run_dir("results/MLP/experiments", "mlp")
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    requested_log = (
        Path(args.log)
        if args.log
        else run_dir / "mlp_training.csv"
    )
    log_path = get_run_log_path(requested_log, args.resume)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    save_run_metadata(
        run_dir,
        {
            "model": "MLP world model",
            "data": args.data,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "resume": args.resume,
            "log_path": str(log_path),
        },
    )

    append_log = args.resume is not None and log_path.exists()
    log_file = log_path.open("a" if append_log else "w", newline="")
    log_writer = csv.DictWriter(
        log_file,
        fieldnames=[
            "epoch",
            "train_loss",
            "train_state_loss",
            "train_reward_loss",
            "test_loss",
            "test_state_loss",
            "test_reward_loss",
            "learning_rate",
        ],
    )
    if not append_log:
        log_writer.writeheader()

    dataset = WorldModelDataset(args.data)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    print(train_size, test_size)

    split_generator = torch.Generator().manual_seed(42)
    train_dataset, test_dataset = random_split(
        dataset,
        [train_size, test_size],
        generator=split_generator,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = MLP_worldModel().to(device)
    state_loss_fn = nn.MSELoss()
    reward_loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    start_epoch = 0
    best_test_loss = float("inf")

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)

        # Supports both new full checkpoints and the old state_dict-only file.
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            best_test_loss = checkpoint.get("best_test_loss", float("inf"))
        else:
            model.load_state_dict(checkpoint)

        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0.0
        total_state_loss = 0.0
        total_reward_loss = 0.0

        for inputs, target_next_state, target_reward in train_loader:
            inputs = inputs.to(device)
            target_next_state = target_next_state.to(device)
            target_reward = target_reward.to(device)

            predicted_state, predicted_reward = model(inputs)
            state_loss = state_loss_fn(predicted_state, target_next_state)
            reward_loss = reward_loss_fn(predicted_reward, target_reward)
            loss = state_loss + reward_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_state_loss += state_loss.item()
            total_reward_loss += reward_loss.item()

        number_of_batches = max(len(train_loader), 1)
        train_loss = total_loss / number_of_batches
        train_state_loss = total_state_loss / number_of_batches
        train_reward_loss = total_reward_loss / number_of_batches
        test_loss, test_state_loss, test_reward_loss = evaluate(
            model,
            test_loader,
            state_loss_fn,
            reward_loss_fn,
            device,
        )
        
        save_checkpoint(
            checkpoint_dir / "latest.pt",
            model,
            optimizer,
            epoch,
            train_loss,
            test_loss,
            best_test_loss,
        )

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            save_checkpoint(
                checkpoint_dir / "best.pt",
                model,
                optimizer,
                epoch,
                train_loss,
                test_loss,
                best_test_loss,
            )

        log_writer.writerow({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_state_loss": train_state_loss,
            "train_reward_loss": train_reward_loss,
            "test_loss": test_loss,
            "test_state_loss": test_state_loss,
            "test_reward_loss": test_reward_loss,
            "learning_rate": optimizer.param_groups[0]["lr"],
        })
        log_file.flush()

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | test_loss={test_loss:.4f} | "
            f"state_loss={test_state_loss:.4f} | "
            f"reward_loss={test_reward_loss:.4f}"
        )

    log_file.close()
    plot_paths = save_training_plots(log_path)
    checkpoint_copy_dir = run_dir / "checkpoints"
    checkpoint_copy_dir.mkdir(exist_ok=True)
    for checkpoint_name in ("best.pt", "latest.pt"):
        checkpoint_path = checkpoint_dir / checkpoint_name
        if checkpoint_path.exists():
            shutil.copy2(checkpoint_path, checkpoint_copy_dir / checkpoint_name)
    print(f"Checkpoints saved in {checkpoint_dir}")
    print(f"Experiment saved to {run_dir}")
    print(f"Training metrics saved to {log_path}")
    for plot_path in plot_paths:
        print(f"Training graph saved to {plot_path}")


if __name__ == "__main__":
    main()
