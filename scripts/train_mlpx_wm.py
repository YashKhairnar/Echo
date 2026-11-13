import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from data.dataset import MultiRequestWorldModelDataset
from src.cooperative_driving.models.MLPx.mlpx import MLPxWorldModel
from scripts.experiment_utils import create_run_dir, save_run_metadata


MLPX_RESULTS_ROOT = Path("results/MLPx/experiments")


def evaluate(model, loader, state_loss_fn, reward_loss_fn):
    model.eval()
    total_loss = total_state_loss = total_reward_loss = 0.0
    batches = 0
    with torch.no_grad():
        for inputs, next_states, rewards in loader:
            predicted_states, predicted_rewards = model(inputs)
            state_loss = state_loss_fn(predicted_states, next_states)
            reward_loss = reward_loss_fn(predicted_rewards, rewards)
            total_loss += (state_loss + reward_loss).item()
            total_state_loss += state_loss.item()
            total_reward_loss += reward_loss.item()
            batches += 1
    batches = max(batches, 1)
    return total_loss / batches, total_state_loss / batches, total_reward_loss / batches


def save_plots(log_path):
    rows = list(csv.DictReader(log_path.open(newline="")))
    if not rows:
        return
    epochs = [int(row["epoch"]) for row in rows]
    train = [float(row["train_loss"]) for row in rows]
    test = [float(row["test_loss"]) for row in rows]
    state = [float(row["test_state_loss"]) for row in rows]
    reward = [float(row["test_reward_loss"]) for row in rows]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train, label="Train loss")
    plt.plot(epochs, test, label="Test loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("MLPx World Model Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(log_path.with_name("mlpx_training_losses.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, state, label="Next-state loss")
    plt.plot(epochs, reward, label="Reward loss")
    plt.xlabel("Epoch")
    plt.ylabel("Test loss")
    plt.title("MLPx Test Loss Components")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(log_path.with_name("mlpx_training_loss_components.png"), dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/randomMultiRequestData_ttc.csv")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--experiment-dir", default=None)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    dataset = MultiRequestWorldModelDataset(args.data)
    if len(dataset) < 2:
        raise ValueError("Collect at least two transitions before training MLPx.")

    resume_checkpoint = None
    if args.resume:
        resume_checkpoint = torch.load(
            args.resume,
            map_location="cpu",
            weights_only=False,
        )

    if args.experiment_dir:
        run_dir = Path(args.experiment_dir)
    elif resume_checkpoint and resume_checkpoint.get("run_dir"):
        run_dir = Path(resume_checkpoint["run_dir"])
    else:
        run_dir = create_run_dir(MLPX_RESULTS_ROOT, "mlpx")
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "mlpx_training.csv"
    checkpoint_dir = Path("checkpoints/MLPx")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    save_run_metadata(run_dir, {
        "model": "MLPx multi-request world model",
        "data": args.data,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "input_size": MLPxWorldModel.INPUT_SIZE,
        "resume": args.resume,
    })

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(
        dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    model = MLPxWorldModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    state_loss_fn = nn.MSELoss()
    reward_loss_fn = nn.MSELoss()
    best_test_loss = float("inf")
    start_epoch = 0

    if args.resume:
        checkpoint = torch.load(
            args.resume,
            map_location="cpu",
            weights_only=False,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_test_loss = checkpoint.get("best_test_loss", float("inf"))
        print(f"Resumed MLPx from {args.resume} at epoch {start_epoch}")

    append_log = args.resume is not None and log_path.exists()
    with log_path.open("a" if append_log else "w", newline="") as log_file:
        writer = csv.DictWriter(log_file, fieldnames=[
            "epoch", "train_loss", "test_loss", "test_state_loss", "test_reward_loss"
        ])
        if not append_log:
            writer.writeheader()

        for epoch in range(start_epoch, args.epochs):
            # Train MLPx to predict both the next observation and total reward.
            model.train()
            total_loss = 0.0
            batches = 0
            for inputs, next_states, rewards in train_loader:
                predicted_states, predicted_rewards = model(inputs)
                state_loss = state_loss_fn(predicted_states, next_states)
                reward_loss = reward_loss_fn(predicted_rewards, rewards)
                loss = state_loss + reward_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                batches += 1

            train_loss = total_loss / max(batches, 1)
            test_loss, test_state_loss, test_reward_loss = evaluate(
                model, test_loader, state_loss_fn, reward_loss_fn
            )
            writer.writerow({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "test_loss": test_loss,
                "test_state_loss": test_state_loss,
                "test_reward_loss": test_reward_loss,
            })
            log_file.flush()
            print(
                f"Epoch {epoch + 1}/{args.epochs} | "
                f"train_loss={train_loss:.6f} | test_loss={test_loss:.6f}"
            )

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "test_loss": test_loss,
                "best_test_loss": best_test_loss,
                "run_dir": str(run_dir),
            }
            # Keep latest and best checkpoints available for resume and inference.
            torch.save(checkpoint, checkpoint_dir / "latest.pt")
            torch.save(checkpoint, run_dir / "latest.pt")
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                checkpoint["best_test_loss"] = best_test_loss
                torch.save(checkpoint, checkpoint_dir / "best.pt")
                torch.save(checkpoint, run_dir / "best.pt")

    save_plots(log_path)
    print(f"MLPx experiment saved to {run_dir}")


if __name__ == "__main__":
    main()
