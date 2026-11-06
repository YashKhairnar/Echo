import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from data.dataset import WorldModelDataset
from src.cooperative_driving.models.MLP.mlp import MLP_worldModel


def evaluate(model, loader, state_loss_fn, reward_loss_fn, device):
    model.eval()
    total_loss = 0.0
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
            total_batches += 1

    return total_loss / max(total_batches, 1)


def save_checkpoint(path, model, optimizer, epoch, train_loss, test_loss):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "test_loss": test_loss,
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/randomPolicyData.csv")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir = Path("checkpoints/mlp")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

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

        train_loss = total_loss / max(len(train_loader), 1)
        test_loss = evaluate(
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
            )

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"test_loss={test_loss:.4f}"
        )

    print(f"Checkpoints saved in {checkpoint_dir}")


if __name__ == "__main__":
    main()
