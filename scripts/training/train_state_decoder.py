"""Train the state decoder (55-value state -> 64 x 256 grayscale frame).

    .venv/bin/python -m scripts.training.train_state_decoder --experiment 3 --epochs 30

Uses data/experiment_N/frame_dataset.npz from collect_frame_dataset. Every
fifth episode is held out for testing. Vehicles cover only a few percent of
the pixels, so vehicle pixels get a higher weight in the loss.

Saves the model with the best test vehicle-pixel IoU to
checkpoints/experiment_N/StateDecoder/best.pt and the run log to
results/experiment_N/StateDecoder/experiments/<time>_state_decoder/.
"""

import argparse
import csv

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from scripts.utils.experiment_utils import (
    add_experiment_argument,
    create_run_dir,
    experiment_checkpoint_dir,
    experiment_data_path,
    experiment_results_dir,
    save_run_metadata,
)
from src.cooperative_driving.models.decoder import StateDecoder
from src.cooperative_driving.visualization.frames import ROAD_GREY

LOG_FIELDS = ["epoch", "train_loss", "test_loss", "test_vehicle_iou", "test_mae"]


def vehicle_mask(gray):
    """Torch version of frames.vehicle_mask (gray values 0-255)."""
    return ((gray - ROAD_GREY).abs() > 20) & (gray < 230)


def split_by_episode(episodes):
    test = episodes % 5 == 0
    return np.flatnonzero(~test), np.flatnonzero(test)


def loss_fn(logits, target, vehicle_weight):
    """Pixel-weighted binary cross-entropy on gray values scaled to [0, 1]."""
    weights = 1.0 + vehicle_weight * vehicle_mask(target * 255).float()
    return F.binary_cross_entropy_with_logits(logits, target, weight=weights)


@torch.no_grad()
def evaluate(model, loader, device, vehicle_weight):
    model.eval()
    losses, ious, maes = [], [], []
    for states, frames in loader:
        states, frames = states.to(device), frames.to(device).float()
        target = frames.unsqueeze(1) / 255
        logits = model(states)
        losses.append(loss_fn(logits, target, vehicle_weight).item())
        predicted = torch.sigmoid(logits) * 255
        real_mask, predicted_mask = vehicle_mask(frames.unsqueeze(1)), vehicle_mask(predicted)
        intersection = (real_mask & predicted_mask).flatten(1).sum(1).float()
        union = (real_mask | predicted_mask).flatten(1).sum(1).float()
        ious.append(torch.where(union > 0, intersection / union.clamp(min=1), torch.ones_like(union)))
        maes.append((predicted - frames.unsqueeze(1)).abs().flatten(1).mean(1))
    return {
        "test_loss": float(np.mean(losses)),
        "test_vehicle_iou": torch.cat(ious).mean().item(),
        "test_mae": torch.cat(maes).mean().item(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_experiment_argument(parser)
    parser.add_argument("--data", default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--vehicle-weight", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.data = args.data or str(experiment_data_path(args.experiment, "frame_dataset.npz"))
    torch.manual_seed(args.seed)
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

    data = np.load(args.data)
    states = torch.as_tensor(data["states"], dtype=torch.float32)
    frames = torch.as_tensor(data["frames"])  # uint8, scaled per batch
    train_indices, test_indices = split_by_episode(data["episodes"])
    train_loader = DataLoader(
        TensorDataset(states[train_indices], frames[train_indices]), batch_size=args.batch_size, shuffle=True
    )
    test_loader = DataLoader(TensorDataset(states[test_indices], frames[test_indices]), batch_size=256)
    print(f"Train frames: {len(train_indices)} | test frames: {len(test_indices)} | device: {device}")

    run_dir = create_run_dir(experiment_results_dir(args.experiment, "StateDecoder") / "experiments", "state_decoder")
    checkpoint_dir = experiment_checkpoint_dir(args.experiment, "StateDecoder")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_run_metadata(run_dir, {"model": "StateDecoder", **vars(args), "device": device,
                                "train_frames": len(train_indices), "test_frames": len(test_indices)})

    model = StateDecoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    best_iou = -1.0
    log_path = run_dir / "state_decoder_training.csv"
    with log_path.open("w", newline="") as log_file:
        log = csv.DictWriter(log_file, fieldnames=LOG_FIELDS)
        log.writeheader()
        for epoch in range(1, args.epochs + 1):
            model.train()
            train_losses = []
            for batch_states, batch_frames in train_loader:
                batch_states = batch_states.to(device)
                target = batch_frames.to(device).float().unsqueeze(1) / 255
                loss = loss_fn(model(batch_states), target, args.vehicle_weight)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
            scheduler.step()

            metrics = evaluate(model, test_loader, device, args.vehicle_weight)
            log.writerow({"epoch": epoch, "train_loss": float(np.mean(train_losses)), **metrics})
            log_file.flush()
            print(f"Epoch {epoch}/{args.epochs} | train={np.mean(train_losses):.4f} | "
                  f"test={metrics['test_loss']:.4f} | IoU={metrics['test_vehicle_iou']:.3f} | "
                  f"MAE={metrics['test_mae']:.2f}", flush=True)

            checkpoint = {"epoch": epoch, "model_state_dict": model.to("cpu").state_dict(),
                          "model_kwargs": {}, "data": args.data, "run_dir": str(run_dir), **metrics}
            model.to(device)
            torch.save(checkpoint, run_dir / "latest.pt")
            if metrics["test_vehicle_iou"] > best_iou:
                best_iou = metrics["test_vehicle_iou"]
                torch.save(checkpoint, run_dir / "best.pt")
                torch.save(checkpoint, checkpoint_dir / "best.pt")

    print(f"Best test vehicle IoU: {best_iou:.3f}. Saved run to {run_dir}")


if __name__ == "__main__":
    main()
