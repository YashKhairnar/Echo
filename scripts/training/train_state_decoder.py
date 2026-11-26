"""Train the state decoder (55-value state -> 64 x 256 grayscale frame).

    .venv/bin/python -m scripts.training.train_state_decoder --experiment 3 --epochs 10

Uses data/experiment_N/frame_dataset.npz from collect_frame_dataset. The data
is split by episode: episode % 5 == 0 is the test set, episode % 5 == 1 is the
validation set, and the rest is for training. The best round is picked on the
validation set; the test set is only used by compare_state_decoders.

Vehicles cover only a few percent of the pixels, so vehicle pixels get a
higher weight in the loss.

``--architecture vehicle`` (default) trains the per-vehicle StateDecoder;
``--architecture mlp_deconv`` trains the MLP baseline.

Saves the model with the best validation vehicle-pixel IoU to
checkpoints/experiment_N/StateDecoder/best.pt (baseline: mlp_deconv_best.pt)
and the run log to results/experiment_N/StateDecoder/experiments/.
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
from src.cooperative_driving.models.decoder import ARCHITECTURES
from src.cooperative_driving.visualization.frames import (
    LINE_GRAY, ROAD_GREY, VEHICLE_GRAY_DIFFERENCE, iou, vehicle_mask,
)

LOG_FIELDS = ["epoch", "train_loss", "validation_loss", "validation_vehicle_iou", "validation_mae"]


def split_by_episode(episodes):
    """Return (train, validation, test) frame indices; no episode is split."""
    group = np.asarray(episodes) % 5
    return np.flatnonzero(group >= 2), np.flatnonzero(group == 1), np.flatnonzero(group == 0)


def loss_fn(logits, frames, vehicle_weight):
    """Pixel-weighted binary cross-entropy.

    ``frames`` are uint8 gray values (batch, 1, H, W). The weight mask uses
    the integer values, with the same thresholds as frames.vehicle_mask.
    """
    gray = frames.int()
    weights = 1.0 + vehicle_weight * (
        ((gray - ROAD_GREY).abs() > VEHICLE_GRAY_DIFFERENCE) & (gray < LINE_GRAY)
    ).float()
    return F.binary_cross_entropy_with_logits(logits, frames.float() / 255, weight=weights)


@torch.no_grad()
def evaluate(model, states, frames, device, vehicle_weight, batch_size=256):
    """Loss, vehicle-pixel IoU and gray MAE, with the same metric as the
    comparison script (frames.vehicle_mask with the real state)."""
    model.eval()
    losses = []
    for start in range(0, len(states), batch_size):
        batch_states = torch.as_tensor(states[start:start + batch_size]).to(device)
        batch_frames = torch.as_tensor(frames[start:start + batch_size]).to(device).unsqueeze(1)
        losses.append(loss_fn(model(batch_states), batch_frames, vehicle_weight).item())
    predicted = model.draw_batch(states, batch_size)
    ious = [iou(vehicle_mask(real, state), vehicle_mask(image, state))
            for state, real, image in zip(states, frames, predicted)]
    mae = np.abs(predicted.astype(np.int16) - frames.astype(np.int16)).mean()
    return {"validation_loss": float(np.mean(losses)), "validation_vehicle_iou": float(np.mean(ious)),
            "validation_mae": float(mae)}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_experiment_argument(parser)
    parser.add_argument("--data", default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--architecture", choices=sorted(ARCHITECTURES), default="vehicle")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--vehicle-weight", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.data = args.data or str(experiment_data_path(args.experiment, "frame_dataset.npz"))
    torch.manual_seed(args.seed)
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

    data = np.load(args.data)
    states, frames = data["states"].astype(np.float32), data["frames"]
    train_indices, validation_indices, test_indices = split_by_episode(data["episodes"])
    train_loader = DataLoader(
        TensorDataset(torch.as_tensor(states[train_indices]), torch.as_tensor(frames[train_indices])),
        batch_size=args.batch_size, shuffle=True,
    )
    validation_states, validation_frames = states[validation_indices], frames[validation_indices]
    print(f"Train frames: {len(train_indices)} | validation frames: {len(validation_indices)} | "
          f"test frames (not used here): {len(test_indices)} | device: {device}")

    run_dir = create_run_dir(experiment_results_dir(args.experiment, "StateDecoder") / "experiments",
                             f"state_decoder_{args.architecture}")
    checkpoint_dir = experiment_checkpoint_dir(args.experiment, "StateDecoder")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_name = "best.pt" if args.architecture == "vehicle" else f"{args.architecture}_best.pt"
    save_run_metadata(run_dir, {"model": "StateDecoder", **vars(args), "device": device,
                                "train_frames": len(train_indices),
                                "validation_frames": len(validation_indices),
                                "test_frames": len(test_indices)})

    model = ARCHITECTURES[args.architecture]().to(device)
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
                logits = model(batch_states.to(device))
                loss = loss_fn(logits, batch_frames.to(device).unsqueeze(1), args.vehicle_weight)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
            scheduler.step()

            metrics = evaluate(model, validation_states, validation_frames, device, args.vehicle_weight)
            log.writerow({"epoch": epoch, "train_loss": float(np.mean(train_losses)), **metrics})
            log_file.flush()
            print(f"Epoch {epoch}/{args.epochs} | train={np.mean(train_losses):.4f} | "
                  f"validation={metrics['validation_loss']:.4f} | "
                  f"IoU={metrics['validation_vehicle_iou']:.3f} | MAE={metrics['validation_mae']:.2f}",
                  flush=True)

            checkpoint = {"epoch": epoch, "model_state_dict": model.to("cpu").state_dict(),
                          "architecture": args.architecture, "model_kwargs": {},
                          "data": args.data, "run_dir": str(run_dir), **metrics}
            model.to(device)
            torch.save(checkpoint, run_dir / "latest.pt")
            if metrics["validation_vehicle_iou"] > best_iou:
                best_iou = metrics["validation_vehicle_iou"]
                torch.save(checkpoint, run_dir / "best.pt")
                torch.save(checkpoint, checkpoint_dir / best_name)

    print(f"Best validation vehicle IoU: {best_iou:.3f}. Saved run to {run_dir}")


if __name__ == "__main__":
    main()
