"""Compare the drawn frame and the learned decoder against the real frame.

For each test state (episode % 5 == 0; not used for training or for picking
the best training round):
- real:     the grayscale render saved by collect_frame_dataset;
- drawn:    StateRenderer, which draws the state directly (no training);
- baseline: MLPDeconvDecoder, the MLP + transposed-conv decoder;
- decoded:  StateDecoder, the per-vehicle learned decoder.

Metrics: vehicle-pixel IoU against the real frame (colour ignored; lane-line
rows are left out except near the cars of the real state, see
frames.vehicle_mask; the IoU with all lane-line rows is reported too), mean
absolute gray error (MAE, 0-255) and time per frame. Saves the metrics as JSON and an example
image (rows: real, drawn, baseline, decoded). A missing baseline checkpoint
is skipped.

Run from the project root:

    .venv/bin/python -m scripts.evaluation.compare_state_decoders --experiment 3
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.training.train_state_decoder import split_by_episode
from scripts.utils.experiment_utils import (
    add_experiment_argument,
    experiment_checkpoint_dir,
    experiment_data_path,
    experiment_results_dir,
)
from src.cooperative_driving.models.decoder import load_decoder
from src.cooperative_driving.visualization import StateRenderer
from src.cooperative_driving.visualization.frames import (
    FRAME_HEIGHT, FRAME_SCALING, FRAME_WIDTH, iou, to_gray, vehicle_mask,
)


def stack_rows(images, gap=3):
    """Join same-sized gray images top to bottom with white lines between."""
    line = np.full((gap, images[0].shape[1]), 255, dtype=np.uint8)
    parts = []
    for image in images:
        parts.extend([image, line])
    return np.concatenate(parts[:-1], axis=0)


def summary(values):
    values = np.asarray(values)
    return {"mean": float(values.mean()), "median": float(np.median(values)),
            "min": float(values.min()), "share_above_0.9": float((values > 0.9).mean())}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_experiment_argument(parser)
    parser.add_argument("--data", default=None)
    parser.add_argument("--decoder-checkpoint", default=None)
    parser.add_argument("--baseline-checkpoint", default=None)
    parser.add_argument("--examples", type=int, default=6)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    data_path = args.data or experiment_data_path(args.experiment, "frame_dataset.npz")
    checkpoint_dir = experiment_checkpoint_dir(args.experiment, "StateDecoder")
    checkpoints = {
        "baseline": Path(args.baseline_checkpoint or checkpoint_dir / "mlp_deconv_best.pt"),
        "decoded": Path(args.decoder_checkpoint or checkpoint_dir / "best.pt"),
    }
    output_dir = Path(args.output_dir or experiment_results_dir(args.experiment, "StateDecoder") / "comparison")
    output_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(data_path)
    _, _, test_indices = split_by_episode(data["episodes"])
    decoders = {name: load_decoder(path) for name, path in checkpoints.items() if path.exists()}
    renderer = StateRenderer(FRAME_WIDTH, FRAME_HEIGHT, FRAME_SCALING)

    states, real_frames = data["states"][test_indices], data["frames"][test_indices]
    predicted, seconds = {}, {}
    start = time.perf_counter()
    predicted["drawn"] = np.stack([to_gray(renderer.draw(state)) for state in states])
    seconds["drawn"] = time.perf_counter() - start
    renderer.close()
    for name, decoder in decoders.items():
        start = time.perf_counter()
        predicted[name] = decoder.draw_batch(states)
        seconds[name] = time.perf_counter() - start

    names = list(predicted)
    results = {
        "test_frames": len(test_indices),
        "test_episodes": int(len(np.unique(data["episodes"][test_indices]))),
    }
    for name in names:
        ious, ious_all_rows, maes = [], [], []
        for state, real, image in zip(states, real_frames, predicted[name]):
            ious.append(iou(vehicle_mask(real, state), vehicle_mask(image, state)))
            ious_all_rows.append(iou(vehicle_mask(real), vehicle_mask(image)))
            maes.append(float(np.abs(image.astype(int) - real.astype(int)).mean()))
        results[name] = {
            "vehicle_iou": summary(ious),
            "vehicle_iou_with_lane_rows": float(np.mean(ious_all_rows)),
            "mae_gray": float(np.mean(maes)),
            "ms_per_frame": 1000 * seconds[name] / len(states),
        }

    examples = np.linspace(0, len(test_indices) - 1, args.examples).astype(int)
    columns = [stack_rows([real_frames[i], *(predicted[name][i] for name in names)]) for i in examples]
    (output_dir / "metrics.json").write_text(json.dumps(results, indent=2))
    gap = np.full((8, columns[0].shape[1]), 255, dtype=np.uint8)
    grid = np.concatenate([part for column in columns for part in (column, gap)][:-1], axis=0)
    plt.imsave(output_dir / "examples.png", grid, cmap="gray", vmin=0, vmax=255)
    print(json.dumps(results, indent=2))
    print(f"Saved metrics and examples (rows: real, {', '.join(names)}) to {output_dir}")


if __name__ == "__main__":
    main()
