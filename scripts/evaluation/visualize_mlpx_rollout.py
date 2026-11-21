"""Show MLPx predictions as frames, next to what really happened.

From a start state, MLPx imagines the next states open-loop (each prediction
is fed back as the next input), with the same ego actions that the real
environment takes. Each step shows three rows:

    real            the real render
    MLPx (drawn)    the predicted state, drawn by StateRenderer
    MLPx (decoded)  the predicted state, turned into a frame by StateDecoder

It also reports, per prediction step, the vehicle-pixel IoU between the
drawn prediction and the real frame, and the state MSE.

Run from the project root:

    .venv/bin/python -m scripts.evaluation.visualize_mlpx_rollout --experiment 3 --horizon 5
"""

import argparse
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.evaluation.compare_state_decoders import stack_rows
from scripts.policy.intent_aware_policy import choose_action
from scripts.utils.experiment_utils import (
    add_experiment_argument,
    experiment_checkpoint_dir,
    experiment_results_dir,
)
from src.cooperative_driving.models.decoder import load_decoder
from src.cooperative_driving.models.MLPx.mlpx import MLPxWorldModel
from src.cooperative_driving.visualization import StateRenderer
from src.cooperative_driving.visualization.frames import (
    FRAME_HEIGHT, FRAME_SCALING, FRAME_WIDTH, iou, make_frame_environment, to_gray, vehicle_mask,
)


def rollout(environment, world_model, renderer, decoder, seed, warmup, horizon, rng):
    """Return per-step (real, drawn, decoded) frames, IoUs and state MSEs."""
    observation, _ = environment.reset(seed=seed)
    for _ in range(warmup):
        observation, _, terminated, truncated, _ = environment.step(choose_action(environment)[0])
        if terminated or truncated:
            return None
    predicted = np.asarray(observation, dtype=np.float32).reshape(-1)
    frames, ious, mses = [], [], []
    for _ in range(horizon):
        action = choose_action(environment)[0] if rng.random() > 0.3 else rng.randrange(5)
        predicted = world_model.predict_step(predicted, action)[0].numpy()
        observation, _, terminated, truncated, _ = environment.step(action)
        real = to_gray(environment.render())
        drawn = to_gray(renderer.draw(predicted))
        frames.append((real, drawn, decoder.draw(predicted)))
        ego_y = np.asarray(observation)[0][2]  # real ego y: leave out the real lane-line rows
        ious.append(iou(vehicle_mask(real, ego_y), vehicle_mask(drawn, ego_y)))
        mses.append(float(np.mean((predicted - np.asarray(observation).reshape(-1)) ** 2)))
        if terminated or truncated:
            break
    return frames, ious, mses


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_experiment_argument(parser)
    parser.add_argument("--mlpx-checkpoint", default=None)
    parser.add_argument("--decoder-checkpoint", default=None)
    parser.add_argument("--seed", type=int, default=10_000)  # away from training seeds
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    mlpx_path = args.mlpx_checkpoint or experiment_checkpoint_dir(args.experiment, "MLPx") / "best.pt"
    decoder_path = args.decoder_checkpoint or experiment_checkpoint_dir(args.experiment, "StateDecoder") / "best.pt"
    output_dir = Path(args.output_dir or experiment_results_dir(args.experiment, "StateDecoder") / "mlpx_rollout")
    output_dir.mkdir(parents=True, exist_ok=True)

    world_model, decoder = MLPxWorldModel.load(mlpx_path), load_decoder(decoder_path)
    environment = make_frame_environment()
    renderer = StateRenderer(FRAME_WIDTH, FRAME_HEIGHT, FRAME_SCALING)
    rng = random.Random(args.seed)

    ious_by_step = [[] for _ in range(args.horizon)]
    mses_by_step = [[] for _ in range(args.horizon)]
    example = None
    for episode in range(args.episodes):
        result = rollout(environment, world_model, renderer, decoder,
                         args.seed + episode, args.warmup, args.horizon, rng)
        if result is None:
            continue
        frames, ious, mses = result
        for step, (score, mse) in enumerate(zip(ious, mses)):
            ious_by_step[step].append(score)
            mses_by_step[step].append(mse)
        if example is None and len(frames) == args.horizon:
            example = frames

    results = {
        "mlpx_checkpoint": str(mlpx_path),
        "episodes": args.episodes,
        "per_step": [
            {"step": step + 1, "rollouts": len(ious), "drawn_vehicle_iou": float(np.mean(ious)),
             "state_mse": float(np.mean(mses))}
            for step, (ious, mses) in enumerate(zip(ious_by_step, mses_by_step)) if ious
        ],
    }
    (output_dir / "metrics.json").write_text(json.dumps(results, indent=2))
    if example:
        gap = np.full((8, FRAME_WIDTH), 255, dtype=np.uint8)
        parts = [part for frames in example for part in (stack_rows(list(frames)), gap)][:-1]
        plt.imsave(output_dir / "rollout.png", np.concatenate(parts, axis=0), cmap="gray", vmin=0, vmax=255)
    print(json.dumps(results, indent=2))
    print(f"Saved to {output_dir} (rollout.png rows per step: real, MLPx drawn, MLPx decoded)")
    renderer.close()
    environment.close()


if __name__ == "__main__":
    main()
