"""Show MLPx predictions as frames, next to what really happened.

From a start state, MLPx imagines the next states open-loop (each prediction
is fed back as the next input), with the same ego actions that the real
environment takes. Each step shows three rows:

    real            the real render
    MLPx (drawn)    the predicted state, drawn by StateRenderer
    MLPx (decoded)  the predicted state, turned into a frame by StateDecoder

It also reports, per prediction step, the vehicle-pixel IoU between the
drawn prediction and the real frame, and the state MSE. Only rollouts that
last the full horizon are counted.

Works with both MLPx state formats: 55 values (Experiments 1-4) and 57 values
(Experiment 5 and later: + ego target speed and target lane). The pictures and
the state MSE use the 55 kinematic values.

``--messages none`` gives MLPx an empty message input (for a model trained
without messages). ``--messages all`` gives it, at each step, the messages of
every visible car in the real environment (for a model trained with them).

Needs an MLPx checkpoint (default checkpoints/experiment_N/MLPx/best.pt) and a
state decoder checkpoint. Run from the project root:

    .venv/bin/python -m scripts.evaluation.visualize_mlpx_rollout --experiment 3 --horizon 5 --messages all

The demo MLPx in PR 11 was made with existing scripts (no messages):

    .venv/bin/python -m scripts.data_collection.collect_random_no_message_dataset \\
        --episodes 1000 --experiment 98
    .venv/bin/python -m scripts.training.train_mlpx_wm --experiment 98 --epochs 30 \\
        --data data/experiment_98/random_no_message_dataset.jsonl
"""

import argparse
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.data_collection.show_visible_neighbors import spatial_vehicle_slots, visible_neighbors
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
from src.cooperative_driving.visualization.state_renderer import FEATURES, ROWS
from src.cooperative_driving.visualization.frames import (
    FRAME_HEIGHT, FRAME_SCALING, FRAME_WIDTH, iou, make_frame_environment, to_gray, vehicle_mask,
)


KINEMATIC_STATE_SIZE = ROWS * FEATURES   # 55: presence, x, y, vx, vy for 11 rows
VISIBILITY_M, SIDE_WINDOW_M = 60.0, 10.0  # same as the request DQN and its renderer
RANDOM_ACTION_PROBABILITY = 0.3            # same mix as collect_frame_dataset


def all_visible_messages(environment):
    """MLPx message input with a message from every visible spatial slot."""
    slots = spatial_vehicle_slots(visible_neighbors(environment, VISIBILITY_M, SIDE_WINDOW_M))
    return {"messages_by_slot": [
        environment.request_neighbor_action(vehicle_id) if vehicle_id is not None else None
        for vehicle_id in slots
    ]}


def rollout(environment, world_model, renderer, decoder, seed, warmup, horizon, messages, rng):
    """Return per-step (real, drawn, decoded) frames, IoUs and state MSEs.

    With ``messages == "all"``, step t uses the messages that the real cars
    would send at step t; with ``"none"``, MLPx gets an empty message input.
    """
    observation, _ = environment.reset(seed=seed)
    for _ in range(warmup):
        observation, _, terminated, truncated, _ = environment.step(choose_action(environment)[0])
        if terminated or truncated:
            return None
    # world_model.observe gives the state in the checkpoint's own format:
    # 55 values, or 57 for ego-control models (+ target speed and target lane).
    predicted = world_model.observe(environment)
    frames, ious, mses = [], [], []
    for _ in range(horizon):
        if rng.random() < RANDOM_ACTION_PROBABILITY:
            action = rng.randrange(environment.action_space.n)
        else:
            action = choose_action(environment)[0]
        message = all_visible_messages(environment) if messages == "all" else None
        predicted = world_model.predict_step(predicted, action, message)[0].numpy()
        observation, _, terminated, truncated, _ = environment.step(action)
        kinematics = predicted[:KINEMATIC_STATE_SIZE]  # the drawing needs only the 55 kinematic values
        real = to_gray(environment.render())
        drawn = to_gray(renderer.draw(kinematics))
        frames.append((real, drawn, decoder.draw(kinematics)))
        # The metric uses the real state: lane rows are left out except near real cars.
        ious.append(iou(vehicle_mask(real, observation), vehicle_mask(drawn, observation)))
        # MSE over the 55 kinematic values, so models with 55 and 57 values compare.
        mses.append(float(np.mean((kinematics - np.asarray(observation).reshape(-1)) ** 2)))
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
    parser.add_argument("--messages", choices=["none", "all"], default="none",
                        help="none: empty message input; all: messages from every visible car")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    mlpx_path = Path(args.mlpx_checkpoint or experiment_checkpoint_dir(args.experiment, "MLPx") / "best.pt")
    decoder_path = Path(args.decoder_checkpoint
                        or experiment_checkpoint_dir(args.experiment, "StateDecoder") / "best.pt")
    for name, path, how in [
        ("MLPx", mlpx_path, "train one with scripts.training.train_mlpx_wm or pass --mlpx-checkpoint"),
        ("state decoder", decoder_path, "train one with scripts.training.train_state_decoder "
                                        "or pass --decoder-checkpoint"),
    ]:
        if not path.exists():
            parser.error(f"No {name} checkpoint at {path}: {how}")
    output_dir = Path(args.output_dir or experiment_results_dir(args.experiment, "StateDecoder") / "mlpx_rollout")
    output_dir.mkdir(parents=True, exist_ok=True)

    world_model, decoder = MLPxWorldModel.load(mlpx_path), load_decoder(decoder_path)
    environment = make_frame_environment()
    renderer = StateRenderer(FRAME_WIDTH, FRAME_HEIGHT, FRAME_SCALING)
    rng = random.Random(args.seed)

    # Only rollouts that last the full horizon count, so every step averages
    # the same rollouts (no mix of model error and drop-outs).
    complete, skipped = [], 0
    for episode in range(args.episodes):
        result = rollout(environment, world_model, renderer, decoder,
                         args.seed + episode, args.warmup, args.horizon, args.messages, rng)
        if result is None or len(result[0]) < args.horizon:
            skipped += 1
            continue
        complete.append(result)

    if not complete:
        parser.error("No rollout lasted the full horizon; use more --episodes or a shorter --horizon")
    ious = np.array([result[1] for result in complete])
    mses = np.array([result[2] for result in complete])
    results = {
        "mlpx_checkpoint": str(mlpx_path),
        "messages": args.messages,
        "episodes": args.episodes,
        "complete_rollouts": len(complete),
        "skipped_rollouts": skipped,
        "per_step": [
            {"step": step + 1, "drawn_vehicle_iou": float(ious[:, step].mean()),
             "state_mse": float(mses[:, step].mean())}
            for step in range(args.horizon)
        ],
    }
    example = complete[0][0]
    (output_dir / "metrics.json").write_text(json.dumps(results, indent=2))
    gap = np.full((8, FRAME_WIDTH), 255, dtype=np.uint8)
    parts = [part for frames in example for part in (stack_rows(list(frames)), gap)][:-1]
    plt.imsave(output_dir / "rollout.png", np.concatenate(parts, axis=0), cmap="gray", vmin=0, vmax=255)
    print(json.dumps(results, indent=2))
    print(f"Saved to {output_dir} (rollout.png rows per step: real, MLPx drawn, MLPx decoded)")
    renderer.close()
    environment.close()


if __name__ == "__main__":
    main()
