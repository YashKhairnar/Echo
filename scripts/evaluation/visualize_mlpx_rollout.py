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

``--scenes traffic`` (default) uses normal CustomEnv traffic; the ego follows
the intent-aware policy with 30% random actions. ``--scenes matched_intent``
uses the Experiment 5 challenge scenes (seeds from 400000): each scene runs
twice, once where the neighbour stays and once where it cuts in, from the same
public observation. The ego keeps IDLE there, so only the predictions differ.
Results are reported for "stay", "merge" and both.

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
from src.cooperative_driving.environments.intent_scenarios import matched_intent_pair
from src.cooperative_driving.models.decoder import load_decoder
from src.cooperative_driving.models.MLPx.mlpx import MLPxWorldModel
from src.cooperative_driving.visualization import StateRenderer
from src.cooperative_driving.visualization.state_renderer import FEATURES, ROWS
from src.cooperative_driving.visualization.frames import (
    FRAME_HEIGHT, FRAME_SCALING, FRAME_WIDTH, frame_config, iou, make_frame_environment, to_gray,
    vehicle_mask,
)


KINEMATIC_STATE_SIZE = ROWS * FEATURES   # 55: presence, x, y, vx, vy for 11 rows
IDLE = 1                                  # DiscreteMetaAction: keep lane and speed
SCENE_SEED = 400_000                      # same first seed as evaluate_intent_scenarios
VISIBILITY_M, SIDE_WINDOW_M = 60.0, 10.0  # same as the request DQN and its renderer
RANDOM_ACTION_PROBABILITY = 0.3            # same mix as collect_frame_dataset


def visible_messages(environment, only_vehicle_id=None):
    """MLPx message input with a message from every visible spatial slot.

    With ``only_vehicle_id``, only that car's slot gets a message (the other
    slots stay empty), as when the ego asks one car.
    """
    slots = spatial_vehicle_slots(visible_neighbors(environment, VISIBILITY_M, SIDE_WINDOW_M))
    return {"messages_by_slot": [
        environment.request_neighbor_action(vehicle_id)
        if vehicle_id is not None and only_vehicle_id in (None, vehicle_id) else None
        for vehicle_id in slots
    ]}


def rollout(environment, world_model, renderer, decoder, seed, warmup, horizon, messages, rng):
    """Normal traffic: reset, drive ``warmup`` steps, then predict ``horizon`` steps.

    The ego follows the intent-aware policy with 30% random actions. Returns
    per-step (real, drawn, decoded) frames, IoUs and state MSEs, or None if
    the episode ends during the warmup.
    """
    environment.reset(seed=seed)
    for _ in range(warmup):
        _, _, terminated, truncated, _ = environment.step(choose_action(environment)[0])
        if terminated or truncated:
            return None

    def policy_action(env):
        if rng.random() < RANDOM_ACTION_PROBABILITY:
            return rng.randrange(env.action_space.n)
        return choose_action(env)[0]

    return predict_and_compare(environment, world_model, renderer, decoder, horizon, messages, policy_action)


def prepare_for_frames(environment):
    """Make a CustomEnv made elsewhere render offscreen frames at the frame size."""
    environment.render_mode = "rgb_array"
    environment.config.update(frame_config())
    environment.config["offscreen_rendering"] = True
    environment.render()  # creates the viewer with the frame size
    environment.viewer.enabled = True


def scene_rollout(seed, mode, world_model, renderer, decoder, horizon, messages):
    """Matched-intent challenge scene (``mode`` "stay" or "merge").

    Both modes start from the same public observation; only a message shows
    whether the neighbour stays or cuts in. The ego keeps IDLE, so both modes
    get the same actions and only the predictions differ.
    """
    environments, metadata = matched_intent_pair(seed, duration=horizon)
    environments["merge" if mode == "stay" else "stay"].close()
    environment = environments[mode]
    try:
        prepare_for_frames(environment)
        return predict_and_compare(environment, world_model, renderer, decoder, horizon, messages,
                                   lambda env: IDLE, neighbour_id=metadata["neighbor_id"])
    finally:
        environment.close()


def predict_and_compare(environment, world_model, renderer, decoder, horizon, messages, choose,
                        neighbour_id=None):
    """Predict open-loop from the current state and compare with the real steps.

    With ``messages == "all"``, step t uses the messages that all visible real
    cars would send at step t; with ``"neighbour"``, only the car
    ``neighbour_id`` sends one (as in evaluate_intent_scenarios); with
    ``"none"``, MLPx gets an empty message input. ``choose(environment)``
    gives the ego action of each step.
    """
    if messages == "neighbour" and neighbour_id is None:
        raise ValueError('messages="neighbour" needs a neighbour (challenge scenes only)')
    # world_model.observe gives the state in the checkpoint's own format:
    # 55 values, or 57 for ego-control models (+ target speed and target lane).
    predicted = world_model.observe(environment)
    frames, ious, mses = [], [], []
    for _ in range(horizon):
        action = choose(environment)
        if messages == "all":
            message = visible_messages(environment)
        elif messages == "neighbour":
            message = visible_messages(environment, only_vehicle_id=neighbour_id)
        else:
            message = None
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


def is_complete(result, horizon):
    return result is not None and len(result[0]) == horizon


def summarize(results, horizon):
    """Mean IoU and state MSE per step over the rollouts that lasted ``horizon`` steps."""
    complete = [result for result in results if is_complete(result, horizon)]
    summary = {"complete_rollouts": len(complete), "skipped_rollouts": len(results) - len(complete)}
    if complete:
        ious = np.array([result[1] for result in complete])
        mses = np.array([result[2] for result in complete])
        summary["per_step"] = [
            {"step": step + 1, "drawn_vehicle_iou": float(ious[:, step].mean()),
             "state_mse": float(mses[:, step].mean())}
            for step in range(horizon)
        ]
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_experiment_argument(parser)
    parser.add_argument("--mlpx-checkpoint", default=None)
    parser.add_argument("--decoder-checkpoint", default=None)
    parser.add_argument("--seed", type=int, default=10_000)  # away from training seeds
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--messages", choices=["none", "all", "neighbour"], default="none",
                        help="none: empty message input; all: messages from every visible car; "
                             "neighbour: only the challenge-scene neighbour sends one")
    parser.add_argument("--scenes", choices=["traffic", "matched_intent"], default="traffic",
                        help="traffic: normal CustomEnv traffic; matched_intent: Experiment 5 "
                             "challenge scenes, each run as 'stay' and 'merge'")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    if args.messages == "neighbour" and args.scenes != "matched_intent":
        parser.error("--messages neighbour needs --scenes matched_intent")
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
    renderer = StateRenderer(FRAME_WIDTH, FRAME_HEIGHT, FRAME_SCALING)

    # Only rollouts that last the full horizon count, so every step averages
    # the same rollouts (no mix of model error and drop-outs).
    groups = {}  # group name -> list of results (None = ended early)
    if args.scenes == "traffic":
        environment = make_frame_environment()
        rng = random.Random(args.seed)
        groups["traffic"] = [
            rollout(environment, world_model, renderer, decoder,
                    args.seed + episode, args.warmup, args.horizon, args.messages, rng)
            for episode in range(args.episodes)
        ]
        environment.close()
    else:
        for mode in ("stay", "merge"):
            groups[mode] = [
                scene_rollout(SCENE_SEED + episode, mode, world_model, renderer, decoder,
                              args.horizon, args.messages)
                for episode in range(args.episodes)
            ]
        groups["both"] = groups["stay"] + groups["merge"]

    results = {
        "mlpx_checkpoint": str(mlpx_path),
        "scenes": args.scenes,
        "messages": args.messages,
        "episodes": args.episodes,
        "groups": {name: summarize(group, args.horizon) for name, group in groups.items()},
    }
    if not any(group["complete_rollouts"] for group in results["groups"].values()):
        parser.error("No rollout lasted the full horizon; use more --episodes or a shorter --horizon")
    # Example picture: the first complete rollout (the "merge" mode for challenge scenes).
    example_group = groups["traffic" if args.scenes == "traffic" else "merge"]
    example = next(result[0] for result in example_group if is_complete(result, args.horizon))
    (output_dir / "metrics.json").write_text(json.dumps(results, indent=2))
    gap = np.full((8, FRAME_WIDTH), 255, dtype=np.uint8)
    parts = [part for frames in example for part in (stack_rows(list(frames)), gap)][:-1]
    plt.imsave(output_dir / "rollout.png", np.concatenate(parts, axis=0), cmap="gray", vmin=0, vmax=255)
    print(json.dumps(results, indent=2))
    print(f"Saved to {output_dir} (rollout.png rows per step: real, MLPx drawn, MLPx decoded)")
    renderer.close()


if __name__ == "__main__":
    main()
