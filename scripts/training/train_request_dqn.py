"""Train a DQN that learns which neighboring vehicle to ask.

The frozen three-step MLPx planner chooses the ego's driving action after the
DQN selects a message request. The DQN actions are eight spatial vehicle
slots plus ask-nobody.

Run:

    .venv/bin/python -m scripts.training.train_request_dqn --episodes 1000
"""

import argparse
import csv
import hashlib
import os
import random
import warnings
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from scripts.data_collection.show_visible_neighbors import SLOT_ORDER, ordered_visible_ids, visible_neighbors
from src.cooperative_driving.environments.customEnv import CustomEnv
from src.cooperative_driving.models.MLPx.mlpx import MLPxWorldModel
from src.cooperative_driving.planner.MCP import MCPPlanner
from src.cooperative_driving.policies.request_dqn import RequestDQN
from scripts.utils.experiment_utils import (
    add_experiment_argument,
    create_run_dir,
    experiment_checkpoint_dir,
    experiment_results_dir,
    save_run_metadata,
)


ASK_NOBODY = 8


def request_state(environment, visible):
    """Encode ego and eight spatial neighbor slots into 45 values."""
    ego = environment.vehicle
    features = [1.0, float(ego.lane_index[2]) / 3.0, ego.speed / 30.0, 0.0, 0.0]
    for slot in SLOT_ORDER:
        candidate = visible.get(slot)
        if candidate is None:
            features.extend([0.0] * 5)
            continue
        vehicle = candidate["vehicle"]
        relative = vehicle.position - ego.position
        relative_velocity = vehicle.velocity - ego.velocity
        features.extend([
            1.0,
            float(relative[0]) / 60.0,
            float(relative[1]) / 4.0,
            float(relative_velocity[0]) / 30.0,
            float(relative_velocity[1]) / 10.0,
        ])
    return np.asarray(features, dtype=np.float32)


def valid_actions(visible, action_encoding="spatial_v1"):
    if action_encoding == "compressed_v0":
        return list(range(len(ordered_visible_ids(visible)))) + [ASK_NOBODY]
    if action_encoding != "spatial_v1":
        raise ValueError(f"Unknown request action encoding: {action_encoding}")
    return [index for index, slot in enumerate(SLOT_ORDER) if slot in visible] + [ASK_NOBODY]


def resolve_requested_vehicle(visible, action, action_encoding="spatial_v1"):
    """Resolve an action without ever compressing fixed spatial positions."""
    if action == ASK_NOBODY:
        return None
    if action not in valid_actions(visible, action_encoding):
        raise ValueError(f"Request action {action} targets an empty or invalid slot")
    if action_encoding == "compressed_v0":
        occupied = [visible[slot]["vehicle"] for slot in SLOT_ORDER if slot in visible]
        return occupied[action]
    return visible[SLOT_ORDER[action]]["vehicle"]


def requested_message(environment, visible, action, action_encoding="spatial_v1"):
    vehicle = resolve_requested_vehicle(visible, action, action_encoding)
    if vehicle is None:
        return None
    messages = [None] * len(SLOT_ORDER)
    vehicle_id = vehicle.vehicle_id
    slot = next(index for index, key in enumerate(SLOT_ORDER) if key in visible and visible[key]["vehicle"].vehicle_id == vehicle_id)
    messages[slot] = environment.request_neighbor_action(vehicle_id)
    return {"messages_by_slot": messages}


def select_action(model, state, valid, epsilon, rng):
    if rng.random() < epsilon:
        return rng.choice(valid)
    with torch.no_grad():
        values = model(torch.tensor(state).unsqueeze(0)).squeeze(0)
        masked = torch.full_like(values, -torch.inf)
        masked[valid] = values[valid]
        return int(masked.argmax())


def optimize(online, target, optimizer, replay, batch_size, gamma):
    if len(replay) < batch_size:
        return None
    batch = random.sample(replay, batch_size)
    states, actions, rewards, next_states, dones, next_valid = zip(*batch)
    states = torch.tensor(np.asarray(states), dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.long)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    next_states = torch.tensor(np.asarray(next_states), dtype=torch.float32)
    dones = torch.tensor(dones, dtype=torch.float32)
    current = online(states).gather(1, actions[:, None]).squeeze(1)
    with torch.no_grad():
        next_values = target(next_states)
        mask = torch.zeros_like(next_values, dtype=torch.bool)
        for index, valid in enumerate(next_valid):
            mask[index, list(valid)] = True
        future = next_values.masked_fill(~mask, -torch.inf).max(dim=1).values
        expected = rewards + gamma * future * (1.0 - dones)
    loss = F.smooth_l1_loss(current, expected)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(online.parameters(), 5.0)
    optimizer.step()
    return float(loss.item())


def training_snapshot(online, target, optimizer, replay, rng, epsilon, completed, returns, config):
    """Capture a completed-episode boundary, including all training RNGs."""
    return {
        "checkpoint_version": 2,
        "model_state_dict": online.state_dict(),
        "target_state_dict": target.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "replay": list(replay),
        "episodes": completed,
        "epsilon": epsilon,
        "returns": list(returns),
        "rng_state": rng.getstate(),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "training_config": config,
        "planning_horizon": config["planning_horizon"],
        "communication_cost": config["communication_cost"],
        "mlpx_checkpoint": config["mlpx_checkpoint"],
        "masked_next_request_targets": True,
        "request_action_encoding": "spatial_v1",
    }


def restore_training(checkpoint, online, target, optimizer, replay, rng):
    if checkpoint.get("request_action_encoding") != "spatial_v1":
        raise ValueError("Cannot resume compressed-action DQN weights as fixed spatial actions")
    online.load_state_dict(checkpoint["model_state_dict"])
    completed = int(checkpoint.get("episodes", 0))
    if checkpoint.get("checkpoint_version") == 2:
        target.load_state_dict(checkpoint["target_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        replay.extend(checkpoint["replay"])
        rng.setstate(checkpoint["rng_state"])
        random.setstate(checkpoint["python_rng_state"])
        np.random.set_state(checkpoint["numpy_rng_state"])
        torch.set_rng_state(checkpoint["torch_rng_state"])
        return completed, checkpoint["epsilon"], list(checkpoint["returns"])
    warnings.warn(
        "Weights-only checkpoint: continuing learned spatial-action weights and episode count, "
        "but restarting optimizer, target network, replay, and random states. "
        "Epsilon is inferred from completed episodes; future latest.pt saves support full resume.",
        UserWarning, stacklevel=2,
    )
    target.load_state_dict(online.state_dict())
    return completed, max(0.05, 0.995 ** completed), []


def atomic_save(checkpoint, path):
    """Replace a checkpoint only after its new file is fully written."""
    temporary = path.with_name(path.name + ".tmp")
    torch.save(checkpoint, temporary)
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_experiment_argument(parser)
    parser.add_argument("--episodes", type=int, default=1000,
                        help="Number of additional episodes to train in this invocation")
    parser.add_argument("--resume", nargs="?", const="auto", default=None,
                        help="Resume a checkpoint; without a path uses latest.pt, then best.pt")
    parser.add_argument("--save-every", type=int, default=50,
                        help="Save full training state every N completed episodes and at the end")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--planning-horizon", type=int, default=3)
    parser.add_argument("--communication-cost", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--replay-size", type=int, default=50000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output", default=None)
    parser.add_argument("--log-path", default=None,
                        help="CSV path; defaults to the selected experiment's RequestDQN run folder")
    args = parser.parse_args()
    if args.episodes < 1 or args.save_every < 1:
        parser.error("episodes and save-every must be positive")

    request_checkpoint_dir = experiment_checkpoint_dir(args.experiment, "RequestDQN")
    resume_checkpoint = None
    resume_path = None
    if args.resume:
        resume_path = Path(args.resume) if args.resume != "auto" else request_checkpoint_dir / "latest.pt"
        if args.resume == "auto" and not resume_path.exists():
            resume_path = request_checkpoint_dir / "best.pt"
        resume_checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
        if resume_checkpoint.get("request_action_encoding") != "spatial_v1":
            parser.error("Resume requires a spatial_v1 DQN checkpoint; compressed actions require retraining")
        saved_config = resume_checkpoint.get("training_config")
        if saved_config:
            if saved_config["experiment"] != args.experiment:
                parser.error("Resume checkpoint experiment differs from --experiment")
            for key in ("seed", "planning_horizon", "communication_cost", "batch_size", "replay_size", "lr"):
                setattr(args, key, saved_config[key])
        else:
            for key in ("planning_horizon", "communication_cost"):
                setattr(args, key, resume_checkpoint.get(key, getattr(args, key)))

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    mlpx_checkpoint = experiment_checkpoint_dir(args.experiment, "MLPx") / "best.pt"
    if resume_checkpoint:
        mlpx_checkpoint = Path(resume_checkpoint.get("mlpx_checkpoint", mlpx_checkpoint))
    mlpx_hash = hashlib.sha256(mlpx_checkpoint.read_bytes()).hexdigest()
    if resume_checkpoint and resume_checkpoint.get("training_config", {}).get("mlpx_sha256", mlpx_hash) != mlpx_hash:
        parser.error("Frozen MLPx checkpoint changed since the saved training state")
    world_model = MLPxWorldModel.load(mlpx_checkpoint)
    planner = None
    online = RequestDQN()
    target = RequestDQN(); target.load_state_dict(online.state_dict())
    optimizer = torch.optim.Adam(online.parameters(), lr=args.lr)
    replay = deque(maxlen=args.replay_size)
    rng = random.Random(args.seed)
    start_episode, epsilon, returns = 0, 1.0, []
    if resume_checkpoint:
        start_episode, epsilon, returns = restore_training(
            resume_checkpoint, online, target, optimizer, replay, rng
        )
        print(f"Resuming {resume_path}: completed={start_episode} epsilon={epsilon:.4f} "
              f"replay={len(replay)}; training {args.episodes} additional episodes")
    output = Path(args.output) if args.output else request_checkpoint_dir / "best.pt"
    if output.name == "latest.pt":
        parser.error("--output must not be named latest.pt; that name is reserved for resumable state")
    output.parent.mkdir(parents=True, exist_ok=True)
    run_root = experiment_results_dir(args.experiment, "RequestDQN") / "experiments"
    run_dir = create_run_dir(run_root, "request_dqn")
    config = {key: getattr(args, key) for key in (
        "experiment", "seed", "planning_horizon", "communication_cost", "batch_size", "replay_size", "lr"
    )}
    config.update(mlpx_checkpoint=str(mlpx_checkpoint), mlpx_sha256=mlpx_hash)
    save_run_metadata(run_dir, {**config, "resume_from": str(resume_path) if resume_path else None,
                               "start_episode": start_episode, "additional_episodes": args.episodes,
                               "request_action_encoding": "spatial_v1"})
    log_path = Path(args.log_path) if args.log_path else run_dir / "request_dqn_training.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing training log: {log_path}")
    log_file = log_path.open("x", newline="")
    log = csv.DictWriter(log_file, fieldnames=[
        "episode", "return", "mean_loss", "epsilon", "requests", "crashed", "truncated"
    ])
    log.writeheader()
    end_episode = start_episode + args.episodes
    environment = None

    def save_progress(completed):
        snapshot = training_snapshot(online, target, optimizer, replay, rng,
                                     epsilon, completed, returns, config)
        atomic_save(snapshot, run_dir / "latest.pt")
        atomic_save(snapshot, output.with_name("latest.pt"))

    save_progress(start_episode)

    try:
        for episode in range(start_episode, end_episode):
            environment = CustomEnv()
            observation, _ = environment.reset(seed=args.seed + episode)
            planner = MCPPlanner(world_model, action_size=environment.action_space.n,
                                 planning_horizon=args.planning_horizon)
            total = 0.0
            episode_losses = []
            requests = 0
            terminated = truncated = False
            for _ in range(environment.config["duration"]):
                visible = visible_neighbors(environment, 60.0, 10.0)
                state = request_state(environment, visible)
                action = select_action(online, state, valid_actions(visible), epsilon, rng)
                message = requested_message(environment, visible, action)
                requests += int(action != ASK_NOBODY)
                driving_action = planner.plan(observation, message)
                next_observation, reward, terminated, truncated, _ = environment.step(driving_action)
                next_visible = visible_neighbors(environment, 60.0, 10.0)
                next_state = request_state(environment, next_visible)
                transition_reward = float(reward) - (args.communication_cost if action != ASK_NOBODY else 0.0)
                total += transition_reward
                replay.append((state, action, transition_reward, next_state,
                               terminated or truncated, valid_actions(next_visible)))
                loss = optimize(online, target, optimizer, replay, args.batch_size, 0.99)
                if loss is not None:
                    episode_losses.append(loss)
                observation = next_observation
                if terminated or truncated:
                    break
            if episode % 50 == 0:
                target.load_state_dict(online.state_dict())
            epsilon = max(0.05, epsilon * 0.995)
            returns.append(total)
            log.writerow({
                "episode": episode + 1,
                "return": total,
                "mean_loss": float(np.mean(episode_losses)) if episode_losses else "",
                "epsilon": epsilon,
                "requests": requests,
                "crashed": int(terminated),
                "truncated": int(truncated),
            })
            log_file.flush()
            if (episode + 1) % 10 == 0:
                print(f"Episode {episode + 1}/{end_episode} | return={np.mean(returns[-10:]):.3f} "
                      f"| loss={np.mean(episode_losses) if episode_losses else float('nan'):.4f} "
                      f"| epsilon={epsilon:.3f}")
            environment.close()
            environment = None
            if (episode + 1) % args.save_every == 0 or episode + 1 == end_episode:
                save_progress(episode + 1)
    except KeyboardInterrupt:
        print(f"Interrupted. Resume the last completed save with --resume {output.with_name('latest.pt')}")
        return
    finally:
        if environment is not None:
            environment.close()
        log_file.close()

    checkpoint = {"model_state_dict": online.state_dict(), "episodes": end_episode,
                  "planning_horizon": args.planning_horizon, "communication_cost": args.communication_cost,
                  "masked_next_request_targets": True, "mlpx_checkpoint": str(mlpx_checkpoint),
                  "request_action_encoding": "spatial_v1"}
    torch.save(checkpoint, output)
    torch.save(checkpoint, run_dir / "best.pt")
    print(f"Saved request DQN to {output}")
    print(f"Saved training log to {log_path}")


if __name__ == "__main__":
    main()
