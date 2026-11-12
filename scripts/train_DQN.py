import argparse
import csv
import random
import shutil
from collections import deque
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from src.cooperative_driving.environments.customEnv import CustomHighwayEnv
from src.cooperative_driving.communication.getMessage import getMessage
from src.cooperative_driving.communication.request_utils import (
    requested_vehicle,
    valid_request_actions,
)
from src.cooperative_driving.models.MLP.mlp import MLP_worldModel
from src.cooperative_driving.policies.DQN import DQNPolicy
from src.cooperative_driving.planner.MCP import MCPPlanner
from scripts.experiment_utils import create_run_dir, save_run_metadata



NUM_EPISODES = 1000
BATCH_SIZE = 128
REPLAY_BUFFER_SIZE = 10_000
LEARNING_RATE = 1e-5
DISCOUNT_FACTOR = 0.99
INITIAL_EPSILON = 1.0
MIN_EPSILON = 0.01
EPSILON_DECAY = 0.995
TARGET_UPDATE_FREQUENCY = 100
CHECKPOINT_PATH = Path("checkpoints/dqn/latest.pt")
WORLD_MODEL_CHECKPOINT = Path("checkpoints/mlp/best.pt")
REQUEST_ACTION_SIZE = 11
NO_REQUEST_ACTION = 10
COMMUNICATION_COST = 0.3
DQN_RESULTS_DIR = Path("results/dqn")
DEFAULT_DQN_LOG = DQN_RESULTS_DIR / "dqn_training.csv"


def get_run_log_path(requested_path):
    """Keep previous DQN training logs when starting a new run."""
    path = Path(requested_path)
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
    """Save report-friendly plots from one DQN run."""
    episodes = []
    rewards = []
    losses = []
    exploration = []
    requests = []
    no_requests = []
    communication_costs = []

    with log_path.open(newline="") as file:
        for row in csv.DictReader(file):
            episodes.append(int(row["episode"]))
            rewards.append(float(row["total_reward"]))
            losses.append(float(row["average_loss"]))
            exploration.append(float(row["exploration_probability"]))
            requests.append(int(row["request_count"]))
            no_requests.append(int(row["no_request_count"]))
            communication_costs.append(float(row["communication_cost"]))

    if not episodes:
        return []

    plot_paths = []
    reward_plot = log_path.with_name(f"{log_path.stem}_rewards.png")
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, rewards, alpha=0.35, label="Episode reward")

    window = min(20, len(rewards))
    if window > 1:
        moving_average = np.convolve(
            rewards,
            np.ones(window) / window,
            mode="valid",
        )
        plt.plot(
            episodes[window - 1:],
            moving_average,
            label=f"{window}-episode average",
        )

    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("DQN Request-Policy Reward")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(reward_plot, dpi=300)
    plt.close()
    plot_paths.append(reward_plot)

    training_plot = log_path.with_name(f"{log_path.stem}_training.png")
    figure, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(episodes, losses, label="Average loss")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("DQN Training Metrics")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(episodes, exploration, label="Exploration probability")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Exploration")
    axes[1].legend()
    axes[1].grid(True)
    figure.tight_layout()
    figure.savefig(training_plot, dpi=300)
    plt.close(figure)
    plot_paths.append(training_plot)

    communication_plot = log_path.with_name(
        f"{log_path.stem}_requests.png"
    )
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, requests, label="Requests")
    plt.plot(episodes, no_requests, label="No request")
    plt.xlabel("Episode")
    plt.ylabel("Count")
    plt.title("DQN Request Actions")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(communication_plot, dpi=300)
    plt.close()
    plot_paths.append(communication_plot)

    cost_plot = log_path.with_name(f"{log_path.stem}_communication_cost.png")
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, communication_costs, label="Communication cost")
    plt.xlabel("Episode")
    plt.ylabel("Cost")
    plt.title("DQN Communication Cost")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(cost_plot, dpi=300)
    plt.close()
    plot_paths.append(cost_plot)

    return plot_paths


class ReplayBuffer:
    def __init__(self, capacity):
        self.transitions = deque(maxlen=capacity)

    def add(self, observation, action, reward, next_observation, done):
        self.transitions.append(
            (observation, action, reward, next_observation, done)
        )

    def sample(self, batch_size):
        return random.sample(self.transitions, batch_size)

    def __len__(self):
        return len(self.transitions)


def message_to_vector(message):
    """Convert a vehicle message to the ten values expected by the MLP."""
    if message is None:
        return [0.0] * 10

    return [
        *message["position"],
        *message["velocity"],
        message["action"]["acceleration"],
        message["action"]["steering"],
        message["intent"]["intent_acceleration"],
        message["intent"]["intent_steering"],
        message["intent"]["intent_lane_change"],
        message["vehicle_id"],
    ]


def get_requested_message(environment, observation, request_action):
    """Return the message for a neighbor slot, or zeros for no request."""
    vehicle = requested_vehicle(environment, observation, request_action)
    if vehicle is None:
        return [0.0] * 10
    return message_to_vector(getMessage(vehicle))


def load_world_model(path):
    """Load a trained MLP and freeze it for MCP action selection."""
    world_model = MLP_worldModel()
    checkpoint = torch.load(path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    world_model.load_state_dict(state_dict)
    world_model.eval()

    for parameter in world_model.parameters():
        parameter.requires_grad = False

    return world_model


def select_request_action(
    policy,
    observation,
    exploration_probability,
    valid_actions,
):
    """Choose which vehicle to request information from."""
    if random.random() < exploration_probability:
        return random.choice(valid_actions)
    return policy.act(observation, valid_actions)


def optimize_model(
    replay_buffer,
    online_network,
    target_network,
    optimizer,
    batch_size,
    discount_factor,
):
    if len(replay_buffer) < batch_size:
        return None

    batch = replay_buffer.sample(batch_size)
    observations, actions, rewards, next_observations, dones = zip(*batch)

    observations = torch.as_tensor(
        np.asarray(observations),
        dtype=torch.float32,
    )
    actions = torch.as_tensor(actions, dtype=torch.long).unsqueeze(1)
    rewards = torch.as_tensor(rewards, dtype=torch.float32)
    next_observations = torch.as_tensor(
        np.asarray(next_observations),
        dtype=torch.float32,
    )
    dones = torch.as_tensor(dones, dtype=torch.float32)

    current_q_values = online_network(observations).gather(1, actions).squeeze(1)

    with torch.no_grad():
        all_next_q_values = target_network(next_observations)
        next_q_values = torch.stack([
            q_values[valid_request_actions(next_observation.numpy())].max()
            for q_values, next_observation in zip(
                all_next_q_values,
                next_observations,
            )
        ])
        target_q_values = rewards + discount_factor * next_q_values * (1 - dones)

    loss = F.smooth_l1_loss(current_q_values, target_q_values)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(online_network.parameters(), max_norm=10.0)
    optimizer.step()

    return loss.item()


def save_checkpoint(
    path,
    episode,
    online_network,
    target_network,
    optimizer,
    exploration_probability,
    replay_buffer,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "episode": episode,
            "online_network": online_network.state_dict(),
            "target_network": target_network.state_dict(),
            "optimizer": optimizer.state_dict(),
            "exploration_probability": exploration_probability,
            "replay_buffer": list(replay_buffer.transitions),
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=NUM_EPISODES)
    parser.add_argument("--log", default=None)
    parser.add_argument("--experiment-dir", default=None)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    run_dir = (
        Path(args.experiment_dir)
        if args.experiment_dir
        else Path(args.log).parent
        if args.log
        else create_run_dir("results/dqn/experiments", "dqn")
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    requested_log = (
        Path(args.log)
        if args.log
        else run_dir / "dqn_training.csv"
    )
    if args.resume and requested_log.exists():
        log_path = requested_log
    else:
        log_path = get_run_log_path(requested_log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    save_run_metadata(
        run_dir,
        {
            "model": "DQN request policy",
            "episodes": args.episodes,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "communication_cost": COMMUNICATION_COST,
            "discount_factor": DISCOUNT_FACTOR,
            "world_model_checkpoint": str(WORLD_MODEL_CHECKPOINT),
            "resume": args.resume,
            "log_path": str(log_path),
        },
    )
    append_log = args.resume is not None and log_path.exists()
    log_file = log_path.open("a" if append_log else "w", newline="")
    log_writer = csv.DictWriter(
        log_file,
        fieldnames=[
            "episode",
            "total_reward",
            "average_loss",
            "exploration_probability",
            "episode_length",
            "request_count",
            "no_request_count",
            "communication_cost",
        ],
    )
    if not append_log:
        log_writer.writeheader()

    environment = CustomHighwayEnv()
    observation, _ = environment.reset()

    observation_size = observation.size
    driving_action_size = environment.action_space.n

    world_model = load_world_model(WORLD_MODEL_CHECKPOINT)
    planner = MCPPlanner(
        world_model=world_model,
        action_size=driving_action_size,
        planning_horizon=5,
        discount_factor=DISCOUNT_FACTOR,
    )

    request_policy = DQNPolicy(observation_size, REQUEST_ACTION_SIZE)
    target_network = DQNPolicy(observation_size, REQUEST_ACTION_SIZE)
    target_network.load_state_dict(request_policy.state_dict())
    target_network.eval()

    optimizer = torch.optim.Adam(request_policy.parameters(), lr=LEARNING_RATE)
    replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)
    exploration_probability = INITIAL_EPSILON
    total_steps = 0
    start_episode = 0

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        request_policy.load_state_dict(checkpoint["online_network"])
        target_network.load_state_dict(checkpoint["target_network"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_episode = checkpoint["episode"] + 1
        exploration_probability = checkpoint.get(
            "exploration_probability",
            INITIAL_EPSILON,
        )
        for transition in checkpoint.get("replay_buffer", []):
            replay_buffer.add(*transition)
        print(
            f"Resumed DQN from {args.resume} at episode "
            f"{start_episode}"
        )

    for episode in range(start_episode, args.episodes):
        observation, _ = environment.reset()
        episode_reward = 0.0
        episode_losses = []
        request_count = 0
        no_request_count = 0

        for step in range(environment.config["duration"]):
            valid_actions = valid_request_actions(observation)
            request_action = select_request_action(
                request_policy,
                observation,
                exploration_probability,
                valid_actions,
            )
            if request_action == NO_REQUEST_ACTION:
                no_request_count += 1
            else:
                request_count += 1

            planning_message = get_requested_message(
                environment,
                observation,
                request_action,
            )

            # MCP selects the driving action using the requested message.
            driving_action = planner.plan(
                observation,
                planning_message,
            )

            next_observation, driving_reward, terminated, truncated, _ = environment.step(
                driving_action
            )
            done = terminated or truncated
            communication_cost = (
                COMMUNICATION_COST
                if request_action != NO_REQUEST_ACTION
                else 0.0
            )
            total_reward = driving_reward - communication_cost

            replay_buffer.add(
                observation,
                request_action,
                total_reward,
                next_observation,
                done,
            )

            loss = optimize_model(
                replay_buffer,
                request_policy,
                target_network,
                optimizer,
                BATCH_SIZE,
                DISCOUNT_FACTOR,
            )
            if loss is not None:
                episode_losses.append(loss)

            observation = next_observation
            episode_reward += total_reward
            total_steps += 1

            if total_steps % TARGET_UPDATE_FREQUENCY == 0:
                target_network.load_state_dict(request_policy.state_dict())

            if done:
                break

        exploration_probability = max(
            MIN_EPSILON,
            exploration_probability * EPSILON_DECAY,
        )

        average_loss = (
            sum(episode_losses) / len(episode_losses)
            if episode_losses
            else 0.0
        )
        print(
            f"Episode {episode + 1}/{args.episodes} | "
            f"steps={step + 1} | reward={episode_reward:.2f} | "
            f"loss={average_loss:.4f} | "
            f"exploration={exploration_probability:.3f}"
        )

        log_writer.writerow({
            "episode": episode + 1,
            "total_reward": episode_reward,
            "average_loss": average_loss,
            "exploration_probability": exploration_probability,
            "episode_length": step + 1,
            "request_count": request_count,
            "no_request_count": no_request_count,
            "communication_cost": request_count * COMMUNICATION_COST,
        })
        log_file.flush()

        if (episode + 1) % 10 == 0:
            save_checkpoint(
                CHECKPOINT_PATH,
                episode,
                request_policy,
                target_network,
                optimizer,
                exploration_probability,
                replay_buffer,
            )

    environment.close()
    log_file.close()
    plot_paths = save_training_plots(log_path)
    checkpoint_copy_dir = run_dir / "checkpoints"
    checkpoint_copy_dir.mkdir(exist_ok=True)
    if CHECKPOINT_PATH.exists():
        shutil.copy2(CHECKPOINT_PATH, checkpoint_copy_dir / CHECKPOINT_PATH.name)
    print(f"Checkpoint saved to {CHECKPOINT_PATH}")
    print(f"Experiment saved to {run_dir}")
    print(f"Training metrics saved to {log_path}")
    for plot_path in plot_paths:
        print(f"Training graph saved to {plot_path}")


if __name__ == "__main__":
    main()
