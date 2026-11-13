import argparse
import csv
import random
from collections import deque
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from scripts.experiment_utils import create_run_dir, save_run_metadata
from src.cooperative_driving.communication.getMessage import getMessage
from src.cooperative_driving.communication.request_utils import (
    visible_neighbor_ids,
)
from src.cooperative_driving.environments.customEnv import CustomHighwayEnv
from src.cooperative_driving.models.MLPx.mlpx import MLPxWorldModel
from src.cooperative_driving.planner.MCP import MCPPlanner
from src.cooperative_driving.policies.DQN import MultiRequestDQNPolicy


MAX_REQUEST_SLOTS = 10
COMMUNICATION_COST = 0.1
BATCH_SIZE = 128
REPLAY_BUFFER_SIZE = 10_000
LEARNING_RATE = 1e-5
DISCOUNT_FACTOR = 0.99
INITIAL_EPSILON = 1.0
MIN_EPSILON = 0.01
EPSILON_DECAY = 0.995
TARGET_UPDATE_FREQUENCY = 100
DQNX_RESULTS_ROOT = Path("results/DQNx/experiments")


class ReplayBuffer:
    def __init__(self, capacity):
        self.transitions = deque(maxlen=capacity)

    def add(self, *transition):
        self.transitions.append(transition)

    def sample(self, batch_size):
        return random.sample(self.transitions, batch_size)

    def __len__(self):
        return len(self.transitions)


def message_for_slots(environment, observation, requested_slots):
    visible_ids = visible_neighbor_ids(observation)
    request_mask = [0] * MAX_REQUEST_SLOTS
    messages = [[0.0] * 4 for _ in range(MAX_REQUEST_SLOTS)]
    for slot in requested_slots:
        if slot >= len(visible_ids):
            continue
        request_mask[slot] = 1
        vehicle = next(
            (car for car in environment.road.vehicles
             if car.vehicle_id == visible_ids[slot]),
            None,
        )
        if vehicle is None:
            continue
        message = getMessage(vehicle)
        messages[slot] = [
            message["intent"]["intent_acceleration"],
            message["intent"]["intent_steering"],
            message["target_lane"],
            message["time_to_collision"],
        ]
    return {"request_mask": request_mask, "messages": messages}


def choose_request_slots(policy, observation, epsilon):
    visible_slots = list(range(len(visible_neighbor_ids(observation))))
    if random.random() < epsilon:
        return [slot for slot in visible_slots if random.random() < 0.5]
    return policy.act(observation, visible_slots)


def optimize_model(buffer, online, target, optimizer):
    if len(buffer) < BATCH_SIZE:
        return None
    # Learn from real environment transitions stored in replay memory.
    batch = buffer.sample(BATCH_SIZE)
    observations, actions, rewards, next_observations, dones = zip(*batch)

    observations = torch.as_tensor(np.asarray(observations), dtype=torch.float32)
    actions = torch.as_tensor(np.asarray(actions), dtype=torch.long)
    rewards = torch.as_tensor(rewards, dtype=torch.float32)
    next_observations = torch.as_tensor(
        np.asarray(next_observations), dtype=torch.float32
    )
    dones = torch.as_tensor(dones, dtype=torch.float32)

    current_q = online(observations).gather(
        2, actions.unsqueeze(-1)
    ).squeeze(-1)

    with torch.no_grad():
        next_all_q = target(next_observations)
        next_q = next_all_q[:, :, 0].clone()
        for index, next_observation in enumerate(next_observations):
            visible_count = len(visible_neighbor_ids(next_observation.numpy()))
            if visible_count:
                next_q[index, :visible_count] = next_all_q[
                    index, :visible_count
                ].max(dim=1).values
        target_q = rewards[:, None] + DISCOUNT_FACTOR * next_q * (1 - dones[:, None])

    loss = F.smooth_l1_loss(current_q, target_q)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(online.parameters(), 10.0)
    optimizer.step()
    return loss.item()


def save_plots(log_path):
    rows = list(csv.DictReader(log_path.open(newline="")))
    if not rows:
        return
    episodes = [int(row["episode"]) for row in rows]
    rewards = [float(row["total_reward"]) for row in rows]
    losses = [float(row["average_loss"]) for row in rows]
    requests = [int(row["request_count"]) for row in rows]
    costs = [float(row["communication_cost"]) for row in rows]

    plots = [
        ("dqnx_training_rewards.png", episodes, [rewards], ["Reward"], "DQNx Reward"),
        ("dqnx_training_loss.png", episodes, [losses], ["Loss"], "DQNx Loss"),
        ("dqnx_training_requests.png", episodes, [requests], ["Requests"], "DQNx Requests"),
        ("dqnx_training_communication_cost.png", episodes, [costs], ["Cost"], "DQNx Communication Cost"),
    ]
    for filename, x, lines, labels, title in plots:
        plt.figure(figsize=(10, 6))
        for line, label in zip(lines, labels):
            plt.plot(x, line, label=label)
        plt.xlabel("Episode")
        plt.title(title)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(log_path.with_name(filename), dpi=300)
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--mlpx-checkpoint", default="checkpoints/MLPx/best.pt")
    parser.add_argument("--experiment-dir", default=None)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

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
        run_dir = create_run_dir(DQNX_RESULTS_ROOT, "dqnx")
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "dqnx_training.csv"
    checkpoint_dir = Path("checkpoints/DQNx")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_run_metadata(run_dir, {
        "model": "Multi-request DQN policy",
        "episodes": args.episodes,
        "communication_cost": COMMUNICATION_COST,
        "mlpx_checkpoint": args.mlpx_checkpoint,
        "request_slots": MAX_REQUEST_SLOTS,
        "resume": args.resume,
    })

    environment = CustomHighwayEnv()
    observation, _ = environment.reset()
    policy = MultiRequestDQNPolicy(observation.size)
    target = MultiRequestDQNPolicy(observation.size)
    target.load_state_dict(policy.state_dict())
    target.eval()
    optimizer = torch.optim.Adam(policy.parameters(), lr=LEARNING_RATE)
    replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)
    world_model = MLPxWorldModel()
    checkpoint = torch.load(args.mlpx_checkpoint, map_location="cpu", weights_only=False)
    world_model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    world_model.eval()
    planner = MCPPlanner(world_model, environment.action_space.n, 5, DISCOUNT_FACTOR)

    epsilon = INITIAL_EPSILON
    total_steps = 0
    best_reward = float("-inf")
    start_episode = 0

    if args.resume:
        checkpoint = torch.load(
            args.resume,
            map_location="cpu",
            weights_only=False,
        )
        policy.load_state_dict(checkpoint["online_network"])
        target.load_state_dict(checkpoint["target_network"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        epsilon = checkpoint.get("exploration_probability", INITIAL_EPSILON)
        start_episode = checkpoint["episode"] + 1
        total_steps = checkpoint.get("total_steps", 0)
        best_reward = checkpoint.get("best_reward", float("-inf"))
        for transition in checkpoint.get("replay_buffer", []):
            replay_buffer.add(*transition)
        print(f"Resumed DQNx from {args.resume} at episode {start_episode}")

    append_log = args.resume is not None and log_path.exists()
    with log_path.open("a" if append_log else "w", newline="") as log_file:
        writer = csv.DictWriter(log_file, fieldnames=[
            "episode", "total_reward", "average_loss", "exploration_probability",
            "episode_length", "request_count", "communication_cost",
        ])
        if not append_log:
            writer.writeheader()

        for episode in range(start_episode, args.episodes):
            # DQNx learns request decisions; MLPx/MCP chooses driving actions.
            observation, _ = environment.reset()
            episode_reward = 0.0
            losses = []
            request_count = 0

            for step in range(environment.config["duration"]):
                request_slots = choose_request_slots(policy, observation, epsilon)
                request_count += len(request_slots)
                message = message_for_slots(environment, observation, request_slots)

                driving_action = planner.plan(observation, message)

                next_observation, driving_reward, terminated, truncated, _ = environment.step(
                    driving_action
                )

                done = terminated or truncated
                total_reward = driving_reward - len(request_slots) * COMMUNICATION_COST
                
                action_vector = [int(slot in request_slots) for slot in range(MAX_REQUEST_SLOTS)]
                replay_buffer.add(observation, action_vector, total_reward, next_observation, done)

                loss = optimize_model(replay_buffer, policy, target, optimizer)
                if loss is not None:
                    losses.append(loss)

                observation = next_observation
                episode_reward += total_reward
                total_steps += 1

                if total_steps % TARGET_UPDATE_FREQUENCY == 0:
                    target.load_state_dict(policy.state_dict())
                if done:
                    break

            epsilon = max(MIN_EPSILON, epsilon * EPSILON_DECAY)
            average_loss = sum(losses) / len(losses) if losses else 0.0
            communication_cost = request_count * COMMUNICATION_COST
            writer.writerow({
                "episode": episode + 1,
                "total_reward": episode_reward,
                "average_loss": average_loss,
                "exploration_probability": epsilon,
                "episode_length": step + 1,
                "request_count": request_count,
                "communication_cost": communication_cost,
            })
            log_file.flush()
            print(
                f"Episode {episode + 1}/{args.episodes} | reward={episode_reward:.2f} | "
                f"loss={average_loss:.4f} | requests={request_count} | epsilon={epsilon:.3f}"
            )
            checkpoint = {
                "episode": episode,
                "online_network": policy.state_dict(),
                "target_network": target.state_dict(),
                "optimizer": optimizer.state_dict(),
                "exploration_probability": epsilon,
                "total_steps": total_steps,
                "best_reward": best_reward,
                "replay_buffer": list(replay_buffer.transitions),
                "run_dir": str(run_dir),
            }
            # Keep the latest state and the best episode for later inference.
            torch.save(checkpoint, checkpoint_dir / "latest.pt")
            torch.save(checkpoint, run_dir / "latest.pt")
            if episode_reward > best_reward:
                best_reward = episode_reward
                checkpoint["best_reward"] = best_reward
                torch.save(checkpoint, checkpoint_dir / "best.pt")
                torch.save(checkpoint, run_dir / "best.pt")

    environment.close()
    save_plots(log_path)
    print(f"DQNx experiment saved to {run_dir}")


if __name__ == "__main__":
    main()
