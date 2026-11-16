"""Train the request DQN: each step, ask one car or nobody.

    .venv/bin/python -m scripts.train_DQN --episodes 3000 --communication-cost 0.05

Each step:
1. the DQN picks whom to ask (one visible car, or nobody);
2. the asked car's message goes to the MLPx planner, which picks the driving action;
3. the DQN's reward = driving reward - communication cost.

MLPx and the planner are frozen; only the DQN learns. Saves
checkpoints/DQN/best.pt and the run's log and plots to
results/DQN/experiments/<time>_dqn_cost<cost>/.
"""

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
from src.cooperative_driving.communication.request_utils import (
    NO_REQUEST_ACTION,
    message_bundle,
    valid_request_actions,
    vehicle_by_id,
    visible_neighbor_ids,
)
from src.cooperative_driving.environments.customEnv import TARGET_LANE, make_environment
from src.cooperative_driving.planner.MCP import build_planner
from src.cooperative_driving.policies.DQN import RequestDQNPolicy

# Learning
BATCH_SIZE = 64
REPLAY_SIZE = 50_000
LEARNING_RATE = 1e-4
DISCOUNT = 0.99
N_STEP = 3                 # learn from 3-step returns: a question pays off a few steps later
LEARNING_STARTS = 1_000    # steps collected before the first update
TARGET_UPDATE_STEPS = 500  # copy the online network to the target network this often

# Exploration (epsilon-greedy)
EPSILON_START, EPSILON_END = 1.0, 0.05
EXPLORATION_FRACTION = 0.6     # epsilon falls linearly over this share of episodes
EXPLORE_ASK_NOBODY = 0.5       # random step: ask nobody this often
EXPLORE_ASK_TARGET_LANE = 0.8  # ...otherwise ask a target-lane car this often, else any car

RESULTS_ROOT = Path("results/DQN/experiments")
CHECKPOINT_DIR = Path("checkpoints/DQN")
LOG_FIELDS = ["episode", "driving_reward", "total_reward", "communication_cost",
              "request_count", "episode_length", "crashed", "merged", "epsilon", "loss"]


class ReplayBuffer:
    """Stores N-step transitions: (obs, action, discounted N-step reward,
    obs N steps later, crashed, discount to apply to that later obs's value)."""

    def __init__(self):
        self.transitions = deque(maxlen=REPLAY_SIZE)
        self.recent = deque()  # the last < N steps that are not stored yet

    def add_step(self, observation, action, reward, next_observation, crashed, episode_over):
        self.recent.append((observation, action, reward))
        if len(self.recent) == N_STEP:
            self._store_oldest(next_observation, crashed)
        if episode_over:
            while self.recent:
                self._store_oldest(next_observation, crashed)

    def _store_oldest(self, next_observation, crashed):
        observation, action, _ = self.recent[0]
        n_step_reward = sum(DISCOUNT ** i * reward for i, (_, _, reward) in enumerate(self.recent))
        self.transitions.append(
            (observation, action, n_step_reward, next_observation, float(crashed), DISCOUNT ** len(self.recent))
        )
        self.recent.popleft()

    def sample(self):
        return random.sample(self.transitions, BATCH_SIZE)

    def __len__(self):
        return len(self.transitions)


def epsilon_at(episode, total_episodes):
    progress = min(1.0, episode / (EXPLORATION_FRACTION * total_episodes))
    return EPSILON_START + progress * (EPSILON_END - EPSILON_START)


def choose_request(policy, environment, observation, epsilon):
    """Epsilon-greedy. Random steps mostly ask cars in the lane the ego wants
    to enter; asking any of ~9 cars at random almost never asks the one that
    matters, so the DQN would never see a question pay off. Which car in that
    lane to ask is still for the DQN to learn."""
    valid = valid_request_actions(observation)
    if random.random() >= epsilon:
        return policy.act(observation, valid)
    if random.random() < EXPLORE_ASK_NOBODY:
        return NO_REQUEST_ACTION
    target_lane_slots = target_lane_cars(environment, observation)
    if target_lane_slots and random.random() < EXPLORE_ASK_TARGET_LANE:
        return random.choice(target_lane_slots)
    cars = [action for action in valid if action != NO_REQUEST_ACTION]
    return random.choice(cars) if cars else NO_REQUEST_ACTION


def target_lane_cars(environment, observation):
    """Slots of visible cars in the target lane (empty once the ego has merged)."""
    if environment.vehicle.lane_index[2] == TARGET_LANE:
        return []
    slots = []
    for slot, vehicle_id in enumerate(visible_neighbor_ids(observation)):
        vehicle = vehicle_by_id(environment, vehicle_id)
        if vehicle is not None and vehicle.lane_index[2] == TARGET_LANE:
            slots.append(slot)
    return slots


def learn(buffer, online, target, optimizer):
    """One Double-DQN update: the online network picks the next action,
    the target network scores it."""
    observations, actions, rewards, next_observations, crashed, discounts = (
        torch.as_tensor(np.asarray(column), dtype=torch.float32) for column in zip(*buffer.sample())
    )
    q = online(observations).gather(1, actions.long()[:, None]).squeeze(1)
    with torch.no_grad():
        next_actions = online(next_observations).argmax(dim=1, keepdim=True)
        next_q = target(next_observations).gather(1, next_actions).squeeze(1)
        target_q = rewards + discounts * next_q * (1.0 - crashed)
    loss = F.smooth_l1_loss(q, target_q)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(online.parameters(), 10.0)
    optimizer.step()
    return loss.item()


def save_plots(log_path):
    rows = list(csv.DictReader(log_path.open()))
    episodes = np.array([int(row["episode"]) for row in rows])
    plots = {
        "dqn_rewards.png": ["driving_reward", "total_reward"],
        "dqn_requests.png": ["request_count"],
        "dqn_safety.png": ["crashed", "merged"],
        "dqn_loss.png": ["loss"],
    }
    for filename, columns in plots.items():
        plt.figure(figsize=(10, 5))
        for column in columns:
            values = np.array([float(row[column] or "nan") for row in rows])
            smoothed = np.convolve(values, np.ones(50) / 50, mode="valid")
            plt.plot(episodes, values, alpha=0.25)
            plt.plot(episodes[len(episodes) - len(smoothed):], smoothed, label=f"{column} (50-episode mean)")
        plt.xlabel("Episode")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(log_path.with_name(filename), dpi=150)
        plt.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--communication-cost", type=float, default=0.05)
    parser.add_argument("--mlpx-checkpoint", default="checkpoints/MLPx/best.pt")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    run_dir = create_run_dir(RESULTS_ROOT, f"dqn_cost{args.communication_cost:g}")
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    save_run_metadata(run_dir, {"model": "Request DQN (one car or nobody per step)", **vars(args),
                                "n_step": N_STEP, "learning_rate": LEARNING_RATE})

    environment = make_environment()
    planner = build_planner(args.mlpx_checkpoint, environment)
    online, target = RequestDQNPolicy(), RequestDQNPolicy()
    target.load_state_dict(online.state_dict())
    optimizer = torch.optim.Adam(online.parameters(), lr=LEARNING_RATE)
    buffer = ReplayBuffer()

    total_steps, best_score = 0, float("-inf")
    recent_totals = deque(maxlen=100)
    log_path = run_dir / "dqn_training.csv"
    with log_path.open("w", newline="") as log_file:
        log = csv.DictWriter(log_file, fieldnames=LOG_FIELDS)
        log.writeheader()

        for episode in range(args.episodes):
            epsilon = epsilon_at(episode, args.episodes)
            observation, _ = environment.reset(seed=args.seed * 1_000_000 + episode)
            driving_total, requests, losses, info = 0.0, 0, [], {}

            for step in range(environment.config["duration"]):
                request = choose_request(online, environment, observation, epsilon)
                asked = [] if request == NO_REQUEST_ACTION else [request]
                action = planner.plan(observation, message_bundle(environment, observation, asked))
                next_observation, driving_reward, crashed, time_up, info = environment.step(action)

                reward = driving_reward - args.communication_cost * len(asked)
                # Only a crash stops the value from continuing; the time limit does not.
                buffer.add_step(observation, request, reward, next_observation, crashed,
                                episode_over=crashed or time_up)
                if len(buffer) >= LEARNING_STARTS:
                    losses.append(learn(buffer, online, target, optimizer))
                total_steps += 1
                if total_steps % TARGET_UPDATE_STEPS == 0:
                    target.load_state_dict(online.state_dict())

                driving_total += driving_reward
                requests += len(asked)
                observation = next_observation
                if crashed or time_up:
                    break

            cost = args.communication_cost * requests
            crashed = bool(info.get("crashed", False))
            log.writerow({
                "episode": episode + 1,
                "driving_reward": driving_total,
                "total_reward": driving_total - cost,
                "communication_cost": cost,
                "request_count": requests,
                "episode_length": step + 1,
                "crashed": int(crashed),
                "merged": int(info.get("merged", False) and not crashed),
                "epsilon": epsilon,
                "loss": np.mean(losses) if losses else "",
            })
            log_file.flush()

            # Keep the latest model, and the one with the best 100-episode
            # average once exploration is mostly over.
            recent_totals.append(driving_total - cost)
            score = float(np.mean(recent_totals))
            checkpoint = {"episode": episode, "online_network": online.state_dict(),
                          "communication_cost": args.communication_cost,
                          "mlpx_checkpoint": args.mlpx_checkpoint, "run_dir": str(run_dir)}
            torch.save(checkpoint, run_dir / "latest.pt")
            if epsilon <= 0.2 and len(recent_totals) == 100 and score > best_score:
                best_score = score
                torch.save(checkpoint, run_dir / "best.pt")
                torch.save(checkpoint, CHECKPOINT_DIR / "best.pt")

            if (episode + 1) % 50 == 0:
                print(f"Episode {episode + 1}/{args.episodes} | epsilon={epsilon:.2f} | "
                      f"total(100)={score:.2f} | requests={requests}", flush=True)

    save_plots(log_path)
    print(f"Saved run to {run_dir}")
    environment.close()


if __name__ == "__main__":
    main()
