"""Train a DQN that learns which neighboring vehicle to ask.

The frozen three-step MLPx planner chooses the ego's driving action after the
DQN selects a message request. The DQN actions are eight spatial vehicle
slots plus ask-nobody.

Run:

    .venv/bin/python -m scripts.training.train_request_dqn --episodes 1000
"""

import argparse
import csv
import random
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from scripts.data_collection.show_visible_neighbors import SLOT_ORDER, ordered_visible_ids, visible_neighbors
from src.cooperative_driving.environments.customEnv import CustomEnv
from src.cooperative_driving.models.MLPx.mlpx import MLPxWorldModel
from src.cooperative_driving.planner.MCP import MCPPlanner
from src.cooperative_driving.policies.request_dqn import RequestDQN


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


def valid_actions(visible):
    return list(range(len(ordered_visible_ids(visible)))) + [ASK_NOBODY]


def requested_message(environment, visible, action):
    if action == ASK_NOBODY:
        return None
    visible_ids = ordered_visible_ids(visible)
    messages = [None] * len(SLOT_ORDER)
    vehicle_id = visible_ids[action]
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
    states, actions, rewards, next_states, dones = zip(*batch)
    states = torch.tensor(np.asarray(states), dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.long)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    next_states = torch.tensor(np.asarray(next_states), dtype=torch.float32)
    dones = torch.tensor(dones, dtype=torch.float32)
    current = online(states).gather(1, actions[:, None]).squeeze(1)
    with torch.no_grad():
        future = target(next_states).max(dim=1).values
        expected = rewards + gamma * future * (1.0 - dones)
    loss = F.smooth_l1_loss(current, expected)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(online.parameters(), 5.0)
    optimizer.step()
    return float(loss.item())


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--planning-horizon", type=int, default=3)
    parser.add_argument("--communication-cost", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--replay-size", type=int, default=50000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output", default="checkpoints/RequestDQN/best.pt")
    parser.add_argument("--log-path", default=None,
                        help="CSV path; defaults to results/RequestDQN/experiments/<timestamp>/training.csv")
    args = parser.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    world_model = MLPxWorldModel.load("checkpoints/MLPx/best.pt")
    planner = None
    online = RequestDQN()
    target = RequestDQN(); target.load_state_dict(online.state_dict())
    optimizer = torch.optim.Adam(online.parameters(), lr=args.lr)
    replay = deque(maxlen=args.replay_size)
    rng = random.Random(args.seed)
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    run_dir = Path("results/RequestDQN/experiments") / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_request_dqn")
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log_path) if args.log_path else run_dir / "request_dqn_training.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("w", newline="")
    log = csv.DictWriter(log_file, fieldnames=[
        "episode", "return", "mean_loss", "epsilon", "requests", "crashed", "truncated"
    ])
    log.writeheader()
    epsilon = 1.0
    returns = []

    try:
        for episode in range(args.episodes):
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
                replay.append((state, action, transition_reward, next_state, terminated or truncated))
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
                print(f"Episode {episode + 1}/{args.episodes} | return={np.mean(returns[-10:]):.3f} "
                      f"| loss={np.mean(episode_losses) if episode_losses else float('nan'):.4f} "
                      f"| epsilon={epsilon:.3f}")
    finally:
        log_file.close()

    checkpoint = {"model_state_dict": online.state_dict(), "episodes": args.episodes,
                  "planning_horizon": args.planning_horizon, "communication_cost": args.communication_cost}
    torch.save(checkpoint, output)
    torch.save(checkpoint, run_dir / "best.pt")
    print(f"Saved request DQN to {output}")
    print(f"Saved training log to {log_path}")


if __name__ == "__main__":
    main()
