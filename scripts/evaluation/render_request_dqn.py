"""Render the trained request DQN with the three-step MLPx planner.

Run from the project root:

    .venv/bin/python -m scripts.evaluation.render_request_dqn --episodes 1
"""

import argparse
import time
import random

import pygame
import torch

from scripts.data_collection.show_visible_neighbors import ordered_visible_ids, visible_neighbors
from scripts.training.train_request_dqn import (
    ASK_NOBODY,
    request_state,
    requested_message,
    select_action,
    valid_actions,
)
from src.cooperative_driving.environments.customEnv import CustomEnv
from src.cooperative_driving.models.MLPx.mlpx import MLPxWorldModel
from src.cooperative_driving.planner.MCP import MCPPlanner
from src.cooperative_driving.policies.request_dqn import RequestDQN


ACTION_NAMES = ["LANE_LEFT", "IDLE", "LANE_RIGHT", "FASTER", "SLOWER"]


def render_frame(environment, requested_vehicle=None):
    """Render the scene and circle the vehicle whose message was requested."""
    environment.render()
    if requested_vehicle is None or environment.viewer is None:
        return
    surface = environment.viewer.sim_surface
    if not surface.is_visible(requested_vehicle.position):
        return
    center = surface.vec2pix(requested_vehicle.position)
    radius = max(18, surface.pix(2.2))
    pygame.draw.circle(surface, (255, 255, 0), center, radius, width=3)
    if not environment.viewer.offscreen:
        environment.viewer.screen.blit(surface, (0, 0))
        pygame.display.flip()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--delay", type=float, default=0.15)
    parser.add_argument("--width", type=int, default=1800)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--scaling", type=float, default=6.0)
    parser.add_argument("--dqn-checkpoint", default="checkpoints/RequestDQN/best.pt")
    parser.add_argument("--mlpx-checkpoint", default="checkpoints/MLPx/best.pt")
    parser.add_argument("--planning-horizon", type=int, default=3)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()

    dqn = RequestDQN.load(args.dqn_checkpoint)
    world_model = MLPxWorldModel.load(args.mlpx_checkpoint)
    rng = random.Random(args.seed)

    for episode in range(args.episodes):
        # Pass the render configuration during construction. CustomEnv performs
        # an initial reset in its constructor, so configuring afterward would
        # leave the first pygame viewer at HighwayEnv's small default size.
        environment = CustomEnv(
            config={
                "screen_width": args.width,
                "screen_height": args.height,
                "scaling": args.scaling,
            },
            render_mode=None if args.no_render else "human",
        )
        observation, _ = environment.reset(seed=args.seed + episode)
        planner = MCPPlanner(
            world_model,
            action_size=environment.action_space.n,
            planning_horizon=args.planning_horizon,
        )
        total_reward = 0.0
        print(f"Episode {episode + 1}/{args.episodes}")

        for timestep in range(environment.config["duration"]):
            visible = visible_neighbors(environment, 60.0, 10.0)
            state = request_state(environment, visible)
            request_action = select_action(
                dqn, state, valid_actions(visible), epsilon=0.0, rng=rng
            )
            message = requested_message(environment, visible, request_action)
            driving_action = planner.plan(observation, message)
            visible_ids = ordered_visible_ids(visible)
            requested_vehicle = None
            if request_action != ASK_NOBODY:
                requested_vehicle = environment.get_vehicle(visible_ids[request_action])

            requested = "none" if request_action == ASK_NOBODY else str(visible_ids[request_action])
            print(
                f"  step={timestep:02d} visible={len(visible_ids)} "
                f"request={requested} driving={ACTION_NAMES[driving_action]}"
            )

            if not args.no_render:
                render_frame(environment, requested_vehicle)
                time.sleep(args.delay)

            observation, reward, terminated, truncated, _ = environment.step(driving_action)
            total_reward += float(reward)
            if terminated or truncated:
                if not args.no_render:
                    render_frame(environment, requested_vehicle)
                    time.sleep(args.delay)
                print(
                    f"  ended: {'crash' if terminated else 'truncated'} "
                    f"total_reward={total_reward:.3f}"
                )
                break

        environment.close()


if __name__ == "__main__":
    main()
