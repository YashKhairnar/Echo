import argparse
import random
import subprocess
import time
from pathlib import Path

import pygame
import torch

from src.cooperative_driving.environments.customEnv import CustomHighwayEnv
from src.cooperative_driving.communication.request_utils import (
    requested_vehicle,
    valid_request_actions,
)
from src.cooperative_driving.models.MLP.mlp import MLP_worldModel
from src.cooperative_driving.policies.DQN import DQNPolicy
from src.cooperative_driving.planner.MCP import MCPPlanner
from scripts.train_DQN import (
    COMMUNICATION_COST,
    NO_REQUEST_ACTION,
    REQUEST_ACTION_SIZE,
    get_requested_message,
)


class VideoRecorder:
    """Stream pygame frames directly to ffmpeg."""

    def __init__(self, screen, output_path, fps):
        width, height = screen.get_size()
        self.process = subprocess.Popen(
            [
                "ffmpeg",
                "-y",
                "-f",
                "rawvideo",
                "-vcodec",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{width}x{height}",
                "-r",
                str(fps),
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(output_path),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def add_frame(self, screen):
        frame = pygame.surfarray.array3d(screen).transpose((1, 0, 2))
        self.process.stdin.write(frame.tobytes())

    def close(self):
        if self.process.stdin:
            self.process.stdin.close()
        self.process.wait()


def load_world_model(path):
    world_model = MLP_worldModel()
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    world_model.load_state_dict(state_dict)
    world_model.eval()
    return world_model


def load_request_policy(path, observation_size):
    policy = DQNPolicy(observation_size, REQUEST_ACTION_SIZE)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("online_network", checkpoint)
    policy.load_state_dict(state_dict)
    policy.eval()
    return policy


def draw_request_overlay(
    environment,
    request_action,
    requested_vehicle,
    driving_action,
):
    """Draw ego action and requested-vehicle information on screen."""
    environment.render()

    if environment.viewer is None or environment.viewer.screen is None:
        return

    action_names = getattr(environment.action_type, "actions", {})
    ego_action = action_names.get(
        int(driving_action),
        str(driving_action),
    )

    if requested_vehicle is None:
        lines = [
            "CURRENT DECISION",
            f"Ego driving action: {ego_action}",
            "Communication: no request",
        ]
    else:
        intent = requested_vehicle.intent
        lines = [
            "CURRENT DECISION",
            f"Ego driving action: {ego_action}",
            f"Requested vehicle ID: {requested_vehicle.vehicle_id}",
            f"Request slot: {request_action}",
            f"Neighbor intent: accel {intent['intent_acceleration']:.1f}, "
            f"steer {intent['intent_steering']:.1f}, "
            f"lane offset {intent['intent_lane_change']:.0f}",
        ]

    if requested_vehicle is not None:
        world_surface = environment.viewer.sim_surface
        if world_surface.is_visible(requested_vehicle.position):
            vehicle_pixel = world_surface.vec2pix(requested_vehicle.position)
            pygame.draw.circle(
                environment.viewer.screen,
                (255, 215, 0),
                vehicle_pixel,
                28,
                width=4,
            )

    font = pygame.font.Font(None, 24)
    line_height = 28
    panel_height = 14 + line_height * len(lines)
    panel = pygame.Surface((470, panel_height), pygame.SRCALPHA)
    panel.fill((0, 0, 0, 180))

    for index, line in enumerate(lines):
        text = font.render(line, True, (255, 255, 255))
        panel.blit(text, (12, 7 + index * line_height))

    environment.viewer.screen.blit(panel, (10, 10))
    pygame.display.flip()


def process_pause_events(paused):
    """Process the play/pause button and window-close events."""
    button_rect = pygame.Rect(10, 180, 150, 42)
    should_close = False

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            should_close = True
        elif (
            event.type == pygame.MOUSEBUTTONDOWN
            and event.button == 1
            and button_rect.collidepoint(event.pos)
        ):
            paused = not paused

    return paused, should_close


def draw_pause_button(environment, paused):
    """Draw a clickable play/pause button below the information overlay."""
    if environment.viewer is None or environment.viewer.screen is None:
        return

    button_rect = pygame.Rect(10, 180, 150, 42)
    button_color = (70, 150, 80) if paused else (70, 100, 160)
    pygame.draw.rect(
        environment.viewer.screen,
        button_color,
        button_rect,
        border_radius=6,
    )

    label = "PLAY" if paused else "PAUSE"
    font = pygame.font.Font(None, 24)
    text = font.render(label, True, (255, 255, 255))
    text_rect = text.get_rect(center=button_rect.center)
    environment.viewer.screen.blit(text, text_rect)
    pygame.display.flip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--delay", type=float, default=0.1)
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument(
        "--video",
        default=None,
        help="Optional MP4 output path, for example results/dqn/policy.mp4.",
    )
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional base seed for reproducible episodes.",
    )
    parser.add_argument(
        "--dqn-checkpoint",
        default="checkpoints/dqn/latest.pt",
    )
    parser.add_argument(
        "--mlp-checkpoint",
        default="checkpoints/mlp/best.pt",
    )
    args = parser.parse_args()
    base_seed = (
        args.seed
        if args.seed is not None
        else random.randint(0, 1_000_000)
    )

    dqn_checkpoint = Path(args.dqn_checkpoint)
    mlp_checkpoint = Path(args.mlp_checkpoint)

    render_config = CustomHighwayEnv.default_config()
    render_config.update({
        "screen_width": args.width,
        "screen_height": args.height,
    })
    environment = CustomHighwayEnv(
        config=render_config,
        render_mode="human",
    )
    observation, _ = environment.reset(seed=base_seed)
    environment.render()

    recorder = None
    if args.video:
        video_path = Path(args.video)
        video_path.parent.mkdir(parents=True, exist_ok=True)
        recorder = VideoRecorder(
            environment.viewer.screen,
            video_path,
            args.fps,
        )

    world_model = load_world_model(mlp_checkpoint)
    request_policy = load_request_policy(
        dqn_checkpoint,
        observation.size,
    )
    planner = MCPPlanner(
        world_model=world_model,
        action_size=environment.action_space.n,
        planning_horizon=5,
        discount_factor=0.99,
    )

    for episode in range(args.episodes):
        observation, _ = environment.reset(seed=base_seed + episode)
        total_reward = 0.0
        request_count = 0
        no_request_count = 0
        paused = False

        for step in range(environment.config["duration"]):
            while paused:
                paused, should_close = process_pause_events(paused)
                if should_close:
                    if recorder:
                        recorder.close()
                    environment.close()
                    return
                draw_pause_button(environment, paused)
                if recorder:
                    recorder.add_frame(environment.viewer.screen)
                time.sleep(0.05)

            paused, should_close = process_pause_events(paused)
            if should_close:
                if recorder:
                    recorder.close()
                environment.close()
                return

            valid_actions = valid_request_actions(observation)
            request_action = request_policy.act(observation, valid_actions)
            if request_action == NO_REQUEST_ACTION:
                no_request_count += 1
            else:
                request_count += 1

            message = get_requested_message(
                environment,
                observation,
                request_action,
            )
            requested_car = requested_vehicle(
                environment,
                observation,
                request_action,
            )
            driving_action = planner.plan(observation, message)

            observation, driving_reward, terminated, truncated, _ = environment.step(
                driving_action
            )
            total_reward += driving_reward

            draw_request_overlay(
                environment,
                request_action,
                requested_car,
                driving_action,
            )
            draw_pause_button(environment, paused)
            if recorder:
                recorder.add_frame(environment.viewer.screen)

            # Human rendering is handled by highway-env; this delay makes the
            # vehicle behavior easier to watch.
            if args.delay > 0:
                time.sleep(args.delay)

            if terminated or truncated:
                break

        communication_cost = request_count * COMMUNICATION_COST
        total_reward_after_cost = total_reward - communication_cost
        print(
            f"Episode {episode + 1}/{args.episodes} | "
            f"steps={step + 1} | "
            f"driving_reward={total_reward:.2f} | "
            f"communication_cost={communication_cost:.2f} | "
            f"total_reward={total_reward_after_cost:.2f} | "
            f"requests={request_count} | "
            f"no_requests={no_request_count}"
        )

    print(f"Render base seed: {base_seed}")
    if recorder:
        recorder.close()
        print(f"Video saved to {args.video}")
    environment.close()


if __name__ == "__main__":
    main()
