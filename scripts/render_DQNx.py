import argparse
import random
import subprocess
import time
from pathlib import Path

import pygame
import torch

from src.cooperative_driving.communication.getMessage import getMessage
from src.cooperative_driving.communication.request_utils import visible_neighbor_ids
from src.cooperative_driving.environments.customEnv import CustomHighwayEnv
from src.cooperative_driving.models.MLPx.mlpx import MLPxWorldModel
from src.cooperative_driving.planner.MCP import MCPPlanner
from src.cooperative_driving.policies.DQN import MultiRequestDQNPolicy


MAX_REQUEST_SLOTS = 10


class VideoRecorder:
    def __init__(self, screen, output_path, fps):
        if isinstance(screen, tuple):
            width, height = screen
        else:
            width, height = screen.get_size()
        self.process = subprocess.Popen(
            [
                "ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
                "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps),
                "-i", "-", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                str(output_path),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def add_frame(self, screen):
        frame = pygame.surfarray.array3d(screen).transpose((1, 0, 2))
        self.process.stdin.write(frame.tobytes())

    def add_array(self, frame):
        """Write an RGB array returned by rgb_array rendering."""
        self.process.stdin.write(frame.astype("uint8").tobytes())

    def close(self):
        if self.process.stdin:
            self.process.stdin.close()
        self.process.wait()


def load_model(model, path, key):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get(key, checkpoint.get("model_state_dict", checkpoint))
    model.load_state_dict(state_dict)
    model.eval()
    return model


def requested_vehicles(environment, observation, requested_slots):
    visible_ids = visible_neighbor_ids(observation)
    vehicles = []
    for slot in requested_slots:
        if slot >= len(visible_ids):
            continue
        vehicle = next(
            (
                car for car in environment.road.vehicles
                if car.vehicle_id == visible_ids[slot]
            ),
            None,
        )
        if vehicle is not None:
            vehicles.append((slot, vehicle))
    return vehicles


def message_for_slots(environment, observation, requested_slots):
    visible_ids = visible_neighbor_ids(observation)
    request_mask = [0] * MAX_REQUEST_SLOTS
    messages = [[0.0] * 4 for _ in range(MAX_REQUEST_SLOTS)]
    for slot in requested_slots:
        if slot >= len(visible_ids):
            continue
        vehicle = next(
            (
                car for car in environment.road.vehicles
                if car.vehicle_id == visible_ids[slot]
            ),
            None,
        )
        if vehicle is None:
            continue
        message = getMessage(vehicle)
        request_mask[slot] = 1
        messages[slot] = [
            message["intent"]["intent_acceleration"],
            message["intent"]["intent_steering"],
            message["target_lane"],
            message["time_to_collision"],
        ]
    return {"request_mask": request_mask, "messages": messages}


def draw_overlay(environment, observation, requested_slots, driving_action):
    environment.render()
    if environment.viewer is None or environment.viewer.screen is None:
        return

    action_names = getattr(environment.action_type, "actions", {})
    action_name = action_names.get(int(driving_action), str(driving_action))
    selected = requested_vehicles(environment, observation, requested_slots)
    lines = [
        "DQNx TTC CURRENT DECISION",
        f"Ego driving action: {action_name}",
        f"Requested cars: {len(selected)}",
    ]
    for slot, vehicle in selected[:5]:
        message = getMessage(vehicle)
        intent = vehicle.intent
        lines.append(
            f"slot {slot} | ID {vehicle.vehicle_id} | "
            f"acc {intent['intent_acceleration']:.1f} | "
            f"steer {intent['intent_steering']:.1f} | "
            f"lane {message['target_lane']} | "
            f"TTC {message['time_to_collision']:.1f}s"
        )
        world_surface = environment.viewer.sim_surface
        if world_surface.is_visible(vehicle.position):
            pixel = world_surface.vec2pix(vehicle.position)
            pygame.draw.circle(
                environment.viewer.screen,
                (255, 215, 0),
                pixel,
                24,
                width=4,
            )

    font = pygame.font.Font(None, 24)
    line_height = 27
    panel = pygame.Surface((620, 18 + line_height * len(lines)), pygame.SRCALPHA)
    panel.fill((0, 0, 0, 185))
    for index, line in enumerate(lines):
        panel.blit(font.render(line, True, (255, 255, 255)), (12, 7 + index * line_height))
    environment.viewer.screen.blit(panel, (10, 10))
    pygame.display.flip()


def main():
    parser = argparse.ArgumentParser(
        description="Render the TTC-message multi-request DQNx policy."
    )
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--delay", type=float, default=0.1)
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--video", default=None)
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Record the RGB scene without opening a rendering window.",
    )
    parser.add_argument("--dqnx-checkpoint", default="checkpoints/DQNx/best.pt")
    parser.add_argument("--mlpx-checkpoint", default="checkpoints/MLPx/best.pt")
    args = parser.parse_args()

    base_seed = args.seed if args.seed is not None else random.randint(0, 1_000_000)
    config = CustomHighwayEnv.default_config()
    config.update({"screen_width": args.width, "screen_height": args.height})
    render_mode = "rgb_array" if args.headless else "human"
    environment = CustomHighwayEnv(config=config, render_mode=render_mode)
    observation, _ = environment.reset(seed=base_seed)
    environment.render()

    mlpx = load_model(MLPxWorldModel(), args.mlpx_checkpoint, "model_state_dict")
    dqnx = load_model(
        MultiRequestDQNPolicy(observation.size),
        args.dqnx_checkpoint,
        "online_network",
    )
    planner = MCPPlanner(mlpx, environment.action_space.n, 5, 0.99)

    recorder = None
    if args.video:
        video_path = Path(args.video)
        video_path.parent.mkdir(parents=True, exist_ok=True)
        if args.headless:
            recorder = VideoRecorder((args.width, args.height), video_path, args.fps)
        else:
            recorder = VideoRecorder(environment.viewer.screen, video_path, args.fps)

    for episode in range(args.episodes):
        observation, _ = environment.reset(seed=base_seed + episode)
        last_requested_ids = []
        for step in range(environment.config["duration"]):
            visible_slots = list(range(len(visible_neighbor_ids(observation))))
            # DQNx selects visible request slots; MLPx/MCP selects driving action.
            requested_slots = dqnx.act(observation, visible_slots)
            last_requested_ids = [
                vehicle.vehicle_id
                for _, vehicle in requested_vehicles(
                    environment, observation, requested_slots
                )
            ]
            message = message_for_slots(environment, observation, requested_slots)
            driving_action = planner.plan(observation, message)
            if args.headless:
                # Capture the actual highway scene even when no GUI is available.
                frame = environment.render()
                if recorder:
                    recorder.add_array(frame)
            else:
                draw_overlay(environment, observation, requested_slots, driving_action)
                if recorder:
                    recorder.add_frame(environment.viewer.screen)
            if args.delay > 0:
                time.sleep(args.delay)
            observation, _, terminated, truncated, _ = environment.step(driving_action)
            if terminated or truncated:
                break

        print(
            f"Episode {episode + 1}/{args.episodes} | "
            f"steps={step + 1} | requested_slots={requested_slots} | "
            f"requested_ids={last_requested_ids}"
        )

    if recorder:
        recorder.close()
        print(f"Video saved to {args.video}")
    environment.close()


if __name__ == "__main__":
    main()
