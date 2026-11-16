"""Render a request policy + the MLPx planner driving the merge scenario.

``--policy dqn`` (default) uses the trained request DQN; ``none``, ``rear``
and ``all`` are the hand-written rules from ``evaluate_merge_policies``.
"""

import argparse
import random
import subprocess
import time
from pathlib import Path

import pygame

from scripts.evaluate_merge_policies import AskAll, AskDQN, AskRear, RequestRule
from src.cooperative_driving.planner.MCP import build_planner
from src.cooperative_driving.communication.getMessage import getMessage
from src.cooperative_driving.communication.request_utils import (
    message_bundle,
    vehicle_by_id,
    visible_neighbor_ids,
)
from src.cooperative_driving.environments.customEnv import CooperativeMergeEnv


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


def asked_vehicles(environment, observation, slots):
    visible_ids = visible_neighbor_ids(observation)
    return [
        (slot, vehicle_by_id(environment, visible_ids[slot]))
        for slot in slots
        if slot < len(visible_ids)
    ]


def draw_overlay(environment, observation, slots, driving_action, title):
    environment.render()
    if environment.viewer is None or environment.viewer.screen is None:
        return

    action_names = getattr(environment.action_type, "actions", {})
    lines = [
        title,
        f"Driving action: {action_names.get(int(driving_action), driving_action)}",
    ]
    asked = asked_vehicles(environment, observation, slots)
    if not asked:
        lines.append("Asked: nobody")
    for slot, vehicle in asked:
        message = getMessage(vehicle)
        lines.append(
            f"Asked car {vehicle.vehicle_id} (slot {slot}): "
            f"{message['driver_type'] or '-'}"
        )
        world_surface = environment.viewer.sim_surface
        if world_surface.is_visible(vehicle.position):
            pygame.draw.circle(
                environment.viewer.screen,
                (255, 215, 0),
                world_surface.vec2pix(vehicle.position),
                24,
                width=4,
            )

    font = pygame.font.Font(None, 26)
    line_height = 28
    panel = pygame.Surface((520, 16 + line_height * len(lines)), pygame.SRCALPHA)
    panel.fill((0, 0, 0, 185))
    for index, line in enumerate(lines):
        panel.blit(font.render(line, True, (255, 255, 255)), (12, 8 + index * line_height))
    environment.viewer.screen.blit(panel, (10, 10))
    pygame.display.flip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--delay", type=float, default=0.3)
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=500)
    parser.add_argument("--fps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--video", default=None)
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Record the RGB scene without opening a window (no overlay).",
    )
    parser.add_argument("--policy", choices=["dqn", "none", "rear", "all"], default="dqn")
    parser.add_argument("--dqn-checkpoint", default="checkpoints/DQN/best.pt")
    parser.add_argument("--mlpx-checkpoint", default="checkpoints/MLPx/best.pt")
    args = parser.parse_args()

    base_seed = args.seed if args.seed is not None else random.randint(0, 1_000_000)
    config = CooperativeMergeEnv.default_config()
    config.update({"screen_width": args.width, "screen_height": args.height})
    environment = CooperativeMergeEnv(
        config=config, render_mode="rgb_array" if args.headless else "human"
    )
    environment.reset(seed=base_seed)
    environment.render()
    planner = build_planner(args.mlpx_checkpoint, environment)
    request_policy = {
        "dqn": lambda: AskDQN(args.dqn_checkpoint),
        "none": RequestRule,
        "rear": AskRear,
        "all": AskAll,
    }[args.policy]()
    title = f"REQUEST POLICY: {args.policy.upper()}  |  MLPx PLANNER"

    recorder = None
    if args.video:
        video_path = Path(args.video)
        video_path.parent.mkdir(parents=True, exist_ok=True)
        size = (args.width, args.height) if args.headless else environment.viewer.screen
        recorder = VideoRecorder(size, video_path, args.fps)

    for episode in range(args.episodes):
        observation, _ = environment.reset(seed=base_seed + episode)
        requests, info = 0, {}
        for step in range(environment.config["duration"]):
            slots = request_policy.slots(environment, observation)
            requests += len(slots)
            driving_action = planner.plan(
                observation, message_bundle(environment, observation, slots)
            )
            if args.headless:
                frame = environment.render()
                if recorder:
                    recorder.add_array(frame)
            else:
                draw_overlay(environment, observation, slots, driving_action, title)
                if recorder:
                    recorder.add_frame(environment.viewer.screen)
            if args.delay > 0:
                time.sleep(args.delay)
            observation, _, terminated, truncated, info = environment.step(driving_action)
            if terminated or truncated:
                break
        print(
            f"Episode {episode + 1}/{args.episodes} | seed={base_seed + episode} | "
            f"steps={step + 1} | requests={requests} | "
            f"crashed={info.get('crashed')} | merged={info.get('merged')}"
        )

    if recorder:
        recorder.close()
        print(f"Video saved to {args.video}")
    environment.close()


if __name__ == "__main__":
    main()
