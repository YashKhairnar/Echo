"""Render the trained request DQN with the three-step MLPx planner.

Run from the project root:

    .venv/bin/python -m scripts.evaluation.render_request_dqn --episodes 1
"""

import argparse
import shutil
import subprocess
import time
import random
from pathlib import Path

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
from scripts.utils.experiment_utils import (
    add_experiment_argument,
    experiment_checkpoint_dir,
)
from src.cooperative_driving.environments.customEnv import CustomEnv
from src.cooperative_driving.models.MLPx.mlpx import MLPxWorldModel
from src.cooperative_driving.planner.MCP import MCPPlanner
from src.cooperative_driving.policies.request_dqn import RequestDQN


ACTION_NAMES = ["LANE_LEFT", "IDLE", "LANE_RIGHT", "FASTER", "SLOWER"]


class VideoWriter:
    """Stream pygame RGB frames to an MP4 through the local ffmpeg binary."""

    def __init__(self, path, width, height, fps):
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("Video recording requires ffmpeg on PATH")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.process = subprocess.Popen(
            [
                ffmpeg,
                "-y",
                "-f",
                "rawvideo",
                "-pixel_format",
                "rgb24",
                "-video_size",
                f"{width}x{height}",
                "-framerate",
                str(fps),
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(self.path),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    def write(self, frame):
        self.process.stdin.write(frame)

    def close(self):
        if self.process.stdin:
            self.process.stdin.close()
        stderr = self.process.stderr.read().decode(errors="replace")
        return_code = self.process.wait()
        if return_code:
            raise RuntimeError(f"ffmpeg failed while writing {self.path}: {stderr}")


def draw_request_overlay(surface, requested_vehicle, message, status):
    """Display the exact response already supplied to the driving planner."""
    if not pygame.font.get_init():
        pygame.font.init()
    font = pygame.font.Font(None, max(18, min(28, surface.get_width() // 55)))
    lines = [status]
    if requested_vehicle is None:
        lines += ["Request: nobody", "Response: None"]
    else:
        response = next((item for item in (message or {}).get("messages_by_slot", []) if item), {})
        target_lane = response.get("target_lane")
        lane = str(target_lane[2]) if target_lane is not None else "n/a"
        def number(key):
            value = response.get(key)
            return f"{float(value):.3f}" if value is not None else "n/a"
        lines += [
            f"Request: vehicle {requested_vehicle.vehicle_id} (yellow circle)",
            f"Response: {response.get('behavior_type', 'unknown')} | target lane: {lane} (0-based) | crashed: {response.get('crashed', False)}",
            f"Target speed: {number('target_speed')} m/s | acceleration: {number('acceleration')} m/s^2 | steering: {number('steering')} rad",
        ]
    rendered = [font.render(line, True, (248, 250, 252)) for line in lines]
    line_height = font.get_linesize() + 5
    panel = pygame.Surface((surface.get_width() - 24, len(lines) * line_height + 20), pygame.SRCALPHA)
    panel.fill((16, 24, 40, 235))
    surface.blit(panel, (12, 12))
    for index, line in enumerate(rendered):
        surface.blit(line, (24, 22 + index * line_height))


def render_frame(environment, requested_vehicle=None, message=None, status=""):
    """Render the scene and circle the vehicle whose message was requested."""
    environment.render()
    if environment.viewer is None:
        return None
    surface = environment.viewer.sim_surface
    if requested_vehicle is not None and surface.is_visible(requested_vehicle.position):
        center = surface.vec2pix(requested_vehicle.position)
        radius = max(18, surface.pix(2.2))
        pygame.draw.circle(surface, (255, 255, 0), center, radius, width=3)
        if not pygame.font.get_init():
            pygame.font.init()
        label = pygame.font.Font(None, 24).render(
            f"ID {requested_vehicle.vehicle_id}", True, (255, 255, 0)
        )
        surface.blit(label, (center[0] - label.get_width() // 2, center[1] - radius - 24))
    draw_request_overlay(surface, requested_vehicle, message, status)
    if not environment.viewer.offscreen:
        environment.viewer.screen.blit(surface, (0, 0))
        pygame.display.flip()
        capture_surface = environment.viewer.screen
    else:
        capture_surface = surface
    return pygame.image.tobytes(capture_surface, "RGB")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_experiment_argument(parser)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--delay", type=float, default=0.15)
    parser.add_argument("--width", type=int, default=1800)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--scaling", type=float, default=6.0)
    parser.add_argument("--dqn-checkpoint", default=None)
    parser.add_argument("--mlpx-checkpoint", default=None)
    parser.add_argument("--planning-horizon", type=int, default=3)
    parser.add_argument("--video", default=None, help="Optional MP4 output path")
    parser.add_argument("--video-fps", type=int, default=8)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()

    if args.video and args.no_render:
        parser.error("--video requires rendering; remove --no-render")

    args.dqn_checkpoint = args.dqn_checkpoint or str(
        experiment_checkpoint_dir(args.experiment, "RequestDQN") / "best.pt"
    )
    args.mlpx_checkpoint = args.mlpx_checkpoint or str(
        experiment_checkpoint_dir(args.experiment, "MLPx") / "best.pt"
    )

    dqn = RequestDQN.load(args.dqn_checkpoint)
    world_model = MLPxWorldModel.load(args.mlpx_checkpoint)
    rng = random.Random(args.seed)
    video_writer = (
        VideoWriter(args.video, args.width, args.height, args.video_fps)
        if args.video
        else None
    )

    try:
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
                render_mode=(
                    "rgb_array"
                    if video_writer
                    else None if args.no_render else "human"
                ),
            )
            observation, _ = environment.reset(seed=args.seed + episode)
            if video_writer:
                # RGB-array mode does not create a viewer during reset.
                # Create it once, then enable its offscreen drawing surface.
                environment.render()
            if video_writer and environment.viewer is not None:
                # HighwayEnv disables drawing when SDL uses the dummy driver.
                # Video mode is offscreen, so explicitly enable the simulator
                # surface even when no visible window is present.
                environment.viewer.enabled = True
            planner = MCPPlanner(
                world_model,
                action_size=environment.action_space.n,
                planning_horizon=args.planning_horizon,
            )
            total_reward = 0.0
            requests_sent = 0
            steps_completed = 0
            episode_outcome = None
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
                requests_sent += int(request_action != ASK_NOBODY)
                requested_vehicle = None
                if request_action != ASK_NOBODY:
                    requested_vehicle = environment.get_vehicle(visible_ids[request_action])

                requested = "none" if request_action == ASK_NOBODY else str(visible_ids[request_action])
                print(
                    f"  step={timestep:02d} visible={len(visible_ids)} "
                    f"request={requested} driving={ACTION_NAMES[driving_action]}"
                )

                if not args.no_render:
                    frame = render_frame(
                        environment, requested_vehicle, message,
                        f"Episode {episode + 1} | step {timestep} | ego action: {ACTION_NAMES[driving_action]}"
                    )
                    if video_writer:
                        video_writer.write(frame)
                    time.sleep(args.delay)

                observation, reward, terminated, truncated, _ = environment.step(driving_action)
                total_reward += float(reward)
                steps_completed += 1
                if terminated or truncated:
                    episode_outcome = "CRASHED" if terminated else "DROVE TO END"
                    if not args.no_render:
                        frame = render_frame(
                            environment, requested_vehicle, message,
                            f"{episode_outcome} | steps {steps_completed} | reward {total_reward:.3f} | requests {requests_sent} | last request/response:"
                        )
                        if video_writer:
                            video_writer.write(frame)
                        time.sleep(args.delay)
                    break

            if episode_outcome is None:
                episode_outcome = (
                    "DROVE TO END"
                    if steps_completed >= environment.config["duration"]
                    else "STOPPED"
                )
            print(
                f"  episode summary: steps={steps_completed}/{environment.config['duration']} "
                f"outcome={episode_outcome} "
                f"crashed={'yes' if episode_outcome == 'CRASHED' else 'no'} "
                f"reward={total_reward:.3f} requests_sent={requests_sent}"
            )
            environment.close()
    finally:
        if video_writer:
            video_writer.close()
            print(f"Saved video to {video_writer.path}")


if __name__ == "__main__":
    main()
