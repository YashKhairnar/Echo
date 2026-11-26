"""Render the trained request DQN with the three-step MLPx planner.

Run from the project root:

    .venv/bin/python -m scripts.evaluation.render_request_dqn --episodes 1
"""

import argparse
import shutil
import subprocess
import time
import random
import math
import warnings
from pathlib import Path

import pygame
import torch

from scripts.data_collection.show_visible_neighbors import ordered_visible_ids, visible_neighbors
from scripts.training.train_request_dqn import (
    ASK_NOBODY,
    request_state,
    requested_message,
    resolve_requested_vehicle,
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
from src.cooperative_driving.environments.intent_scenarios import intent_episode
from src.cooperative_driving.planner.controller import driving_action as choose_driving
from src.cooperative_driving.policies import load_request_policy


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
    parser.add_argument("--duration", type=int, default=None,
                        help="Override the environment episode length in decision steps")
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
    parser.add_argument("--safety-guard", action="store_true", help="Enable the optional geometry braking filter")
    parser.add_argument("--min-clearance", type=float, default=8.)
    parser.add_argument("--min-ttc", type=float, default=3.)
    parser.add_argument('--comfort-ttc', type=float, default=None,
                        help='Proactively change lanes or slow before the emergency TTC threshold')
    parser.add_argument('--lane-change-guard', action='store_true')
    parser.add_argument('--max-target-speed', type=float, default=None)
    parser.add_argument('--batched-planner', action='store_true',
                        help='Use the batched planner, as in training and evaluate_request_dqn')
    parser.add_argument('--scenario', choices=['ordinary', 'paired-intent'], default='ordinary',
                        help='paired-intent renders the matched-intent challenge scene of --seed + episode')
    parser.add_argument('--hidden-mode', choices=['stay', 'merge'], default='merge',
                        help='Hidden neighbor intent for --scenario paired-intent')
    parser.add_argument('--policy', choices=['dqn', 'none'], default='dqn',
                        help='none renders ask nobody with the same planner and guard')
    args = parser.parse_args()
    if (args.lane_change_guard or args.max_target_speed is not None) and not args.safety_guard:
        parser.error('lane-change-guard and max-target-speed require safety-guard')
    if args.comfort_ttc is not None and (not args.safety_guard or not args.lane_change_guard):
        parser.error('comfort-ttc requires safety-guard and lane-change-guard')
    if args.max_target_speed is not None and (not math.isfinite(args.max_target_speed) or args.max_target_speed <= 0):
        parser.error('max-target-speed must be finite and positive')
    if args.duration is not None and args.duration < 1:
        parser.error('duration must be positive')
    if args.comfort_ttc is not None and (not math.isfinite(args.comfort_ttc) or args.comfort_ttc <= args.min_ttc):
        parser.error('comfort-ttc must be finite and greater than min-ttc')

    if args.video and args.no_render:
        parser.error("--video requires rendering; remove --no-render")
    if not all(math.isfinite(v) and v > 0 for v in (args.min_clearance, args.min_ttc)):
        parser.error("guard thresholds must be finite and positive")

    args.dqn_checkpoint = args.dqn_checkpoint or str(
        experiment_checkpoint_dir(args.experiment, "RequestDQN") / "best.pt"
    )
    dqn = load_request_policy(args.dqn_checkpoint)
    args.mlpx_checkpoint = args.mlpx_checkpoint or dqn.mlpx_checkpoint or str(
        experiment_checkpoint_dir(args.experiment, "MLPx") / "best.pt"
    )

    if dqn.training_config.get('safety_guard') and not args.safety_guard:
        warnings.warn('DQN was trained with a safety guard. Pass its guard flags to reproduce the trained controller.')
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
            render_mode = "rgb_array" if video_writer else None if args.no_render else "human"
            if args.scenario == 'paired-intent':
                # Same scene builder and seed as evaluate_request_dqn. The
                # viewer is created at the first render, so setting the
                # render configuration now gives the requested frame size.
                environment = intent_episode(args.seed + episode, args.hidden_mode)
                environment.config.update(screen_width=args.width, screen_height=args.height,
                                          scaling=args.scaling)
                if args.duration is not None:
                    environment.config["duration"] = args.duration
                environment.render_mode = render_mode
            else:
                environment_config = {
                    "screen_width": args.width,
                    "screen_height": args.height,
                    "scaling": args.scaling,
                }
                if args.duration is not None:
                    environment_config["duration"] = args.duration
                environment = CustomEnv(
                    config=environment_config,
                    render_mode=render_mode,
                )
                environment.reset(seed=args.seed + episode)
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
            guard_overrides = 0
            controller_state = {} if args.comfort_ttc is not None else None
            episode_outcome = None
            speeds = []
            policy_label = (
                "ddqn"
                if args.policy == "dqn" and getattr(dqn, "POLICY_TYPE", "") == "request_ddqn"
                else args.policy
            )
            label = f"{policy_label} | {args.scenario}" + (
                f" {args.hidden_mode}" if args.scenario == 'paired-intent' else "")
            print(f"Episode {episode + 1}/{args.episodes} ({label}, seed {args.seed + episode})")

            for timestep in range(environment.config["duration"]):
                observation = world_model.observe(environment)
                visible = visible_neighbors(environment, 60.0, 10.0)
                if args.policy == 'none':
                    request_action = ASK_NOBODY
                else:
                    state = request_state(environment, visible, dqn.request_state_encoding)
                    request_action = select_action(
                        dqn, state, valid_actions(visible, dqn.request_action_encoding), epsilon=0.0, rng=rng
                    )
                message = requested_message(environment, visible, request_action, dqn.request_action_encoding)
                speed = float(environment.vehicle.speed)
                speeds.append(speed)
                # The shared controller is the one training and evaluation use.
                driving_action, guard = choose_driving(
                    planner, environment, observation, message, visible=visible,
                    safety_guard=args.safety_guard, min_clearance=args.min_clearance,
                    min_ttc=args.min_ttc, lane_change_guard=args.lane_change_guard,
                    max_target_speed=args.max_target_speed, comfort_ttc=args.comfort_ttc,
                    batched=args.batched_planner, controller_state=controller_state)
                guard_overrides += int(bool(guard and guard["overridden"]))
                guard_label = " | guard: " + (", ".join(guard["reasons"]) or "pass") if guard else ""
                visible_ids = ordered_visible_ids(visible)
                requests_sent += int(request_action != ASK_NOBODY)
                requested_vehicle = resolve_requested_vehicle(
                    visible, request_action, dqn.request_action_encoding
                )

                requested = "none" if requested_vehicle is None else str(requested_vehicle.vehicle_id)
                print(
                    f"  step={timestep:02d} speed={speed:.2f} visible={len(visible_ids)} "
                    f"request={requested} driving={ACTION_NAMES[driving_action]}{guard_label}"
                )

                if not args.no_render:
                    frame = render_frame(
                        environment, requested_vehicle, message,
                        f"{label} | step {timestep} | speed {speed:.2f} m/s | ego action: {ACTION_NAMES[driving_action]}{guard_label}"
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
                            f"{label} | {episode_outcome} | steps {steps_completed} | speed {environment.vehicle.speed:.2f} m/s | reward {total_reward:.3f} | requests {requests_sent} | guard overrides {guard_overrides}"
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
                f"reward={total_reward:.3f} requests_sent={requests_sent} guard_overrides={guard_overrides} "
                f"mean_speed={sum(speeds) / len(speeds):.2f} final_speed={environment.vehicle.speed:.2f}"
            )
            environment.close()
    finally:
        if video_writer:
            video_writer.close()
            print(f"Saved video to {video_writer.path}")


if __name__ == "__main__":
    main()
