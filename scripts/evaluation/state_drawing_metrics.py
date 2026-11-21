"""Measure how well StateRenderer matches the simulator.

For random-action episodes, it reports:
- position error: decoded state vs true simulator positions [m];
- vehicle-pixel IoU: overlap of vehicle pixels in the drawn image and the
  real render (colours are ignored, because the state has no vehicle type);
- draw time per image [ms].

Run from the project root:

    .venv/bin/python -m scripts.evaluation.state_drawing_metrics --episodes 50
"""

import argparse
import json
import random
import time

import numpy as np

from src.cooperative_driving.environments.customEnv import CustomEnv
from src.cooperative_driving.visualization import StateRenderer, decode_state

ROAD_GREY = 100


def vehicle_mask(image):
    """Pixels that are neither road grey nor white lane markings."""
    image = image.astype(np.int16)
    not_road = np.abs(image - ROAD_GREY).max(axis=-1) > 25
    not_line = image.min(axis=-1) < 230
    return not_road & not_line


def iou(first, second):
    union = np.logical_or(first, second).sum()
    return float(np.logical_and(first, second).sum() / union) if union else 1.0


def position_errors(observation, environment):
    """Error of each decoded vehicle relative to the ego, and of the ego y."""
    decoded = decode_state(observation)
    ego = environment.vehicle.position
    true_relative = np.array([v.position - ego for v in environment.road.vehicles])
    decoded_ego = np.array([decoded[0]["x"], decoded[0]["y"]])
    errors = []
    for item in decoded[1:]:
        relative = np.array([item["x"], item["y"]]) - decoded_ego
        errors.append(np.linalg.norm(true_relative - relative, axis=1).min())
    return errors, abs(decoded[0]["y"] - ego[1])


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=600)
    parser.add_argument("--height", type=int, default=150)
    parser.add_argument("--scaling", type=float, default=5.5)
    args = parser.parse_args()

    screen = {"screen_width": args.width, "screen_height": args.height, "scaling": args.scaling}
    environment = CustomEnv(config=screen, render_mode="rgb_array")
    environment.reset(seed=args.seed)
    environment.render()
    environment.viewer.enabled = True
    renderer = StateRenderer(args.width, args.height, args.scaling)
    rng = random.Random(args.seed)

    neighbour_errors, ego_y_errors, ious, draw_ms, steps = [], [], [], [], 0
    for episode in range(args.episodes):
        observation, _ = environment.reset(seed=args.seed + episode)
        for _ in range(environment.config["duration"]):
            errors, ego_y_error = position_errors(observation, environment)
            neighbour_errors.extend(errors)
            ego_y_errors.append(ego_y_error)

            real = environment.render()
            start = time.perf_counter()
            drawn = renderer.draw(observation)
            draw_ms.append(1000 * (time.perf_counter() - start))
            ious.append(iou(vehicle_mask(real), vehicle_mask(drawn)))
            steps += 1

            observation, _, terminated, truncated, _ = environment.step(
                rng.randrange(environment.action_space.n)
            )
            if terminated or truncated:
                break

    results = {
        "episodes": args.episodes,
        "steps": steps,
        "vehicles_checked": len(neighbour_errors),
        "neighbour_position_error_m": {
            "mean": float(np.mean(neighbour_errors)), "max": float(np.max(neighbour_errors))
        },
        "ego_y_error_m": {"mean": float(np.mean(ego_y_errors)), "max": float(np.max(ego_y_errors))},
        "vehicle_pixel_iou": {
            "mean": float(np.mean(ious)), "min": float(np.min(ious)),
            "share_above_0.9": float(np.mean(np.array(ious) > 0.9)),
        },
        "draw_time_ms": {"mean": float(np.mean(draw_ms)), "p95": float(np.percentile(draw_ms, 95))},
    }
    print(json.dumps(results, indent=2))
    renderer.close()
    environment.close()


if __name__ == "__main__":
    main()
