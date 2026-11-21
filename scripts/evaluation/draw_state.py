"""Compare the real render with the image drawn from the state vector.

For each step, the top image is the real CustomEnv render and the bottom image
is drawn only from the 55-value observation. Use this to check StateRenderer.

Run from the project root:

    .venv/bin/python -m scripts.evaluation.draw_state --seed 0 --steps 6
"""

import argparse
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.cooperative_driving.environments.customEnv import CustomEnv
from src.cooperative_driving.visualization import StateRenderer

SEPARATOR = 4  # white pixels between images


def stack(images, axis):
    """Join images with a thin white line between them."""
    height, width, _ = images[0].shape
    shape = (SEPARATOR, width, 3) if axis == 0 else (height, SEPARATOR, 3)
    gap = np.full(shape, 255, dtype=np.uint8)
    parts = []
    for image in images:
        parts.extend([image, gap])
    return np.concatenate(parts[:-1], axis=axis)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--width", type=int, default=600)
    parser.add_argument("--height", type=int, default=150)
    parser.add_argument("--scaling", type=float, default=5.5)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    output = Path(args.output or f"results/state_drawing/seed{args.seed}.png")
    output.parent.mkdir(parents=True, exist_ok=True)

    screen = {"screen_width": args.width, "screen_height": args.height, "scaling": args.scaling}
    environment = CustomEnv(config=screen, render_mode="rgb_array")
    observation, _ = environment.reset(seed=args.seed)
    environment.render()
    environment.viewer.enabled = True
    renderer = StateRenderer(args.width, args.height, args.scaling)
    rng = random.Random(args.seed)

    columns = []
    for _ in range(args.steps):
        columns.append(stack([environment.render(), renderer.draw(observation)], axis=0))
        observation, _, terminated, truncated, _ = environment.step(
            rng.randrange(environment.action_space.n)
        )
        if terminated or truncated:
            break

    plt.imsave(output, stack(columns, axis=0))
    print(f"Saved {len(columns)} steps to {output} (top: real, bottom: drawn)")
    renderer.close()
    environment.close()


if __name__ == "__main__":
    main()
