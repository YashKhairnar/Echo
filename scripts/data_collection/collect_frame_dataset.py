"""Collect (state, frame) pairs for the state-to-frame decoder.

Each step saves the 55-value CustomEnv observation and the grayscale render
of the same moment (64 x 256). The ego follows the rule-based intent-aware
policy, with some random actions so that lane changes and crashes appear too.

The frames are kept apart from the JSONL world-model dataset: one raw frame
per JSONL record would need tens of gigabytes. Episodes use the same seeds
scheme (seed + episode), so a frame can be matched to a JSONL rollout state.

Run from the project root:

    .venv/bin/python -m scripts.data_collection.collect_frame_dataset \
        --episodes 1000 --experiment 3
"""

import argparse
import random
from pathlib import Path

import numpy as np

from scripts.policy.intent_aware_policy import choose_action
from scripts.utils.experiment_utils import add_experiment_argument, experiment_data_path
from src.cooperative_driving.visualization.frames import make_frame_environment, to_gray


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_experiment_argument(parser)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--random-action-probability", type=float, default=0.3)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    output = Path(args.output or experiment_data_path(args.experiment, "frame_dataset.npz"))
    output.parent.mkdir(parents=True, exist_ok=True)

    environment = make_frame_environment()
    rng = random.Random(args.seed)
    states, frames, episodes, timesteps = [], [], [], []
    crashes = 0

    for episode in range(args.episodes):
        observation, _ = environment.reset(seed=args.seed + episode)
        for timestep in range(environment.config["duration"]):
            states.append(np.asarray(observation, dtype=np.float32).reshape(-1))
            frames.append(to_gray(environment.render()))
            episodes.append(episode)
            timesteps.append(timestep)

            if rng.random() < args.random_action_probability:
                action = rng.randrange(environment.action_space.n)
            else:
                action, _ = choose_action(environment)
            observation, _, terminated, truncated, _ = environment.step(action)
            if terminated or truncated:
                # Keep the last frame too, so crashes are in the data.
                states.append(np.asarray(observation, dtype=np.float32).reshape(-1))
                frames.append(to_gray(environment.render()))
                episodes.append(episode)
                timesteps.append(timestep + 1)
                crashes += int(terminated)
                break
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{args.episodes} | frames={len(frames)}", flush=True)

    np.savez_compressed(
        output,
        states=np.stack(states),
        frames=np.stack(frames),
        episodes=np.array(episodes, dtype=np.int32),
        timesteps=np.array(timesteps, dtype=np.int32),
        seed=np.array(args.seed),
    )
    print(f"Wrote {len(frames)} frames from {args.episodes} episodes to {output}")
    print(f"Episodes ending in a crash: {crashes} ({crashes / args.episodes:.1%})")
    environment.close()


if __name__ == "__main__":
    main()
