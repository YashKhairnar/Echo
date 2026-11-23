"""Measure whether a world model uses the neighbour message.

On each matched-intent pair, the "stay" and "merge" twins have the same public
observation. Only the neighbour message tells them apart. For every first
action, the script runs the real simulator for three steps (first action, then
IDLE) and compares it with the world model rollout:

- next-state MSE at step 1 and step 3, with messages "none" and "neighbour";
- the stay-vs-merge gap: the L2 distance between the merge and the stay
  prediction (with each twin's own message), next to the real gap;
- the gap error: the squared error of (merge - stay), predicted vs real.

With messages "none" the two twins get the same input, so the predicted gap
is 0. A model that uses the message predicts a gap close to the real gap.

    .venv/bin/python -m scripts.evaluation.evaluate_message_sensitivity \\
        --mlpx-checkpoint checkpoints/experiment_5/MLPx/best.pt --pairs 100
"""

import argparse
import copy
import json

import numpy as np
import torch

from scripts.data_collection.show_visible_neighbors import SLOT_ORDER, visible_neighbors
from src.cooperative_driving.environments.intent_scenarios import matched_intent_pair
from src.cooperative_driving.models.MLPx.per_vehicle import load_world_model

HORIZON = 3
IDLE = 1


def neighbour_bundle(environment, neighbour_id):
    """Put the neighbour message in its spatial slot, as evaluate_intent_scenarios does."""
    visible = visible_neighbors(environment, 60., 10.)
    message = environment.request_neighbor_action(neighbour_id)
    return {'messages_by_slot': [message if slot in visible and visible[slot]['vehicle'].vehicle_id == neighbour_id
                                 else None for slot in SLOT_ORDER]}


def real_rollout(model, environment, first_action):
    """Return the real states after each step (stops at a terminal step)."""
    branch = copy.deepcopy(environment)
    states = []
    try:
        for step in range(HORIZON):
            _, _, terminated, truncated, _ = branch.step(first_action if step == 0 else IDLE)
            states.append(model.observe(branch))
            if terminated or (truncated and step + 1 < HORIZON):
                break
    finally:
        branch.close()
    return np.asarray(states)


def model_rollout(model, state, first_action, message):
    states, current = [], torch.as_tensor(state, dtype=torch.float32)
    for step in range(HORIZON):
        current, _ = model.predict_step(current, first_action if step == 0 else IDLE, message)
        states.append(current.numpy().copy())
    return np.asarray(states)


def evaluate(model, seed, pairs):
    errors = {(m, k): [] for m in ('none', 'neighbour') for k in (1, HORIZON)}
    gaps = {k: {'predicted': [], 'real': [], 'gap_error': []} for k in (1, HORIZON)}
    for index in range(pairs):
        environments, metadata = matched_intent_pair(seed + index, duration=HORIZON)
        try:
            start = {mode: model.observe(env) for mode, env in environments.items()}
            np.testing.assert_array_equal(start['stay'], start['merge'])
            bundles = {mode: neighbour_bundle(env, metadata['neighbor_id']) for mode, env in environments.items()}
            for action in range(5):
                real, predicted = {}, {}
                for mode, environment in environments.items():
                    real[mode] = real_rollout(model, environment, action)
                    for name, message in (('none', None), ('neighbour', bundles[mode])):
                        rollout = model_rollout(model, start[mode], action, message)
                        predicted[mode, name] = rollout
                        for k in (1, HORIZON):
                            if len(real[mode]) >= k:
                                errors[name, k].append(float(np.mean((rollout[k - 1] - real[mode][k - 1]) ** 2)))
                for k in (1, HORIZON):
                    if len(real['stay']) >= k and len(real['merge']) >= k:
                        real_gap = real['merge'][k - 1] - real['stay'][k - 1]
                        predicted_gap = predicted['merge', 'neighbour'][k - 1] - predicted['stay', 'neighbour'][k - 1]
                        gaps[k]['real'].append(float(np.linalg.norm(real_gap)))
                        gaps[k]['predicted'].append(float(np.linalg.norm(predicted_gap)))
                        gaps[k]['gap_error'].append(float(np.mean((predicted_gap - real_gap) ** 2)))
        finally:
            for environment in environments.values():
                environment.close()
    summary = {f'mse_{name}_step{k}': float(np.mean(values)) for (name, k), values in errors.items()}
    for k, values in gaps.items():
        summary[f'real_gap_step{k}'] = float(np.mean(values['real']))
        summary[f'predicted_gap_step{k}'] = float(np.mean(values['predicted']))
        summary[f'gap_ratio_step{k}'] = summary[f'predicted_gap_step{k}'] / summary[f'real_gap_step{k}']
        summary[f'gap_error_step{k}'] = float(np.mean(values['gap_error']))
    summary['samples_step1'] = len(errors['none', 1])
    summary['samples_step3'] = len(errors['none', HORIZON])
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--mlpx-checkpoint', required=True)
    parser.add_argument('--pairs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=400000)
    parser.add_argument('--output', default=None, help='Optional JSON file for the summary')
    args = parser.parse_args()
    model = load_world_model(args.mlpx_checkpoint)
    summary = {'checkpoint': args.mlpx_checkpoint, 'architecture': type(model).__name__,
               'pairs': args.pairs, 'seed': args.seed, **evaluate(model, args.seed, args.pairs)}
    text = json.dumps(summary, indent=2)
    print(text)
    if args.output:
        with open(args.output, 'w') as file:
            file.write(text + '\n')


if __name__ == '__main__':
    main()
