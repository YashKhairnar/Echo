"""Collect privileged oracle request labels for request DQN training.

The ego drives with the oracle's request choice. At every step the script
stores the public 45-value request state, the oracle value of each request
option, the oracle choice, and the real transition. The training script uses
these records for a behaviour-cloning warm start and for DQfD-style
demonstrations.

Run from the project root with:

    .venv/bin/python -m scripts.data_collection.collect_oracle_requests \
        --experiment 7 --ordinary-episodes 100 --intent-scenes 500 \
        --mlpx-checkpoint checkpoints/experiment_5/MLPx/best.pt \
        --safety-guard --lane-change-guard --max-target-speed 30
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.data_collection.show_visible_neighbors import visible_neighbors
from scripts.policy.request_oracle import (
    ACTION_COUNT,
    oracle_choice,
    oracle_request_values,
    oracle_target_distribution,
)
from scripts.training.train_request_dqn import ASK_NOBODY, request_state, valid_actions
from scripts.utils.experiment_utils import add_experiment_argument, experiment_data_path
from src.cooperative_driving.environments.customEnv import CustomEnv
from src.cooperative_driving.environments.intent_scenarios import intent_episode
from src.cooperative_driving.models.MLPx.mlpx import MLPxWorldModel
from src.cooperative_driving.planner.MCP import MCPPlanner


def valid_mask(visible):
    mask = np.zeros(ACTION_COUNT, dtype=bool)
    mask[valid_actions(visible)] = True
    return mask


def oracle_episode(environment, world_model, args, controls, source, seed, mode):
    """Drive one episode with the oracle and return one record per step."""
    planner = MCPPlanner(world_model, action_size=environment.action_space.n,
                         planning_horizon=args.planning_horizon)
    records = []
    for _ in range(environment.config['duration']):
        visible = visible_neighbors(environment, args.visibility, args.side_window)
        state = request_state(environment, visible, args.request_state_encoding)
        values, first_actions = oracle_request_values(
            environment, visible, planner, world_model, controls,
            lookahead=args.lookahead, communication_cost=args.communication_cost,
            visibility=args.visibility, side_window=args.side_window)
        choice = oracle_choice(values)
        _, reward, terminated, truncated, _ = environment.step(first_actions[choice])
        next_visible = visible_neighbors(environment, args.visibility, args.side_window)
        records.append({
            'state': state,
            'values': values,
            'valid': valid_mask(visible),
            'choice': choice,
            'target': oracle_target_distribution(values),
            'reward': float(reward) - (args.communication_cost if choice != ASK_NOBODY else 0.0),
            'next_state': request_state(environment, next_visible, args.request_state_encoding),
            'done': bool(terminated or truncated),
            'next_valid': valid_mask(next_visible),
            'source': source, 'seed': seed, 'mode': mode or '',
            'crashed': bool(terminated),
        })
        if terminated or truncated:
            break
    return records


def stack(records):
    keys = ('state', 'values', 'valid', 'choice', 'target', 'reward', 'next_state',
            'done', 'next_valid', 'seed', 'crashed')
    arrays = {key: np.asarray([record[key] for record in records]) for key in keys}
    arrays['source'] = np.asarray([record['source'] for record in records])
    arrays['mode'] = np.asarray([record['mode'] for record in records])
    return arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    add_experiment_argument(parser, default=7)
    parser.add_argument('--ordinary-episodes', type=int, default=100)
    parser.add_argument('--intent-scenes', type=int, default=500,
                        help='Each scene gives two episodes: stay and merge')
    parser.add_argument('--ordinary-seed', type=int, default=1000000)
    parser.add_argument('--challenge-seed', type=int, default=1100000)
    parser.add_argument('--lookahead', type=int, default=5)
    parser.add_argument('--planning-horizon', type=int, default=3)
    parser.add_argument('--communication-cost', type=float, default=0.05)
    parser.add_argument('--visibility', type=float, default=60.0)
    parser.add_argument('--side-window', type=float, default=10.0)
    parser.add_argument('--request-state-encoding', choices=['kinematics_v1', 'ego_control_v1'],
                        default='ego_control_v1')
    parser.add_argument('--safety-guard', action='store_true')
    parser.add_argument('--lane-change-guard', action='store_true')
    parser.add_argument('--max-target-speed', type=float)
    parser.add_argument('--min-clearance', type=float, default=8.)
    parser.add_argument('--min-ttc', type=float, default=3.)
    parser.add_argument('--mlpx-checkpoint', required=True)
    parser.add_argument('--output', default=None)
    args = parser.parse_args()
    if min(args.ordinary_episodes, args.intent_scenes) < 0 or args.lookahead < 1:
        parser.error('episode counts must be nonnegative and lookahead positive')
    if (args.lane_change_guard or args.max_target_speed is not None) and not args.safety_guard:
        parser.error('lane-change-guard and max-target-speed require safety-guard')

    controls = dict(safety_guard=args.safety_guard, lane_change_guard=args.lane_change_guard,
                    max_target_speed=args.max_target_speed, min_clearance=args.min_clearance,
                    min_ttc=args.min_ttc, batched=True)
    world_model = MLPxWorldModel.load(args.mlpx_checkpoint)
    records = []
    for index in range(args.intent_scenes):
        seed = args.challenge_seed + index
        for mode in ('stay', 'merge'):
            environment = intent_episode(seed, mode)
            records.extend(oracle_episode(environment, world_model, args, controls, 'intent', seed, mode))
            environment.close()
        if (index + 1) % 50 == 0:
            print(f'intent scenes {index + 1}/{args.intent_scenes}', flush=True)
    for index in range(args.ordinary_episodes):
        seed = args.ordinary_seed + index
        environment = CustomEnv()
        environment.reset(seed=seed)
        records.extend(oracle_episode(environment, world_model, args, controls, 'ordinary', seed, None))
        environment.close()
        if (index + 1) % 10 == 0:
            print(f'ordinary episodes {index + 1}/{args.ordinary_episodes}', flush=True)
    if not records:
        parser.error('No records collected')

    arrays = stack(records)
    output = Path(args.output) if args.output else experiment_data_path(args.experiment, 'oracle_requests.npz')
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {key: getattr(args, key) for key in vars(args)}
    metadata['mlpx_sha256'] = hashlib.sha256(Path(args.mlpx_checkpoint).read_bytes()).hexdigest()
    np.savez_compressed(output, metadata=json.dumps(metadata), **arrays)
    for source in ('intent', 'ordinary'):
        selected = arrays['source'] == source
        if selected.any():
            asks = float(np.mean(arrays['choice'][selected] != ASK_NOBODY))
            print(f'{source}: {int(selected.sum())} steps, oracle ask rate {asks:.3f}')
    print(f'Saved oracle labels to {output}')


if __name__ == '__main__':
    main()
