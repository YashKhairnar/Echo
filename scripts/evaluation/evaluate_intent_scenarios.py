"""Evaluate matched stay/merge intentions with identical public observations.

Each candidate is one first action followed by the same public-geometry
IDLE/braking controller. Enumerating the simulator outcomes is an information
value diagnostic, not a trained controller or a world-model training dataset.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.data_collection.enumerate_message_requests import jsonable
from scripts.data_collection.show_visible_neighbors import SLOT_ORDER, visible_neighbors
from scripts.training.train_request_dqn import request_state
from scripts.utils.experiment_utils import add_experiment_argument, create_run_dir, experiment_results_dir, save_run_metadata
from src.cooperative_driving.environments.intent_scenarios import matched_intent_pair, SCENARIO_SPEC
from src.cooperative_driving.models.MLPx.per_vehicle import load_world_model
from src.cooperative_driving.planner.MCP import MCPPlanner
from src.cooperative_driving.planner.safety import guard_action


def guarded(environment, values):
    return guard_action(values, environment.vehicle, visible_neighbors(environment, 60., 10.),
                        4, check_lane_changes=True, max_target_speed=30.)[0]


def preferred_action(environment, action):
    scores = [0.] * 5
    scores[action] = 1.
    return guarded(environment, scores)


def outcome(environment, first_action, horizon):
    branch = copy.deepcopy(environment)
    total = 0.
    start = float(branch.vehicle.position[0])
    actions = []
    try:
        terminated = truncated = False
        for step in range(horizon):
            action = preferred_action(branch, first_action if step == 0 else 1)
            _, reward, terminated, truncated, _ = branch.step(action)
            actions.append(action)
            total += float(reward)
            if terminated or truncated:
                break
        return {'reward': total, 'crashed': bool(terminated), 'truncated': bool(truncated),
                'steps': len(actions), 'actions': actions,
                'progress_m': float(branch.vehicle.position[0]) - start}
    finally:
        branch.close()


def message_rule(environment, message):
    """Use only the requested target lane and the ego's own lane."""
    target = message.get('target_lane')
    entering = target is not None and int(target[2]) == environment.vehicle.lane_index[2]
    return preferred_action(environment, 4 if entering else 3)


def evaluate_pair(seed, horizon, cost, model=None):
    environments, metadata = matched_intent_pair(seed, duration=horizon)
    try:
        observations = {mode: env.observation_type.observe() for mode, env in environments.items()}
        np.testing.assert_array_equal(observations['stay'], observations['merge'])
        public_states = [request_state(env, visible_neighbors(env, 60., 10.)) for env in environments.values()]
        np.testing.assert_array_equal(*public_states)
        messages, tables = {}, {}
        for mode, environment in environments.items():
            visible = visible_neighbors(environment, 60., 10.)
            if metadata['neighbor_id'] not in [v['vehicle'].vehicle_id for v in visible.values()]:
                raise RuntimeError('Queried neighbor is not in a visible spatial slot')
            before = json.dumps(jsonable(environment.get_all_vehicle_states()), sort_keys=True)
            messages[mode] = environment.request_neighbor_action(metadata['neighbor_id'])
            after = json.dumps(jsonable(environment.get_all_vehicle_states()), sort_keys=True)
            if before != after:
                raise RuntimeError('Preview modified the physical scene')
            tables[mode] = [outcome(environment, action, horizon) for action in range(5)]
        expected = [(tables['stay'][a]['reward'] + tables['merge'][a]['reward']) / 2 for a in range(5)]
        allowed = guard_action([0.] * 5, environments['stay'].vehicle,
                               visible_neighbors(environments['stay'], 60., 10.), 4,
                               check_lane_changes=True, max_target_speed=30.)[1]['allowed_actions']
        blind_action = max(allowed, key=expected.__getitem__)
        oracle = {mode: max(allowed, key=lambda a: table[a]['reward']) for mode, table in tables.items()}
        gross = (tables['stay'][oracle['stay']]['reward'] + tables['merge'][oracle['merge']]['reward']) / 2 - expected[blind_action]
        worthwhile = gross > cost
        policies = {}
        for mode, environment in environments.items():
            rule = message_rule(environment, messages[mode])
            swapped = message_rule(environment, messages['merge' if mode == 'stay' else 'stay'])
            choices = {'geometry': (preferred_action(environment, 3), 0),
                       'conservative': (preferred_action(environment, 4), 0),
                       'informed_rule': (rule, 1), 'swapped_message_rule': (swapped, 1),
                       'blind_oracle': (blind_action, 0), 'informed_oracle': (oracle[mode], 1),
                       'selective_oracle': (oracle[mode] if worthwhile else blind_action, int(worthwhile))}
            if model is not None:
                visible = visible_neighbors(environment, 60., 10.)
                bundle = {'messages_by_slot': [messages[mode] if slot in visible and
                          visible[slot]['vehicle'].vehicle_id == metadata['neighbor_id'] else None
                          for slot in SLOT_ORDER]}
                assert sum(item is not None for item in bundle['messages_by_slot']) == 1
                planner = MCPPlanner(model, planning_horizon=3)
                model_state = model.observe(environment)
                choices['wm_none'] = (guarded(environment, planner.action_values_batched(model_state)), 0)
                choices['wm_informed'] = (guarded(environment, planner.action_values_batched(model_state, bundle)), 1)
            policies[mode] = {name: {'candidate_action': action,
                                     'first_action': tables[mode][action]['actions'][0], 'requests': requests,
                                     **tables[mode][action],
                                     'return': tables[mode][action]['reward'] - requests * cost}
                              for name, (action, requests) in choices.items()}
        assert policies['stay']['geometry']['first_action'] == policies['merge']['geometry']['first_action']
        if model is not None:
            assert policies['stay']['wm_none']['first_action'] == policies['merge']['wm_none']['first_action']
        return {**metadata, 'public_observation_sha256': hashlib.sha256(observations['stay'].tobytes()).hexdigest(),
                'observations_equal': True, 'messages': messages, 'outcomes': tables,
                'blind_oracle_action': blind_action, 'informed_oracle_actions': oracle,
                'allowed_first_actions': allowed,
                'gross_information_value': gross, 'net_information_value': gross - cost,
                'policies': policies}
    finally:
        for environment in environments.values():
            environment.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_experiment_argument(parser, default=5)
    parser.add_argument('--pairs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=400000)
    parser.add_argument('--horizon', type=int, default=3)
    parser.add_argument('--communication-cost', type=float, default=.05)
    parser.add_argument('--mlpx-checkpoint', default=None)
    parser.add_argument('--output-root', default=None)
    args = parser.parse_args()
    if min(args.pairs, args.horizon) < 1 or not np.isfinite(args.communication_cost) or args.communication_cost < 0:
        parser.error('pairs/horizon must be positive and cost finite/nonnegative')
    # The checkpoint records its architecture (flat MLPx or per_vehicle).
    model = load_world_model(args.mlpx_checkpoint) if args.mlpx_checkpoint else None
    output = create_run_dir(args.output_root or experiment_results_dir(args.experiment, 'intent_scenarios'), 'matched_intents')
    save_run_metadata(output, {**vars(args), 'hidden_prior': {'stay': .5, 'merge': .5},
                              'scenario_spec': SCENARIO_SPEC,
                              'continuation': 'IDLE plus public full guard after first decision',
                              'purpose': 'controlled benchmark; no production dataset or training',
                              'checkpoint_sha256': hashlib.sha256(Path(args.mlpx_checkpoint).read_bytes()).hexdigest() if model else None})
    pairs = []
    with (output / 'pairs.jsonl').open('w') as file:
        for index in range(args.pairs):
            pair = evaluate_pair(args.seed + index, args.horizon, args.communication_cost, model)
            pairs.append(pair)
            file.write(json.dumps(jsonable(pair), allow_nan=False) + '\n')
            if (index + 1) % 10 == 0 or index + 1 == args.pairs:
                print(f'{index + 1}/{args.pairs} pairs', flush=True)
    names = list(pairs[0]['policies']['stay'])
    summary = {'pairs': len(pairs), 'mode_episodes': 2 * len(pairs),
               'oracle_action_changes_with_message': sum(p['informed_oracle_actions']['stay'] != p['informed_oracle_actions']['merge'] for p in pairs),
               'positive_net_information_pairs': sum(p['net_information_value'] > 1e-8 for p in pairs),
               'mean_gross_information_value': float(np.mean([p['gross_information_value'] for p in pairs])),
               'mean_net_information_value': float(np.mean([p['net_information_value'] for p in pairs])),
               'policies': {}}
    for name in names:
        records = [p['policies'][mode][name] for p in pairs for mode in ['stay', 'merge']]
        summary['policies'][name] = {
            'mean_return': float(np.mean([r['return'] for r in records])),
            'mean_reward': float(np.mean([r['reward'] for r in records])),
            'crashes': sum(r['crashed'] for r in records),
            'mean_progress_m': float(np.mean([r['progress_m'] for r in records])),
            'mean_requests': float(np.mean([r['requests'] for r in records])),
            'by_mode': {mode: {'crashes': sum(p['policies'][mode][name]['crashed'] for p in pairs),
                              'mean_return': float(np.mean([p['policies'][mode][name]['return'] for p in pairs]))}
                        for mode in ['stay', 'merge']},
        }
    if model:
        summary['wm_action_changes_with_message'] = sum(
            p['policies'][mode]['wm_none']['first_action'] != p['policies'][mode]['wm_informed']['first_action']
            for p in pairs for mode in ['stay', 'merge'])
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(output, json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
