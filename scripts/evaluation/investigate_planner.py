"""Replay diagnostic crash approaches and compare learned search with simulator search.

Simulator search is a privileged diagnostic oracle, never a deployable policy.
Run against an existing diagnose_driving directory. All branches share the same
copied scene; future rewards are zero after termination or truncation.
"""

import argparse
import copy
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

from scripts.data_collection.enumerate_message_requests import jsonable
from scripts.utils.experiment_utils import create_run_dir
from src.cooperative_driving.environments.customEnv import CustomEnv
from src.cooperative_driving.models.MLPx.mlpx import MLPxWorldModel


def real_search(environment, horizon, discount=.99):
    """Return per-first-action best real returns and their sequences."""
    results = []
    for action in range(environment.action_space.n):
        branch = copy.deepcopy(environment)
        try:
            _, reward, terminated, truncated, _ = branch.step(action)
            sequence = [action]
            if horizon > 1 and not (terminated or truncated):
                children = real_search(branch, horizon - 1, discount)
                best = max(children, key=lambda item: item['value'])
                reward += discount * best['value']
                sequence += best['sequence']
            results.append({'value': float(reward), 'sequence': sequence})
        finally:
            branch.close()
    return results


def simulate(environment, sequence, discount=.99):
    branch = copy.deepcopy(environment)
    total, rewards = 0., []
    terminated = truncated = False
    try:
        for depth, action in enumerate(sequence):
            _, reward, terminated, truncated, _ = branch.step(action)
            rewards.append(float(reward))
            total += discount ** depth * reward
            if terminated or truncated:
                break
        return {'value': float(total), 'rewards': rewards,
                'crashed': bool(terminated), 'truncated': bool(truncated)}
    finally:
        branch.close()


def learned_search(model, observation, horizon, discount=.99):
    sequences = list(itertools.product(range(5), repeat=horizon))
    states = np.tile(np.asarray(observation).reshape(1, -1), (len(sequences), 1))
    values = np.zeros(len(sequences))
    for depth in range(horizon):
        states, rewards = model.predict_batch(states, [s[depth] for s in sequences], None)
        values += discount ** depth * rewards.numpy()
    return [max(({'value': float(values[i]), 'sequence': list(sequence)}
                 for i, sequence in enumerate(sequences) if sequence[0] == action),
                key=lambda item: item['value']) for action in range(5)]


def controller_alias_probes(seed=100000):
    """Demonstrate omitted controller memory using identical input observations."""
    environment = CustomEnv()
    try:
        observation, _ = environment.reset(seed=seed)
        lane = int(environment.vehicle.lane_index[2])
        probes = []
        for target_speed, target_lane in [(20, lane), (30, lane), (25, (lane + 1) % 4)]:
            branch = copy.deepcopy(environment)
            try:
                branch.vehicle.target_speed = target_speed
                branch.vehicle.target_lane_index = (*branch.vehicle.lane_index[:2], target_lane)
                before = branch.observation_type.observe()
                after, reward, _, _, _ = branch.step(1)
                probes.append({'target_speed': target_speed, 'target_lane': target_lane,
                               'input_equal': bool(np.array_equal(observation, before)),
                               'next_state': after.tolist(), 'reward': float(reward),
                               'next_speed': float(branch.vehicle.speed)})
            finally:
                branch.close()
        return probes
    finally:
        environment.close()


def audit_cases(model, cases, traces):
    """Separate candidate-ranking errors from future rewards after real termination."""
    for case in cases:
        action = case['learned_action']
        state = traces[case['seed'], case['timestep']]['state']
        rewards = []
        for candidate in case['predicted'][action]['sequence']:
            state, reward = model.predict_step(state, candidate)
            rewards.append(reward)
        realized = case['realized_learned_sequences'][action]
        count = len(realized['rewards'])
        case['predicted_step_rewards'] = rewards
        case['predicted_return_after_real_terminal'] = sum(.99**i * r for i, r in enumerate(rewards) if i >= count)
        case['real_best_among_learned_sequences'] = max(
            range(5), key=lambda a: case['realized_learned_sequences'][a]['value'])
    return {
        'danger_state_ranking_agreement': float(np.mean([
            c['learned_action'] == c['real_best_among_learned_sequences'] for c in cases])),
        'first_action_regret_above_005': sum(c['first_action_gap'] > .05 for c in cases),
        'crashing_selected_sequences': sum(c['learned_sequence_crashes'] for c in cases),
        'predicted_future_after_real_terminal_total': sum(c['predicted_return_after_real_terminal'] for c in cases),
        'selected_sequences_with_future_after_terminal': sum(c['predicted_return_after_real_terminal'] > 0 for c in cases),
        'controller_alias_probes': controller_alias_probes(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diagnostic-dir', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--horizon', type=int, default=3)
    parser.add_argument('--steps-before-crash', type=int, default=3)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.horizon <= 4 or args.steps_before_crash < 1:
        parser.error('horizon must be 1–4 and steps-before-crash must be positive')
    config = json.loads((args.diagnostic_dir / 'config.json').read_text())
    traces = {}
    for line in (args.diagnostic_dir / 'steps.jsonl').open():
        row = json.loads(line)
        if row['policy'] == 'wm_none':
            traces.setdefault(row['seed'], []).append(row)
    model = MLPxWorldModel.load(args.checkpoint)
    output = create_run_dir(args.output_root, 'planner_investigation')
    (output / 'config.json').write_text(json.dumps(
        {**jsonable(vars(args)),
         'checkpoint_sha256': hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
         'selection': 'wm_none crashes only; one scene per episode, steps-before-crash decisions before termination',
         'oracle': 'privileged simulator tree search; zero future return after termination/truncation'},
        indent=2, default=str))
    cases = []
    for seed, rows in sorted(traces.items()):
        if not rows[-1]['terminated']:
            continue
        index = max(0, len(rows) - args.steps_before_crash)
        environment = CustomEnv(config=config['environment_config'])
        try:
            observation, _ = environment.reset(seed=seed)
            for row in rows[:index]:
                observation, _, terminated, truncated, _ = environment.step(row['action'])
                if terminated or truncated:
                    raise RuntimeError('Replay ended before selected scene')
            observation = model.observe(environment)
            np.testing.assert_allclose(observation, np.asarray(rows[index]['state']).reshape(-1), atol=1e-6)
            predicted = learned_search(model, observation, args.horizon)
            oracle = real_search(environment, args.horizon)
            action = max(range(5), key=lambda a: predicted[a]['value'])
            oracle_action = max(range(5), key=lambda a: oracle[a]['value'])
            realized = [simulate(environment, item['sequence']) for item in predicted]
            oracle_realized = simulate(environment, oracle[oracle_action]['sequence'])
            case = {'seed': seed, 'timestep': index, 'learned_action': action,
                    'oracle_action': oracle_action, 'predicted': predicted,
                    'oracle': oracle, 'realized_learned_sequences': realized,
                    'oracle_selected_outcome': oracle_realized,
                    'selected_return_optimism': predicted[action]['value'] - realized[action]['value'],
                    'oracle_return_gap': oracle[oracle_action]['value'] - realized[action]['value'],
                    'first_action_gap': oracle[oracle_action]['value'] - oracle[action]['value'],
                    'learned_sequence_crashes': realized[action]['crashed'],
                    'oracle_sequence_crashes': oracle_realized['crashed']}
            cases.append(case)
            print(f"seed={seed} learned={action} oracle={oracle_action} "
                  f"crashes={case['learned_sequence_crashes']}/{case['oracle_sequence_crashes']} "
                  f"optimism={case['selected_return_optimism']:.3f}", flush=True)
        finally:
            environment.close()
    if not cases:
        raise RuntimeError('No wm_none crash approaches found')
    trace_lookup = {(seed, row['timestep']): row for seed, rows in traces.items() for row in rows}
    audit = audit_cases(model, cases, trace_lookup)
    (output / 'cases.json').write_text(json.dumps(cases, indent=2))
    (output / 'audit.json').write_text(json.dumps(audit, indent=2))
    summary = {'cases': len(cases),
               'first_action_agreement': float(np.mean([r['learned_action'] == r['oracle_action'] for r in cases])),
               'learned_sequence_crashes': sum(r['learned_sequence_crashes'] for r in cases),
               'oracle_sequence_crashes': sum(r['oracle_sequence_crashes'] for r in cases),
               'avoidable_within_horizon': sum(r['learned_sequence_crashes'] and not r['oracle_sequence_crashes'] for r in cases),
               **{key: float(np.mean([r[key] for r in cases])) for key in
                  ['selected_return_optimism', 'oracle_return_gap', 'first_action_gap']}}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(output, json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
