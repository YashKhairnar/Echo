"""Collect versioned ego-control states and paired committed-intent trees.

Challenge trees cover all five commands at every depth through three seconds.
The initial response/slot context is frozen over each tree, matching MCP's
message reuse. Ordinary traffic contributes fresh one-step counterfactuals.
All branches and hidden variants share their source episode split group.
"""
import argparse
import copy
import json
import multiprocessing
import random
from concurrent.futures import ProcessPoolExecutor

from scripts.data_collection.enumerate_message_requests import jsonable
from scripts.data_collection.show_visible_neighbors import SLOT_ORDER, visible_neighbors, spatial_vehicle_slots
from scripts.policy.intent_aware_policy import choose_action
from scripts.utils.experiment_utils import add_experiment_argument, create_run_dir, experiment_data_path, experiment_results_dir, save_run_metadata
from src.cooperative_driving.environments.customEnv import CustomEnv
from src.cooperative_driving.environments.intent_scenarios import matched_intent_pair, SCENARIO_SPEC
def state(environment):
    # Pure state encoding; no model weights or hidden neighbor controls.
    import numpy as np
    return np.concatenate([environment.observation_type.observe().reshape(-1),
                           [environment.vehicle.target_speed / 30., environment.vehicle.target_lane_index[2] / 3.]]).astype(np.float32).tolist()


def context(environment, neighbor_id=None):
    visible = visible_neighbors(environment, 60., 10.)
    slots = spatial_vehicle_slots(visible)
    ids = [item for item in slots if item is not None]
    if neighbor_id is not None:
        if neighbor_id not in ids:
            raise RuntimeError('Challenge request is outside visible slots')
        ids = [neighbor_id]
    messages = {str(i): environment.request_neighbor_action(i) for i in ids}
    return slots, messages


def records_for_edge(environment, branch, action, result, source, episode, seed, slots, messages, mode, depth):
    observation, reward, terminated, truncated, info = result
    base = {'dataset_source': source, 'episode': episode, 'seed': seed,
            'state_encoding': 'kinematics_ego_control_v1', 'message_slot_encoding': 'spatial_v1',
            'state': state(environment), 'next_state': state(branch), 'ego_action': action,
            'reward': float(reward), 'terminated': bool(terminated), 'truncated': bool(truncated),
            'visible_vehicle_slots': slots, 'hidden_mode': mode, 'depth': depth,
            'message_context': 'initial_decision_frozen' if source == 'matched_intent' else 'current_decision'}
    return [json.dumps(jsonable({**base, 'messages': bundle}), allow_nan=False) + '\n'
            for bundle in ([{}, messages] if messages else [{}])]


def tree(environment, depth, source, episode, seed, slots, messages, mode, records):
    for action in range(5):
        branch = copy.deepcopy(environment)
        try:
            result = branch.step(action)
            records.extend(records_for_edge(environment, branch, action, result, source, episode, seed, slots, messages, mode, depth))
            if depth < 2 and not (result[2] or result[3]):
                tree(branch, depth + 1, source, episode, seed, slots, messages, mode, records)
        finally:
            branch.close()


def collect_task(task):
    source, episode, seed = task
    records = []
    if source == 'matched_intent':
        environments, metadata = matched_intent_pair(seed, duration=3)
        try:
            for mode, environment in environments.items():
                slots, messages = context(environment, metadata['neighbor_id'])
                tree(environment, 0, source, episode, seed, slots, messages, mode, records)
        finally:
            for environment in environments.values():
                environment.close()
    else:
        environment = CustomEnv()
        environment.reset(seed=seed)
        rng = random.Random(seed)
        try:
            for timestep in range(environment.config['duration']):
                slots, messages = context(environment)
                for action in range(5):
                    branch = copy.deepcopy(environment)
                    try:
                        result = branch.step(action)
                        records.extend(records_for_edge(environment, branch, action, result, source, episode, seed, slots, messages, None, timestep))
                    finally:
                        branch.close()
                action = choose_action(environment, messages={})[0] if episode % 2 == 0 else rng.randrange(5)
                _, _, terminated, truncated, _ = environment.step(action)
                if terminated or truncated:
                    break
        finally:
            environment.close()
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_experiment_argument(parser, default=5)
    parser.add_argument('--pairs', type=int, default=200)
    parser.add_argument('--ordinary-episodes', type=int, default=100)
    parser.add_argument('--seed', type=int, default=500000)
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    if min(args.pairs, args.workers) < 1 or args.ordinary_episodes < 0:
        parser.error('pairs/workers must be positive; ordinary-episodes nonnegative')
    output = experiment_data_path(args.experiment, 'combined_world_model_dataset.jsonl')
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        parser.error(f'Refusing to overwrite {output}')
    run = create_run_dir(experiment_results_dir(args.experiment, 'data_collection'), 'intent_control_states')
    save_run_metadata(run, {**vars(args), 'output': str(output), 'state_encoding': 'kinematics_ego_control_v1',
                            'scenario_spec': SCENARIO_SPEC, 'split_group': '(dataset_source, episode)',
                            'challenge_context': 'frozen initial response and slots throughout each three-step action tree'})
    tasks = [('matched_intent', i, args.seed+i) for i in range(args.pairs)]
    tasks += [('ordinary_traffic', i, args.seed+100000+i) for i in range(args.ordinary_episodes)]
    counts = {'matched_intent': 0, 'ordinary_traffic': 0}
    with output.open('x') as file, ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn')) as executor:
        for index, records in enumerate(executor.map(collect_task, tasks, buffersize=args.workers*2)):
            file.writelines(records)
            counts[tasks[index][0]] += len(records)
            if (index+1) % 10 == 0 or index+1 == len(tasks):
                print(f'{index+1}/{len(tasks)} groups; {sum(counts.values())} records', flush=True)
    (run/'summary.json').write_text(json.dumps(counts, indent=2))
    print(output, counts, flush=True)


if __name__ == '__main__':
    main()
