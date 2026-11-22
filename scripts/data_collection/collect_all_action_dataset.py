"""Collect all five counterfactual ego actions per state and message scenario.

Defaults to no-message, each single-vehicle request, and all-visible requests.
Every action starts from the same copied scene. Rollouts mix rule and random
episodes to include both ordinary traffic and approaching-danger states.
"""

import argparse
import copy
import itertools
import json
import multiprocessing
import random
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from pathlib import Path

from scripts.data_collection.enumerate_message_requests import jsonable
from scripts.data_collection.show_visible_neighbors import ordered_visible_ids, spatial_vehicle_slots, visible_neighbors
from scripts.policy.intent_aware_policy import ACTION_NAMES, choose_action
from scripts.utils.experiment_utils import (
    add_experiment_argument, create_run_dir, experiment_data_path,
    experiment_results_dir, save_run_metadata,
)
from src.cooperative_driving.environments.customEnv import CustomEnv


def request_scenarios(ids, modes):
    """Unique request subsets; do not count all/single duplicates twice."""
    requests = []
    if "none" in modes:
        requests.append(())
    if "single" in modes:
        requests.extend((vehicle_id,) for vehicle_id in ids)
    if "all" in modes:
        requests.append(tuple(ids))
    if "subsets" in modes:
        for size in range(1, len(ids) + 1):
            requests.extend(itertools.combinations(ids, size))
    return list(dict.fromkeys(requests))


def action_records(environment, observation, episode, seed, timestep, modes, visibility, side_window):
    visible = visible_neighbors(environment, visibility, side_window)
    ids = ordered_visible_ids(visible)
    slots = spatial_vehicle_slots(visible)
    requests = request_scenarios(ids, modes)
    needed_ids = {vehicle_id for subset in requests for vehicle_id in subset}
    previews = {vehicle_id: environment.request_neighbor_action(vehicle_id) for vehicle_id in needed_ids}
    ego = environment.vehicle
    if not requests:
        return
    # Requests are model inputs, not simulator interventions. Reusing the same
    # five physical outcomes across message bundles is exactly equivalent to
    # independently stepping a fresh copy for every (bundle, action) pair.
    outcomes = []
    for action in range(environment.action_space.n):
        scenario = copy.deepcopy(environment)
        try:
            next_state, reward, terminated, truncated, info = scenario.step(action)
            outcomes.append((next_state, float(reward), bool(terminated), bool(truncated), info,
                             int(scenario.vehicle.lane_index[2]), int(scenario.vehicle.target_lane_index[2])))
        finally:
            scenario.close()
    for subset in requests:
        messages = {vehicle_id: previews[vehicle_id] for vehicle_id in subset}
        for action, outcome in enumerate(outcomes):
            next_state, reward, terminated, truncated, info, next_lane, next_target = outcome
            lane = int(ego.lane_index[2])
            target = int(ego.target_lane_index[2])
            destination = target + (-1 if action == 0 else 1 if action == 2 else 0)
            yield {
                    "dataset_source": "all_action_counterfactual", "episode": episode,
                    "seed": seed, "timestep": timestep, "scene_id": f"{seed}:{timestep}",
                    "goal_lane_id": environment.goal_lane_id, "state": observation,
                    "visible_vehicle_ids": ids, "visible_vehicle_slots": slots,
                    "message_slot_encoding": "spatial_v1",
                    "requested_vehicle_ids": list(subset), "messages": messages or None,
                    "ego_action": action, "ego_action_name": ACTION_NAMES[action],
                    "next_state": next_state, "reward": float(reward),
                    "terminated": bool(terminated), "truncated": bool(truncated), "info": info,
                    "boundary_lane_command": action in (0, 2) and not 0 <= destination < environment.config["lanes_count"],
                    "ego_lane_before": lane, "ego_target_lane_before": target,
                    "ego_lane_after": next_lane,
                    "ego_target_lane_after": next_target,
                }


def collect_episode(task):
    """A seed-local rollout makes parallel and serial collection identical."""
    episode, config = task
    seed = config["seed"] + episode
    environment = CustomEnv()
    rng = random.Random(seed)
    counts, modes = Counter(), Counter()
    lines = []
    crashes = truncations = steps = 0
    terminated = truncated = False
    rollout = config["rollout_policy"]
    if rollout == "mixed":
        rollout = rng.choice(["rule", "random"])
    try:
        observation, _ = environment.reset(seed=seed)
        limit = min(config["steps"] or environment.config["duration"], environment.config["duration"])
        for timestep in range(limit):
            for record in action_records(environment, observation, episode, seed, timestep,
                                         config["message_modes"], config["visibility"], config["side_window"]):
                record["rollout_policy"] = rollout
                lines.append(json.dumps(jsonable(record), allow_nan=False) + "\n")
                crashes += int(record["terminated"])
                truncations += int(record["truncated"])
                counts[record["ego_action_name"]] += 1
                modes["with_messages" if record["messages"] else "no_messages"] += 1
            if rollout == "random":
                action = rng.randrange(environment.action_space.n)
            else:
                action, _ = choose_action(environment, config["visibility"], config["side_window"], messages={})
            observation, _, terminated, truncated, _ = environment.step(action)
            steps += 1
            if terminated or truncated:
                break
    finally:
        environment.close()
    return {"text": "".join(lines), "records": len(lines), "actions": dict(counts),
            "message_modes": dict(modes), "rollout": rollout, "crashes": crashes,
            "truncations": truncations, "steps": steps,
            "actual_crashed": int(terminated), "actual_truncated": int(truncated)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_experiment_argument(parser)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--seed", type=int, default=200000)
    parser.add_argument("--message-modes", nargs="+", choices=["none", "single", "all", "subsets"], default=["none", "single", "all"])
    parser.add_argument("--rollout-policy", choices=["rule", "random", "mixed"], default="mixed")
    parser.add_argument("--visibility", type=float, default=60.)
    parser.add_argument("--side-window", type=float, default=10.)
    parser.add_argument("--output")
    parser.add_argument("--run-root", help="Optional metadata/results root (useful for smoke tests)")
    parser.add_argument("--workers", type=int, default=1, help="Independent episode workers; output remains in seed order")
    args = parser.parse_args()
    if min(args.episodes, args.workers) < 1 or (args.steps is not None and args.steps < 1):
        parser.error("episodes, workers, and steps must be positive")
    output = Path(args.output or experiment_data_path(args.experiment, "combined_world_model_dataset.jsonl"))
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        parser.error(f"Refusing to overwrite {output}; choose another output or experiment")
    run = create_run_dir(Path(args.run_root or experiment_results_dir(args.experiment, "data_collection")), "all_actions")
    counts, modes, rollout_counts = Counter(), Counter(), Counter()
    records = crashes = truncations = actual_crashes = actual_truncations = actual_steps = 0
    started = time.monotonic()
    save_run_metadata(run, {**vars(args), "output": str(output), "environment_config": CustomEnv.default_config(),
                            "rollout_rng": "per_episode_seed_v1",
                            "counterfactual": "all actions including boundary no-ops; all start from an identical scene"})
    pool = (ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"))
            if args.workers > 1 else nullcontext(None))
    with pool as executor:
        tasks = ((episode, vars(args)) for episode in range(args.episodes))
        results = (executor.map(collect_episode, tasks, buffersize=args.workers * 2)
                   if executor else map(collect_episode, tasks))
        with output.open("x") as file:
            for episode, result in enumerate(results):
                file.write(result["text"])
                records += result["records"]
                crashes += result["crashes"]
                truncations += result["truncations"]
                actual_steps += result["steps"]
                actual_crashes += result["actual_crashed"]
                actual_truncations += result["actual_truncated"]
                counts.update(result["actions"])
                modes.update(result["message_modes"])
                rollout_counts[result["rollout"]] += 1
                file.flush()
                print(f"Episode {episode + 1}/{args.episodes} rollout={result['rollout']} records={records:,} "
                      f"elapsed={time.monotonic() - started:.1f}s", flush=True)
    summary = {"records": records, "actions": dict(counts), "message_modes": dict(modes),
               "counterfactual_crash_flags": crashes, "counterfactual_truncation_flags": truncations,
               "actual_steps": actual_steps, "actual_crashes": actual_crashes,
               "actual_truncations": actual_truncations, "rollout_episodes": dict(rollout_counts)}
    (run / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Wrote {records:,} records to {output}\n" + json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
