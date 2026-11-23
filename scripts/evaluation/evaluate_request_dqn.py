"""Evaluate the Request DQN against communication baselines.

All policies use the same fixed environment seeds. The DQN chooses one visible
vehicle or asks nobody; the planner then chooses the ego driving action. The
evaluation reports driving reward, communication-adjusted return, requests,
crashes, and truncations.

Run from the project root with:

    .venv/bin/python -m scripts.evaluation.evaluate_request_dqn \
        --experiment 1 --episodes 200
"""

import argparse
import csv
import hashlib
import json
import random
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

from scripts.data_collection.show_visible_neighbors import SLOT_ORDER, ordered_visible_ids, visible_neighbors
from scripts.data_collection.enumerate_message_requests import jsonable
from scripts.training.train_request_dqn import (
    ASK_NOBODY,
    request_state,
    requested_message,
    select_action,
    valid_actions,
)
from scripts.policy.request_oracle import agrees_with_oracle, oracle_choice, oracle_request_values
from scripts.utils.experiment_utils import (
    add_experiment_argument,
    experiment_checkpoint_dir,
    experiment_results_dir,
)
from src.cooperative_driving.environments.customEnv import CustomEnv
from src.cooperative_driving.models.MLPx.mlpx import MLPxWorldModel
from src.cooperative_driving.planner.MCP import MCPPlanner
from src.cooperative_driving.planner.safety import guard_action
from src.cooperative_driving.planner.controller import driving_action as choose_driving
from src.cooperative_driving.environments.intent_scenarios import intent_episode, SCENARIO_SPEC
from src.cooperative_driving.policies.request_dqn import RequestDQN


def all_messages(environment, visible):
    messages = [None] * len(SLOT_ORDER)
    for index, slot in enumerate(SLOT_ORDER):
        candidate = visible.get(slot)
        if candidate is not None:
            messages[index] = environment.request_neighbor_action(
                candidate["vehicle"].vehicle_id
            )
    return {"messages_by_slot": messages}


def oracle_controls(args):
    """Planner controls the oracle shares with the evaluated controller."""
    return dict(safety_guard=args.safety_guard, min_clearance=args.min_clearance,
                min_ttc=args.min_ttc, lane_change_guard=args.lane_change_guard,
                max_target_speed=args.max_target_speed,
                comfort_ttc=getattr(args, 'comfort_ttc', None),
                batched=getattr(args, 'batched_planner', False))


def choose_request(policy, dqn, environment, visible, rng):
    visible_ids = ordered_visible_ids(visible)
    if policy == "none" or not visible_ids:
        return ASK_NOBODY, None
    if policy == "all":
        return len(visible_ids), all_messages(environment, visible)
    if policy == "random":
        action = rng.choice([a for a in valid_actions(visible) if a != ASK_NOBODY])
        return action, requested_message(environment, visible, action)

    state = request_state(environment, visible, getattr(dqn, 'request_state_encoding', 'kinematics_v1'))
    encoding = dqn.request_action_encoding
    action = select_action(dqn, state, valid_actions(visible, encoding), epsilon=0.0, rng=rng)
    return action, requested_message(environment, visible, action, encoding)


def evaluate_policy(policy, dqn, world_model, args):
    rows = []
    policy_offsets = {"dqn": 0, "none": 1000, "random": 2000, "all": 3000, "oracle": 4000}
    rng = random.Random(args.seed + policy_offsets[policy])
    scenario = getattr(args, 'scenario', 'ordinary')
    cases = [(index, mode) for index in range(args.episodes)
             for mode in (['stay', 'merge'] if scenario == 'paired-intent' else [None])]
    for episode, (scene_index, hidden_mode) in enumerate(cases):
        environment_seed = args.seed + scene_index
        if hidden_mode is not None:
            environment = intent_episode(environment_seed, hidden_mode)
            observation = environment.observation_type.observe()
            rng = random.Random(environment_seed + policy_offsets[policy])
        else:
            environment = CustomEnv()
            observation, _ = environment.reset(seed=environment_seed)
        initial_signature = hashlib.sha256(json.dumps(jsonable(environment.get_all_vehicle_states()), sort_keys=True).encode()).hexdigest()
        planner = MCPPlanner(
            world_model,
            action_size=environment.action_space.n,
            planning_horizon=args.planning_horizon,
        )
        total_return = 0.0
        driving_reward = 0.0
        requests = 0
        guard_overrides = 0
        terminated = False
        truncated = False
        speeds = []
        completed_lane_changes = 0
        settled_lane = int(environment.vehicle.lane_index[2])
        controller_state = {} if getattr(args, 'comfort_ttc', None) is not None else None
        initial_request = initial_driving = None
        oracle_steps = oracle_agreements = oracle_asks = 0

        for timestep in range(environment.config["duration"]):
            observation = world_model.observe(environment)
            speeds.append(float(environment.vehicle.speed))
            visible = visible_neighbors(
                environment,
                args.visibility,
                args.side_window,
            )
            oracle_values = None
            if policy == 'oracle' or getattr(args, 'oracle_agreement', False):
                oracle_values, _ = oracle_request_values(
                    environment, visible, planner, world_model, oracle_controls(args),
                    lookahead=args.oracle_lookahead, communication_cost=args.communication_cost,
                    visibility=args.visibility, side_window=args.side_window)
            if policy == 'oracle':
                request_action = oracle_choice(oracle_values)
                message = requested_message(environment, visible, request_action)
            else:
                request_action, message = choose_request(
                    policy, dqn, environment, visible, rng
                )
            if oracle_values is not None and policy != 'all':
                oracle_steps += 1
                oracle_agreements += int(agrees_with_oracle(request_action, oracle_values))
                oracle_asks += int(oracle_choice(oracle_values) != ASK_NOBODY)
            requested_count = (
                len(ordered_visible_ids(visible))
                if policy == "all"
                else int(request_action != ASK_NOBODY)
            )
            driving_action, guard = choose_driving(planner, environment, observation, message,
                visible=visible, safety_guard=args.safety_guard, min_clearance=args.min_clearance,
                min_ttc=args.min_ttc, lane_change_guard=args.lane_change_guard,
                max_target_speed=args.max_target_speed,
                comfort_ttc=getattr(args, 'comfort_ttc', None),
                batched=getattr(args, 'batched_planner', False),
                controller_state=controller_state)
            guard_overrides += int(bool(guard and guard['overridden']))
            if timestep == 0:
                initial_request, initial_driving = request_action, driving_action
            observation, reward, terminated, truncated, _ = environment.step(
                driving_action
            )
            ego = environment.vehicle
            _, lateral = environment.road.network.get_lane(ego.target_lane_index).local_coordinates(ego.position)
            lane = int(ego.lane_index[2])
            if (not ego.crashed and lane == ego.target_lane_index[2]
                    and abs(float(lateral)) <= .3 and abs(float(ego.heading)) <= .05
                    and lane != settled_lane):
                completed_lane_changes += 1
                settled_lane = lane
            driving_reward += float(reward)
            requests += requested_count
            total_return += float(reward) - args.communication_cost * requested_count
            if terminated or truncated:
                break

        rows.append(
            {
                "policy": policy,
                "episode": episode,
                "seed": environment_seed,
                "scenario": scenario, "hidden_mode": hidden_mode,
                "initial_request_action": initial_request, "initial_driving_action": initial_driving,
                "initial_scene_sha256": initial_signature,
                "steps": timestep + 1,
                "mean_speed": float(np.mean(speeds)),
                "completed_lane_changes": completed_lane_changes,
                "return": total_return,
                "driving_reward": driving_reward,
                "requests": requests,
                "guard_overrides": guard_overrides,
                "crashed": int(terminated),
                "truncated": int(truncated),
                "oracle_steps": oracle_steps,
                "oracle_agreements": oracle_agreements,
                "oracle_asks": oracle_asks,
            }
        )
        environment.close()
        if (episode + 1) % 20 == 0 or episode + 1 == len(cases):
            print(f'{policy}: {episode + 1}/{len(cases)} episodes', flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_experiment_argument(parser)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=100000)
    parser.add_argument("--planning-horizon", type=int, default=3)
    parser.add_argument("--communication-cost", type=float, default=0.05)
    parser.add_argument("--visibility", type=float, default=60.0)
    parser.add_argument("--side-window", type=float, default=10.0)
    parser.add_argument("--safety-guard", action="store_true", help="Apply geometry braking and first-action boundary masking")
    parser.add_argument("--min-clearance", type=float, default=8.)
    parser.add_argument("--min-ttc", type=float, default=3.)
    parser.add_argument('--comfort-ttc', type=float, default=None,
                        help='Proactively change lanes or slow before the emergency TTC threshold')
    parser.add_argument('--lane-change-guard', action='store_true')
    parser.add_argument('--max-target-speed', type=float, default=None)
    parser.add_argument('--scenario', choices=['ordinary', 'paired-intent'], default='ordinary',
                        help='paired-intent evaluates BOTH variants for every --episodes public scene')
    parser.add_argument('--batched-planner', action='store_true')
    parser.add_argument(
        "--policies",
        nargs="+",
        choices=["dqn", "none", "random", "all", "oracle"],
        default=["dqn", "none", "random", "all"],
        help="oracle is the privileged planner-based request oracle (upper reference)",
    )
    parser.add_argument('--oracle-agreement', action='store_true',
                        help='At every step also run the oracle and record agreement with it')
    parser.add_argument('--oracle-lookahead', type=int, default=5)
    parser.add_argument("--dqn-checkpoint", default=None)
    parser.add_argument("--mlpx-checkpoint", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    if (args.lane_change_guard or args.max_target_speed is not None) and not args.safety_guard:
        parser.error('lane-change-guard and max-target-speed require safety-guard')
    if args.comfort_ttc is not None and (not args.safety_guard or not args.lane_change_guard):
        parser.error('comfort-ttc requires safety-guard and lane-change-guard')
    if args.max_target_speed is not None and (not np.isfinite(args.max_target_speed) or args.max_target_speed <= 0):
        parser.error('max-target-speed must be finite and positive')
    if args.comfort_ttc is not None and (not np.isfinite(args.comfort_ttc) or args.comfort_ttc <= args.min_ttc):
        parser.error('comfort-ttc must be finite and greater than min-ttc')
    if min(args.episodes, args.planning_horizon) < 1:
        parser.error('episodes and planning-horizon must be positive')
    if not np.isfinite([args.min_clearance, args.min_ttc]).all() or min(args.min_clearance, args.min_ttc) <= 0:
        parser.error("guard thresholds must be finite and positive")

    dqn_checkpoint = args.dqn_checkpoint or str(
        experiment_checkpoint_dir(args.experiment, "RequestDQN") / "best.pt"
    )
    dqn = RequestDQN.load(dqn_checkpoint) if 'dqn' in args.policies else None
    mlpx_checkpoint = args.mlpx_checkpoint or (getattr(dqn, 'mlpx_checkpoint', None) if dqn else None) or str(
        experiment_checkpoint_dir(args.experiment, "MLPx") / "best.pt"
    )
    output_dir = Path(args.output_dir or experiment_results_dir(args.experiment, "evaluation") / (
        "request_dqn_guarded" if args.safety_guard else "request_dqn"))
    output_dir.mkdir(parents=True, exist_ok=True)
    if dqn and dqn.training_config.get('safety_guard') and not args.safety_guard:
        warnings.warn('DQN was trained with a safety guard. Pass its guard flags to reproduce the trained controller.')
    world_model = MLPxWorldModel.load(mlpx_checkpoint)

    rows = []
    for policy in args.policies:
        rows.extend(evaluate_policy(policy, dqn, world_model, args))
    signatures = {}
    for row in rows:
        key = (row['seed'], row['hidden_mode'])
        previous = signatures.setdefault(key, row['initial_scene_sha256'])
        if previous != row['initial_scene_sha256']:
            raise RuntimeError('Initial scene differs across request policies')

    csv_path = output_dir / "request_dqn_evaluation.csv"
    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["policy"]].append(row)
    summary = {}
    for policy, group in sorted(grouped.items()):
        summary[policy] = {
            "episodes": len(group),
            "mean_return": float(np.mean([row["return"] for row in group])),
            "mean_driving_reward": float(
                np.mean([row["driving_reward"] for row in group])
            ),
            "mean_requests": float(np.mean([row["requests"] for row in group])),
            "crashes": int(sum(row["crashed"] for row in group)),
            "crash_rate": float(np.mean([row["crashed"] for row in group])),
            "truncations": int(sum(row["truncated"] for row in group)),
            "guard_overrides": int(sum(row["guard_overrides"] for row in group)),
            "mean_steps": float(np.mean([row['steps'] for row in group])),
            "mean_episode_speed": float(np.mean([row['mean_speed'] for row in group])),
            "completed_lane_changes": sum(row['completed_lane_changes'] for row in group),
            "requests_per_step": sum(row['requests'] for row in group) / sum(row['steps'] for row in group),
        }
        oracle_steps = sum(row['oracle_steps'] for row in group)
        if oracle_steps:
            summary[policy]['oracle_agreement'] = sum(row['oracle_agreements'] for row in group) / oracle_steps
            summary[policy]['oracle_ask_rate_on_visited_steps'] = sum(row['oracle_asks'] for row in group) / oracle_steps

    (output_dir / "request_dqn_summary.json").write_text(
        json.dumps(
            {
                "experiment": args.experiment,
                "dqn_checkpoint": dqn_checkpoint,
                "request_action_encoding": dqn.request_action_encoding if dqn else None,
                "request_state_encoding": getattr(dqn, 'request_state_encoding', None),
                "dqn_sha256": hashlib.sha256(Path(dqn_checkpoint).read_bytes()).hexdigest() if dqn else None,
                "mlpx_checkpoint": mlpx_checkpoint,
                "mlpx_sha256": hashlib.sha256(Path(mlpx_checkpoint).read_bytes()).hexdigest(),
                "environment_config": CustomEnv.default_config(),
                "paired_initial_scenes": len(signatures),
                "planning_horizon": args.planning_horizon,
                "visibility": args.visibility,
                "side_window": args.side_window,
                "episodes": args.episodes,
                "scenario": args.scenario, "batched_planner": args.batched_planner,
                "scenario_spec": SCENARIO_SPEC if args.scenario == 'paired-intent' else None,
                "scenario_overrides": {'duration': 3, 'initial_vehicle_count': 0,
                                       'spawned_background_vehicles': 3, 'ego_lane': 'seeded from scenario_spec'}
                                      if args.scenario == 'paired-intent' else None,
                "scenes_per_policy": args.episodes,
                "episodes_per_policy": len(rows) // len(args.policies),
                "seed_start": args.seed,
                "communication_cost": args.communication_cost,
                "safety_guard": args.safety_guard,
                "lane_change_guard": args.lane_change_guard,
                "max_target_speed": args.max_target_speed,
                "guard_thresholds": {"min_clearance": args.min_clearance, "min_ttc": args.min_ttc} if args.safety_guard else None,
                "comfort_ttc": args.comfort_ttc,
                "oracle_lookahead": args.oracle_lookahead
                if 'oracle' in args.policies or args.oracle_agreement else None,
                "summary": summary,
            },
            indent=2,
        )
    )
    print(f"Saved Request DQN evaluation to {output_dir}")
    for policy, metrics in summary.items():
        print(
            f"{policy:7s}: return={metrics['mean_return']:.3f} "
            f"driving={metrics['mean_driving_reward']:.3f} "
            f"requests={metrics['mean_requests']:.2f} "
            f"crash_rate={metrics['crash_rate']:.2%}"
            + (f" oracle_agreement={metrics['oracle_agreement']:.3f}"
               if 'oracle_agreement' in metrics else "")
        )


if __name__ == "__main__":
    main()
