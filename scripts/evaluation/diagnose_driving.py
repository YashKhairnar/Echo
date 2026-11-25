"""Compare driving controllers on paired seeds and record crash lead-ups.

Run with --experiment 3 --episodes 20. Creates a fresh timestamped run;
existing experiment evaluations and checkpoints are left intact.
"""

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, deque
from pathlib import Path

import numpy as np

from scripts.data_collection.enumerate_message_requests import jsonable
from scripts.data_collection.show_visible_neighbors import SLOT_ORDER, visible_neighbors
from scripts.evaluation.evaluate_request_dqn import choose_request
from scripts.policy.intent_aware_policy import ACTION_NAMES, choose_action
from scripts.utils.experiment_utils import (
    add_experiment_argument, create_run_dir, experiment_checkpoint_dir,
    experiment_results_dir, save_run_metadata,
)
from src.cooperative_driving.environments.customEnv import CustomEnv
from src.cooperative_driving.models.MLPx.mlpx import MLPxWorldModel
from src.cooperative_driving.planner.MCP import MCPPlanner
from src.cooperative_driving.planner.safety import guard_action
from src.cooperative_driving.policies import load_request_policy


def scene_snapshot(environment, visible):
    ego = environment.vehicle
    slots = {candidate["vehicle"].vehicle_id: "/".join(slot)
             for slot, candidate in visible.items()}
    neighbors = []
    for vehicle in environment.road.vehicles:
        if vehicle is ego:
            continue
        relative = vehicle.position - ego.position
        speed_difference = float(ego.velocity[0] - vehicle.velocity[0])
        gap = abs(float(relative[0])) - (ego.LENGTH + vehicle.LENGTH) / 2
        closing = speed_difference if relative[0] >= 0 else -speed_difference
        neighbors.append({
            "vehicle_id": vehicle.vehicle_id,
            "lane": int(vehicle.lane_index[2]),
            "dx": float(relative[0]), "dy": float(relative[1]),
            "distance": float(np.linalg.norm(relative)),
            "longitudinal_clearance": gap,
            "speed": float(vehicle.speed), "closing_speed": closing,
            "longitudinal_ttc": max(0., gap) / closing if closing > 0 else None,
            "visible_slot": slots.get(vehicle.vehicle_id),
            "crashed": bool(vehicle.crashed),
        })
    return {
        "ego": {"vehicle_id": ego.vehicle_id, "lane": int(ego.lane_index[2]),
                "target_lane": int(ego.target_lane_index[2]),
                "position": ego.position.tolist(), "speed": float(ego.speed),
                "target_speed": float(ego.target_speed), "heading": float(ego.heading)},
        "neighbors": neighbors,
    }


def classify_crash(before, after, action):
    """Heuristic collision geometry, not an attribution of fault."""
    newly_crashed = [other for other in after["neighbors"] if other["crashed"]
                     and not next(item["crashed"] for item in before["neighbors"]
                                  if item["vehicle_id"] == other["vehicle_id"])
                     and other["distance"] <= 8.]
    if not newly_crashed:
        return {"category": "unknown", "partner_id": None}
    other = min(newly_crashed, key=lambda item: item["distance"])
    prior = next(item for item in before["neighbors"] if item["vehicle_id"] == other["vehicle_id"])
    lateral_motion = abs(after["ego"]["position"][1] - before["ego"]["position"][1]) > .5
    changing = before["ego"]["lane"] != before["ego"]["target_lane"] or lateral_motion
    if prior["lane"] != before["ego"]["lane"] or changing:
        category = "lane_change_or_lateral_conflict"
    else:
        category = "front_contact" if prior["dx"] >= 0 else "rear_contact"
    return {"category": category, "partner_id": other["vehicle_id"],
            "lane_change_command": ACTION_NAMES[action] in ("LANE_LEFT", "LANE_RIGHT")}


def select_driving(policy, environment, visible, dqn, planner, observation, rng,
                   min_clearance=8., min_ttc=3., max_target_speed=None,
                   lane_change_guard=False):
    if policy.startswith("rule"):
        # An empty mapping means no private messages. None would query previews!
        messages = {} if policy == "rule_none" else {
            candidate["vehicle"].vehicle_id: environment.request_neighbor_action(candidate["vehicle"].vehicle_id)
            for candidate in visible.values()
        }
        action, explanation = choose_action(environment, messages=messages)
        bundle = {"messages_by_slot": [messages.get(visible[slot]["vehicle"].vehicle_id)
                                       if slot in visible else None for slot in SLOT_ORDER]}
        return action, bundle if messages else None, explanation, None
    _, bundle = choose_request("none" if policy.removesuffix("_guarded") == "wm_none" else "dqn",
                               dqn, environment, visible, rng)
    values = planner.action_values(observation, bundle)
    selection_values = list(values)
    speed_cap_blocked = (max_target_speed is not None
                         and environment.vehicle.target_speed >= max_target_speed)
    if speed_cap_blocked:
        selection_values[3] = -1e30
    if policy.endswith("_guarded"):
        action, guard = guard_action(selection_values, environment.vehicle, visible,
                                    environment.config["lanes_count"], min_clearance, min_ttc,
                                    check_lane_changes=lane_change_guard)
        guard['speed_cap_blocked'] = speed_cap_blocked
        return action, bundle, guard, values
    return int(np.argmax(selection_values)), bundle, {'speed_cap_blocked': speed_cap_blocked}, values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_experiment_argument(parser)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=100000)
    parser.add_argument("--planning-horizon", type=int, default=3)
    parser.add_argument("--tail-steps", type=int, default=5)
    parser.add_argument("--min-clearance", type=float, default=8.)
    parser.add_argument("--min-ttc", type=float, default=3.)
    parser.add_argument("--max-target-speed", type=float, default=None,
                        help="Diagnostic only: mask FASTER once the ego target reaches this speed")
    parser.add_argument('--lane-change-guard', action='store_true',
                        help='Also check destination gaps and prevent stacked lane commands in guarded policies')
    parser.add_argument("--policies", nargs="+", choices=["rule_none", "rule_all", "wm_none", "wm_dqn", "wm_none_guarded", "wm_dqn_guarded"],
                        default=["rule_none", "rule_all", "wm_none", "wm_dqn"])
    parser.add_argument("--mlpx-checkpoint", default=None)
    parser.add_argument("--dqn-checkpoint", default=None)
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args()
    if args.max_target_speed is not None and (not np.isfinite(args.max_target_speed) or args.max_target_speed <= 0):
        parser.error('max-target-speed must be finite and positive')
    args.policies = list(dict.fromkeys(args.policies))
    if args.lane_change_guard and not any(policy.endswith('_guarded') for policy in args.policies):
        parser.error('lane-change-guard requires at least one guarded policy')
    if min(args.episodes, args.tail_steps, args.planning_horizon) < 1:
        parser.error("episodes, tail-steps, and planning-horizon must be positive")
    if not np.isfinite([args.min_clearance, args.min_ttc]).all() or min(args.min_clearance, args.min_ttc) <= 0:
        parser.error("guard thresholds must be finite and positive")
    wm_path = Path(args.mlpx_checkpoint or experiment_checkpoint_dir(args.experiment, "MLPx") / "best.pt")
    dqn_path = Path(args.dqn_checkpoint or experiment_checkpoint_dir(args.experiment, "RequestDQN") / "best.pt")
    model = MLPxWorldModel.load(wm_path)
    dqn = load_request_policy(dqn_path) if any(policy.startswith("wm_dqn") for policy in args.policies) else None
    output = create_run_dir(Path(args.output_root or experiment_results_dir(args.experiment, "diagnostics")), "driving")
    save_run_metadata(output, {**vars(args), "mlpx_checkpoint": str(wm_path),
                              "environment_config": CustomEnv.default_config(),
                              "mlpx_sha256": hashlib.sha256(wm_path.read_bytes()).hexdigest(),
                              "dqn_checkpoint": str(dqn_path) if dqn else None,
                              "dqn_sha256": hashlib.sha256(dqn_path.read_bytes()).hexdigest() if dqn else None,
                              "crash_categories": "heuristic geometry; not fault attribution"})
    rows, initial_scenes = [], {}
    print(f"Diagnostic run: {output}", flush=True)
    with (output / "steps.jsonl").open("w") as steps_file, (output / "crashes.jsonl").open("w") as crash_file:
        for policy in args.policies:
            for episode in range(args.episodes):
                seed = args.seed + episode
                environment = CustomEnv()
                try:
                    observation, _ = environment.reset(seed=seed)
                    visible = visible_neighbors(environment, 60., 10.)
                    initial = json.dumps(jsonable(environment.get_all_vehicle_states()), sort_keys=True)
                    signature = hashlib.sha256(initial.encode()).hexdigest()
                    if seed in initial_scenes and initial_scenes[seed] != signature:
                        raise RuntimeError(f"Initial scene differs between policies for seed {seed}")
                    initial_scenes[seed] = signature
                    planner = MCPPlanner(model, planning_horizon=args.planning_horizon)
                    rng = random.Random(seed)
                    history = deque(maxlen=args.tail_steps)
                    total = requests = invalid = lane_commands = overrides = brake_triggers = completed_changes = lane_crossings = 0
                    settled_lane = int(environment.vehicle.lane_index[2])
                    speeds = []
                    category = None
                    for timestep in range(environment.config["duration"]):
                        observation = model.observe(environment)
                        visible = visible_neighbors(environment, 60., 10.)
                        before = scene_snapshot(environment, visible)
                        action, message, explanation, values = select_driving(
                            policy, environment, visible, dqn, planner, observation, rng,
                            args.min_clearance, args.min_ttc, args.max_target_speed, args.lane_change_guard)
                        guard = explanation if policy.endswith("_guarded") else None
                        requested = [item["vehicle_id"] for item in (message or {}).get("messages_by_slot", []) if item]
                        predicted_state, predicted_reward = model.predict_step(observation, action, message)
                        invalid_lane = ((ACTION_NAMES[action] == "LANE_LEFT" and before["ego"]["lane"] == 0)
                                        or (ACTION_NAMES[action] == "LANE_RIGHT" and before["ego"]["lane"] == environment.config["lanes_count"] - 1))
                        next_observation, reward, terminated, truncated, _ = environment.step(action)
                        next_observation = model.observe(environment)
                        after = scene_snapshot(environment, visible_neighbors(environment, 60., 10.))
                        # Lane index crossing, not just an issued command.
                        lane_crossings += int(before['ego']['lane'] != after['ego']['lane'])
                        target_geometry = environment.road.network.get_lane(environment.vehicle.target_lane_index)
                        _, lateral_offset = target_geometry.local_coordinates(environment.vehicle.position)
                        settled = (after['ego']['lane'] == after['ego']['target_lane']
                                   and abs(float(lateral_offset)) <= .3
                                   and abs(after['ego']['heading']) <= .05)
                        if settled and not environment.vehicle.crashed and after['ego']['lane'] != settled_lane:
                            completed_changes += 1
                            settled_lane = after['ego']['lane']
                        trace = {"policy": policy, "seed": seed, "timestep": timestep,
                                 "before": before, "after": after,
                                 "state": observation, "next_state": next_observation,
                                 "messages": message, "requested_vehicle_ids": requested,
                                 "action": action, "action_name": ACTION_NAMES[action],
                                 "rule_explanation": explanation if policy.startswith("rule") else None,
                                 "safety_guard": guard, "planner_action_values": values,
                                 "speed_cap_blocked": bool(explanation and isinstance(explanation, dict)
                                                           and explanation.get('speed_cap_blocked')),
                                 "invalid_boundary_lane_command": invalid_lane,
                                 "predicted_next_state": predicted_state,
                                 "predicted_reward": predicted_reward, "real_reward": float(reward),
                                 "state_mae": float(np.mean(np.abs(predicted_state.numpy().reshape(-1) - np.asarray(next_observation).reshape(-1)))),
                                 "reward_abs_error": abs(predicted_reward - float(reward)),
                                 "terminated": bool(terminated), "truncated": bool(truncated)}
                        # jsonable handles NumPy; predicted_state is a CPU tensor.
                        trace["predicted_next_state"] = predicted_state.tolist()
                        history.append(trace)
                        steps_file.write(json.dumps(jsonable(trace), allow_nan=False) + "\n")
                        total += float(reward)
                        requests += len(requested)
                        invalid += int(invalid_lane)
                        lane_commands += int(ACTION_NAMES[action] in ("LANE_LEFT", "LANE_RIGHT"))
                        overrides += int(bool(guard and guard["overridden"]))
                        brake_triggers += int(bool(guard and guard["braking_required"]))
                        speeds.append(before["ego"]["speed"])
                        observation = next_observation
                        if terminated or truncated:
                            if terminated:
                                category = classify_crash(before, after, action)
                                crash_file.write(json.dumps(jsonable({"policy": policy, "seed": seed,
                                                 **category, "last_decisions": list(history)}), allow_nan=False) + "\n")
                            break
                    rows.append({"policy": policy, "seed": seed, "initial_scene_sha256": signature,
                                 "steps": timestep + 1, "driving_reward": total,
                                 "return_after_cost_005": total - .05 * requests,
                                 "requests": requests, "crashed": bool(terminated), "truncated": bool(truncated),
                                 "boundary_lane_commands": invalid, "lane_change_commands": lane_commands,
                                 "lane_index_changes": lane_crossings,
                                 "completed_lane_changes": completed_changes,
                                 "guard_overrides": overrides, "guard_braking_decisions": brake_triggers,
                                 "mean_speed": float(np.mean(speeds)),
                                 "crash_category": category["category"] if category else None})
                    print(f"{policy}: episode {episode + 1}/{args.episodes} seed={seed} "
                          f"steps={timestep + 1} crashed={terminated} reward={total:.3f}", flush=True)
                finally:
                    environment.close()
    with (output / "episodes.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for policy in args.policies:
        group = [row for row in rows if row["policy"] == policy]
        summary[policy] = {"episodes": len(group), "crash_rate": float(np.mean([row["crashed"] for row in group])),
                           "mean_driving_reward": float(np.mean([row["driving_reward"] for row in group])),
                           "mean_steps": float(np.mean([row["steps"] for row in group])),
                           "mean_requests": float(np.mean([row["requests"] for row in group])),
                           "boundary_lane_commands": sum(row["boundary_lane_commands"] for row in group),
                           "lane_change_commands": sum(row["lane_change_commands"] for row in group),
                           "lane_index_changes": sum(row["lane_index_changes"] for row in group),
                           "completed_lane_changes": sum(row["completed_lane_changes"] for row in group),
                           "guard_overrides": sum(row["guard_overrides"] for row in group),
                           "guard_braking_decisions": sum(row["guard_braking_decisions"] for row in group),
                           "mean_episode_speed": float(np.mean([row["mean_speed"] for row in group])),
                           "crash_categories": dict(Counter(row["crash_category"] for row in group if row["crashed"]))}
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
