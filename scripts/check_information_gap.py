"""Check that the merge scenario's private information is hidden and useful.

Run this before collecting world-model data or training any learned model.

Part A — Is it hidden?
    Train a classifier to predict each target-lane car's driver type from
    what the ego observes (relative position, velocity, and the change in
    velocity since the previous step). Before the ego starts merging, accuracy
    should be near chance (50%). During a merge the reaction becomes visible,
    but by then the ego has already committed.

Part B — Does it change the outcome?
    Run hand-written driving rules on the same seeds:

    - merge_now:   merge immediately, never ask (a gamble);
    - no_info:     merge only into a gap that is safe whatever the rear
                   driver's type, otherwise slow down and wait;
    - ask_rear:    same as no_info, but when the rear gap is only safe for a
                   yielding driver, ask the rear car and merge if it yields.

    If ask_rear clearly beats both no-communication rules, the message carries
    information that changes the best driving decision.

Example:
    .venv/bin/python -m scripts.check_information_gap --episodes 300
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from scripts.exploration import FASTER, IDLE, LANE_LEFT, SLOWER
from src.cooperative_driving.communication.getMessage import getMessage
from src.cooperative_driving.environments.customEnv import (
    TARGET_LANE,
    make_environment,
)
from src.cooperative_driving.vehicles.customVehicles import (
    YIELDING,
    DriverTypeIDMVehicle,
)

# A rear car further back than this does not react to the merge at all.
REACTION_DISTANCE = DriverTypeIDMVehicle.REACTION_DISTANCE
MERGE_DURATION = 3.0      # [s] roughly how long a lane change takes
YIELD_DECELERATION = 4.0  # [m/s2] braking we may ask of a yielding driver
# Minimum gaps (centre-to-centre distance in metres).
MIN_REAR_GAP = 12.0
MIN_FRONT_GAP = 12.0
MIN_WAITING_SPEED = 15.0
FEATURE_COUNT = 6  # presence, x, y, vx, vy, vehicle_id


# ---------------------------------------------------------------------------
# Part A
# ---------------------------------------------------------------------------

def observation_rows(observation):
    rows = np.asarray(observation, dtype=np.float32).reshape(-1, FEATURE_COUNT)
    return {
        int(round(row[5])): row[1:5]
        for row in rows[1:]
        if row[0] > 0
    }


def driver_types(environment):
    return {
        vehicle.vehicle_id: vehicle.driver_type
        for vehicle in environment.road.vehicles
        if isinstance(vehicle, DriverTypeIDMVehicle)
    }


def hidden_information_samples(environment, episodes, merge):
    """Collect (observed features, is_yielding) pairs.

    With ``merge=False`` the ego keeps its lane, so cars never react. With
    ``merge=True`` the ego merges at step 0 and only the rear car's reaction
    one step later is recorded.
    """
    features, labels = [], []
    for seed in range(episodes):
        observation, _ = environment.reset(seed=seed)
        types = driver_types(environment)
        _, rear = environment.target_lane_neighbours()
        previous = observation_rows(observation)
        for timestep in range(3):
            action = LANE_LEFT if merge and timestep == 0 else IDLE
            observation, _, terminated, truncated, _ = environment.step(action)
            current = observation_rows(observation)
            if merge:
                watched = [rear.vehicle_id] if rear is not None and timestep == 0 else []
            else:
                watched = list(current)
            for vehicle_id in watched:
                if vehicle_id in types and vehicle_id in previous and vehicle_id in current:
                    velocity_change = current[vehicle_id][2:4] - previous[vehicle_id][2:4]
                    features.append(np.concatenate([current[vehicle_id], velocity_change]))
                    labels.append(int(types[vehicle_id] == YIELDING))
            previous = current
            if terminated or truncated or merge:
                break
    return np.asarray(features), np.asarray(labels)


def classifier_accuracy(features, labels):
    if len(set(labels)) < 2 or len(labels) < 20:
        return float("nan")
    model = LogisticRegression(max_iter=1000)
    return float(cross_val_score(model, features, labels, cv=5).mean())


# ---------------------------------------------------------------------------
# Part B
# ---------------------------------------------------------------------------

def gaps(environment):
    front, rear = environment.target_lane_neighbours()
    ego_x = environment.vehicle.position[0]
    front_gap = front.position[0] - ego_x if front is not None else float("inf")
    rear_gap = ego_x - rear.position[0] if rear is not None else float("inf")
    return front, rear, front_gap, rear_gap


def closing_speed(environment, rear):
    return max(0.0, rear.speed - environment.vehicle.speed) if rear else 0.0


def safe_for_any_driver(environment, rear, rear_gap):
    """The rear car stays out of reaction range for the whole merge."""
    if rear is None:
        return True
    return rear_gap - closing_speed(environment, rear) * MERGE_DURATION > REACTION_DISTANCE


def safe_if_yielding(environment, rear, rear_gap):
    """A yielding rear car can brake comfortably to open the gap."""
    closing = closing_speed(environment, rear)
    return rear_gap >= MIN_REAR_GAP + closing ** 2 / (2 * YIELD_DECELERATION)


def wait_action(environment):
    """Fall back behind the platoon: slow down, but keep moving."""
    if environment.vehicle.target_speed > MIN_WAITING_SPEED:
        return SLOWER
    return IDLE


class MergeNow:
    name = "merge_now"

    def act(self, environment, timestep):
        return LANE_LEFT if timestep == 0 else IDLE


class NoInfo:
    """Merge only when the gap is safe for either driver type."""

    name = "no_info"

    def act(self, environment, timestep):
        ego = environment.vehicle
        if ego.target_lane_index[2] == TARGET_LANE:
            return FASTER if ego.target_speed < 25 else IDLE
        _, rear, front_gap, rear_gap = gaps(environment)
        if front_gap >= MIN_FRONT_GAP and safe_for_any_driver(environment, rear, rear_gap):
            return LANE_LEFT
        return wait_action(environment)


class AskRear(NoInfo):
    """Ask the rear car only when its answer would change the decision."""

    name = "ask_rear"

    def __init__(self):
        self.requests = 0

    def act(self, environment, timestep):
        ego = environment.vehicle
        if ego.target_lane_index[2] == TARGET_LANE:
            return FASTER if ego.target_speed < 25 else IDLE
        _, rear, front_gap, rear_gap = gaps(environment)
        if front_gap >= MIN_FRONT_GAP:
            if safe_for_any_driver(environment, rear, rear_gap):
                return LANE_LEFT
            if safe_if_yielding(environment, rear, rear_gap):
                self.requests += 1
                if getMessage(rear)["driver_type"] == YIELDING:
                    return LANE_LEFT
        return wait_action(environment)


def run_policy(environment, policy, episodes):
    results = []
    for seed in range(episodes):
        environment.reset(seed=seed)
        if hasattr(policy, "requests"):
            policy.requests = 0
        total_reward, info = 0.0, {}
        for timestep in range(environment.config["duration"]):
            action = policy.act(environment, timestep)
            _, reward, terminated, truncated, info = environment.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        results.append({
            "policy": policy.name,
            "seed": seed,
            "return": total_reward,
            "crashed": bool(info.get("crashed", False)),
            "merged": bool(info.get("merged", False)) and not info.get("crashed", False),
            "merge_time": info.get("merge_time"),
            "requests": getattr(policy, "requests", 0),
        })
    return results


def summarise(results):
    merge_times = [r["merge_time"] for r in results if r["merged"] and r["merge_time"] is not None]
    returns = np.asarray([r["return"] for r in results])
    return {
        "episodes": len(results),
        "mean_return": float(returns.mean()),
        "return_std_error": float(returns.std(ddof=1) / np.sqrt(len(returns))),
        "crash_rate": float(np.mean([r["crashed"] for r in results])),
        "merge_success_rate": float(np.mean([r["merged"] for r in results])),
        "mean_merge_time_s": float(np.mean(merge_times)) if merge_times else float("nan"),
        "requests_per_episode": float(np.mean([r["requests"] for r in results])),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--output-dir", default="results/information_gap")
    args = parser.parse_args()

    environment = make_environment()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Part A: can the ego infer driver type from its own observation?")
    hidden = {}
    for label, merge in (("before_merge", False), ("one_step_after_merge_starts", True)):
        features, labels = hidden_information_samples(environment, args.episodes, merge)
        accuracy = classifier_accuracy(features, labels)
        chance = max(labels.mean(), 1 - labels.mean()) if len(labels) else float("nan")
        hidden[label] = {"samples": int(len(labels)), "accuracy": accuracy, "chance": float(chance)}
        print(f"  {label:<30} n={len(labels):5d}  accuracy={accuracy:.3f}  chance={chance:.3f}")

    print("\nPart B: same seeds, rule-based drivers with and without the message")
    all_results, summary = [], {}
    for policy in (MergeNow(), NoInfo(), AskRear()):
        results = run_policy(environment, policy, args.episodes)
        all_results.extend(results)
        summary[policy.name] = summarise(results)

    header = f"  {'policy':<10} {'return':>14} {'crash':>7} {'merged':>7} {'merge t':>8} {'requests':>9}"
    print(header)
    for name, stats in summary.items():
        print(
            f"  {name:<10} {stats['mean_return']:7.2f} ± {stats['return_std_error']:4.2f}"
            f" {stats['crash_rate']:7.1%} {stats['merge_success_rate']:7.1%}"
            f" {stats['mean_merge_time_s']:7.1f}s {stats['requests_per_episode']:9.2f}"
        )

    with (output_dir / "episodes.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(all_results[0]))
        writer.writeheader()
        writer.writerows(all_results)
    with (output_dir / "summary.json").open("w") as file:
        json.dump({"hidden_information": hidden, "policies": summary}, file, indent=2)
    print(f"\nSaved results to {output_dir}/")
    environment.close()


if __name__ == "__main__":
    main()
