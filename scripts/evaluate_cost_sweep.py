"""Evaluate the cost sweep: how the request DQN changes as questions get more expensive.

Every DQN checkpoint found under the sweep folders is driven on the 200 test
roads at the cost it was trained with. The hand-written baselines (ask nobody,
rear-car rule, ask everyone) are driven once; their total at each cost is
driving reward minus cost x questions.

    .venv/bin/python -m scripts.evaluate_cost_sweep

Writes results/cost_sweep/summary.json and results/cost_sweep/cost_sweep.png.
"""

import argparse
import json
import os
from multiprocessing import Pool
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

CHECKPOINT_PATTERNS = ["checkpoints/DQN_sweep/cost*_seed*/best.pt", "checkpoints/DQN_seeds/val_seed*/best.pt"]
OUTPUT_DIR = Path("results/cost_sweep")


def evaluate(job):
    """Run one policy on the test roads in its own process."""
    name, checkpoint, cost, episodes = job
    torch.set_num_threads(1)
    from scripts.evaluate_merge_policies import AskAll, AskDQN, AskRear, RequestRule, run, summarise
    from src.cooperative_driving.environments.customEnv import make_environment
    from src.cooperative_driving.planner.MCP import build_planner

    environment = make_environment()
    planner = build_planner("checkpoints/MLPx/best.pt", environment)
    rule = {"none": RequestRule, "rear": AskRear, "all": AskAll}.get(name)
    policy = rule() if rule else AskDQN(checkpoint)
    return name, checkpoint, cost, summarise(run(environment, planner, policy, episodes, cost))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = parser.parse_args()

    checkpoints = sorted(str(p) for pattern in CHECKPOINT_PATTERNS for p in Path().glob(pattern))
    jobs = [(name, None, 0.0, args.episodes) for name in ("none", "rear", "all")]
    for path in checkpoints:
        cost = torch.load(path, map_location="cpu", weights_only=False)["communication_cost"]
        jobs.append(("dqn", path, cost, args.episodes))
    print(f"Evaluating {len(jobs)} policies on {args.episodes} test roads with {args.workers} workers")

    baselines, dqns = {}, []
    with Pool(args.workers) as pool:
        for name, checkpoint, cost, summary in pool.imap_unordered(evaluate, jobs):
            print(f"  {name:<5} cost {cost:<5g} {checkpoint or ''}: total {summary['total_reward']:.2f}, "
                  f"{summary['requests_per_episode']:.1f} questions/episode", flush=True)
            if name == "dqn":
                dqns.append({"checkpoint": checkpoint, "cost": cost, **summary})
            else:
                baselines[name] = summary

    costs = sorted({d["cost"] for d in dqns})
    table = []
    for cost in costs:
        runs = [d for d in dqns if d["cost"] == cost]
        row = {
            "cost": cost,
            "runs": len(runs),
            "dqn_total_mean": float(np.mean([d["total_reward"] for d in runs])),
            "dqn_total_min": float(np.min([d["total_reward"] for d in runs])),
            "dqn_total_max": float(np.max([d["total_reward"] for d in runs])),
            "dqn_questions_mean": float(np.mean([d["requests_per_episode"] for d in runs])),
            "dqn_crash_mean": float(np.mean([d["crash_rate"] for d in runs])),
            "dqn_rear_share_mean": float(np.nanmean([d["share_of_requests_to_rear_car"] for d in runs])),
        }
        for name, summary in baselines.items():
            row[f"{name}_total"] = summary["driving_reward"] - cost * summary["requests_per_episode"]
            row[f"{name}_questions"] = summary["requests_per_episode"]
        table.append(row)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(
        {"episodes": args.episodes, "table": table, "dqn_runs": dqns, "baselines": baselines}, indent=2))

    print(f"\n  {'cost':>5} {'DQN total (min-max)':>22} {'DQN questions':>14} {'DQN crashes':>12}"
          f" {'nobody':>7} {'rear rule':>10} {'everyone':>9}")
    for row in table:
        print(f"  {row['cost']:5g} {row['dqn_total_mean']:8.2f} ({row['dqn_total_min']:.2f}-{row['dqn_total_max']:.2f})"
              f" {row['dqn_questions_mean']:14.1f} {row['dqn_crash_mean']:12.0%}"
              f" {row['none_total']:7.2f} {row['rear_total']:10.2f} {row['all_total']:9.2f}")

    x = [row["cost"] for row in table]
    figure, (left, right) = plt.subplots(1, 2, figsize=(12, 4.5))
    for d in dqns:
        left.scatter(d["cost"], d["total_reward"], color="#2a78d6", alpha=0.35, s=18)
        right.scatter(d["cost"], d["requests_per_episode"], color="#2a78d6", alpha=0.35, s=18)
    left.plot(x, [r["dqn_total_mean"] for r in table], "o-", color="#2a78d6", label="request DQN (mean of seeds)")
    left.plot(x, [r["rear_total"] for r in table], "s--", color="#1a7f37", label="rear-car rule (hand-written)")
    left.plot(x, [r["none_total"] for r in table], "^--", color="#6e7781", label="ask nobody")
    left.set(xlabel="Cost per question", ylabel="Total score (driving − cost)", title="Score")
    right.plot(x, [r["dqn_questions_mean"] for r in table], "o-", color="#2a78d6", label="request DQN (mean of seeds)")
    right.plot(x, [r["rear_questions"] for r in table], "s--", color="#1a7f37", label="rear-car rule")
    right.set(xlabel="Cost per question", ylabel="Questions per episode", title="How much it asks")
    for axis in (left, right):
        axis.grid(True, alpha=0.3)
        axis.legend()
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "cost_sweep.png", dpi=150)
    print(f"\nSaved {OUTPUT_DIR / 'summary.json'} and {OUTPUT_DIR / 'cost_sweep.png'}")


if __name__ == "__main__":
    main()
