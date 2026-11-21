"""Plot ordinary and intent Request-DQN evaluation summaries."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.utils.experiment_utils import add_experiment_argument


def load_summary(path):
    return json.loads((path / "request_dqn_summary.json").read_text())["summary"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_experiment_argument(parser)
    parser.add_argument("--ordinary-dir", type=Path, required=True)
    parser.add_argument("--intent-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    summaries = [load_summary(args.ordinary_dir), load_summary(args.intent_dir)]
    policies = [policy for policy in ("none", "dqn", "random", "all")
                if policy in summaries[0] and policy in summaries[1]]
    colors = {"none": "#4f70c9", "dqn": "#279987",
              "random": "#e5a437", "all": "#9270c8"}
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for column, (name, summary) in enumerate(zip(("Ordinary", "Intent"), summaries)):
        returns = [summary[p]["mean_return"] for p in policies]
        crash_rates = [100 * summary[p]["crash_rate"] for p in policies]
        requests = [summary[p]["mean_requests"] for p in policies]
        axes[0, column].bar(policies, returns, color=[colors[p] for p in policies])
        axes[0, column].set_title(f"{name}: adjusted return")
        axes[0, column].set_ylabel("Return after message cost")
        axes[1, column].bar(policies, requests, color=[colors[p] for p in policies])
        axes[1, column].set_title(f"{name}: requests and crash rate")
        axes[1, column].set_ylabel("Requests per trial")
        twin = axes[1, column].twinx()
        twin.plot(policies, crash_rates, color="#d62728", marker="o", linewidth=2)
        twin.set_ylabel("Crash rate (%)", color="#d62728")
        twin.set_ylim(0, max(1.0, max(crash_rates) * 1.25))
        for axis, values in ((axes[0, column], returns), (axes[1, column], requests)):
            axis.grid(axis="y", alpha=.25)
            for index, value in enumerate(values):
                axis.text(index, value, f"{value:.2f}", ha="center", va="bottom")
    fig.suptitle(f"Experiment {args.experiment} · selective Request DQN")
    fig.tight_layout()
    output = args.output or Path("docs/policy_graphs") / f"experiment_{args.experiment}_dqn_comparison.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
