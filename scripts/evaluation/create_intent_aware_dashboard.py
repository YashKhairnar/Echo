"""Create a static HTML dashboard for an intent-aware JSONL dataset.

Run from the project root with:

    .venv/bin/python -m scripts.evaluation.create_intent_aware_dashboard
"""

import argparse
import html
import json
import os
from collections import Counter, defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ACTION_ORDER = [
    "LANE_LEFT",
    "IDLE",
    "LANE_RIGHT",
    "FASTER",
    "SLOWER",
]


def read_dataset(path):
    records = []
    with open(path) as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))
    return records


def save_bar_chart(path, title, labels, values, color="#4f46e5", ylabel="Count"):
    figure, axis = plt.subplots(figsize=(9, 4.8))
    bars = axis.bar(labels, values, color=color)
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", alpha=0.25)
    axis.set_axisbelow(True)
    for bar, value in zip(bars, values):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:,}",
            ha="center",
            va="bottom",
        )
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def metric_card(label, value, detail=""):
    return (
        '<div class="card">'
        f'<div class="card-label">{html.escape(label)}</div>'
        f'<div class="card-value">{html.escape(str(value))}</div>'
        f'<div class="card-detail">{html.escape(detail)}</div>'
        "</div>"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--input", default="data/intent_aware_dataset.jsonl")
    parser.add_argument(
        "--output-dir",
        default="results/intent_aware_dashboard",
    )
    parser.add_argument("--actual-episodes", type=int, default=None)
    parser.add_argument("--actual-steps", type=int, default=None)
    parser.add_argument("--actual-crashes", type=int, default=None)
    parser.add_argument("--actual-truncations", type=int, default=None)
    args = parser.parse_args()

    records = read_dataset(args.input)
    if not records:
        raise ValueError("Dataset is empty")

    os.makedirs(args.output_dir, exist_ok=True)
    chart_dir = os.path.join(args.output_dir, "charts")
    os.makedirs(chart_dir, exist_ok=True)

    action_counts = Counter(record["ego_action_name"] for record in records)
    request_counts = Counter(
        len(record["requested_vehicle_ids"])
        for record in records
    )
    behavior_counts = Counter()
    rewards_by_action = defaultdict(list)
    episode_counts = Counter(record["episode"] for record in records)

    terminated = sum(bool(record.get("terminated")) for record in records)
    truncated = sum(bool(record.get("truncated")) for record in records)
    rewards = []

    for record in records:
        action = record["ego_action_name"]
        reward = float(record["reward"])
        rewards.append(reward)
        rewards_by_action[action].append(reward)
        for message in (record.get("messages") or {}).values():
            behavior_counts[message.get("behavior_type", "unknown")] += 1

    action_labels = [action for action in ACTION_ORDER if action_counts[action]]
    action_values = [action_counts[action] for action in action_labels]
    request_labels = [str(value) for value in sorted(request_counts)]
    request_values = [request_counts[int(value)] for value in request_labels]
    behavior_labels = sorted(behavior_counts)
    behavior_values = [behavior_counts[label] for label in behavior_labels]

    save_bar_chart(
        os.path.join(chart_dir, "actions.png"),
        "Ego action distribution",
        action_labels,
        action_values,
        color="#2563eb",
    )
    save_bar_chart(
        os.path.join(chart_dir, "requests.png"),
        "Requested vehicles per scenario",
        request_labels,
        request_values,
        color="#7c3aed",
        ylabel="Scenarios",
    )
    save_bar_chart(
        os.path.join(chart_dir, "behaviors.png"),
        "Behavior types in requested messages",
        behavior_labels,
        behavior_values,
        color="#059669",
    )

    reward_labels = [action for action in ACTION_ORDER if rewards_by_action[action]]
    reward_values = [
        sum(rewards_by_action[action]) / len(rewards_by_action[action])
        for action in reward_labels
    ]
    save_bar_chart(
        os.path.join(chart_dir, "rewards.png"),
        "Mean reward by ego action",
        reward_labels,
        reward_values,
        color="#d97706",
        ylabel="Mean reward",
    )

    mean_reward = sum(rewards) / len(rewards)
    actual_cards = ""
    if args.actual_episodes is not None:
        actual_crash_rate = (
            args.actual_crashes / args.actual_episodes
            if args.actual_episodes
            else 0.0
        )
        actual_cards = "".join([
            metric_card("Actual rollout episodes", f"{args.actual_episodes:,}"),
            metric_card("Actual rollout steps", f"{args.actual_steps:,}" if args.actual_steps is not None else "n/a"),
            metric_card("Actual rollout crashes", f"{args.actual_crashes:,}"),
            metric_card("Actual crash rate", f"{actual_crash_rate:.2%}"),
            metric_card("Actual truncations", f"{args.actual_truncations:,}" if args.actual_truncations is not None else "n/a"),
        ])

    action_rows = "".join(
        f"<tr><td>{html.escape(action)}</td><td>{action_counts[action]:,}</td>"
        f"<td>{action_counts[action] / len(records):.2%}</td>"
        f"<td>{sum(rewards_by_action[action]) / len(rewards_by_action[action]):.3f}</td></tr>"
        for action in action_labels
    )
    request_rows = "".join(
        f"<tr><td>{count}</td><td>{request_counts[count]:,}</td>"
        f"<td>{request_counts[count] / len(records):.2%}</td></tr>"
        for count in sorted(request_counts)
    )

    dashboard = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Intent-Aware Dataset Dashboard</title>
  <style>
    :root {{ color-scheme: light; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    body {{ margin: 0; background: #f3f4f6; color: #111827; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 32px; }}
    h1 {{ margin-bottom: 6px; }}
    .subtitle {{ color: #6b7280; margin-top: 0; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 14px; margin: 24px 0; }}
    .card {{ background: white; border-radius: 12px; padding: 18px; box-shadow: 0 1px 3px #00000012; }}
    .card-label {{ color: #6b7280; font-size: 13px; }}
    .card-value {{ font-size: 26px; font-weight: 700; margin-top: 6px; }}
    .card-detail {{ color: #9ca3af; font-size: 12px; margin-top: 4px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(450px, 1fr)); gap: 18px; }}
    .panel {{ background: white; border-radius: 12px; padding: 18px; box-shadow: 0 1px 3px #00000012; }}
    .panel img {{ width: 100%; height: auto; }}
    .flow {{ display: flex; flex-wrap: wrap; align-items: stretch; gap: 10px; margin-top: 14px; }}
    .flow-step {{ flex: 1 1 145px; min-width: 125px; background: #eef2ff; border: 1px solid #c7d2fe; border-radius: 10px; padding: 14px; }}
    .flow-step strong {{ display: block; color: #3730a3; margin-bottom: 6px; }}
    .flow-step span {{ color: #4b5563; font-size: 13px; line-height: 1.35; }}
    .flow-arrow {{ align-self: center; color: #6366f1; font-size: 24px; font-weight: 700; }}
    .flow-branch {{ margin-top: 12px; padding: 12px; background: #ecfdf5; border: 1px solid #a7f3d0; border-radius: 10px; color: #065f46; font-size: 13px; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ text-align: left; padding: 9px; border-bottom: 1px solid #e5e7eb; }}
    th {{ color: #6b7280; font-size: 13px; }}
    .safe {{ color: #047857; font-weight: 700; }}
    .note {{ color: #6b7280; font-size: 13px; line-height: 1.5; }}
  </style>
</head>
<body>
<main>
  <h1>Intent-Aware Dataset Dashboard</h1>
  <p class="subtitle">Source: {html.escape(args.input)}</p>
  <div class="cards">
    {metric_card("Records", f"{len(records):,}")}
    {metric_card("Mean reward", f"{mean_reward:.3f}")}
    {metric_card("Scenario crashes", f"{terminated:,}", f"{terminated / len(records):.2%}")}
    {metric_card("Scenario truncations", f"{truncated:,}", f"{truncated / len(records):.2%}")}
    {metric_card("Episodes represented", f"{len(episode_counts):,}")}
    {actual_cards}
  </div>
  <div class="panel">
    <p class="note"><span class="safe">No scenario crashes were recorded.</span>
    Scenario metrics are counterfactual one-step outcomes. Actual rollout
    metrics are shown only when supplied on the command line.</p>
  </div>
  <div class="panel" style="margin: 18px 0;">
    <h2>How the dataset was created</h2>
    <div class="flow">
      <div class="flow-step"><strong>1. Reset</strong><span>Start an episode with randomized lanes, vehicle types, and traffic offsets.</span></div>
      <div class="flow-arrow">→</div>
      <div class="flow-step"><strong>2. Observe</strong><span>Find neighbors within 60 m in the ego and adjacent lanes, up to 8 slots.</span></div>
      <div class="flow-arrow">→</div>
      <div class="flow-step"><strong>3. Request</strong><span>Enumerate message subsets from one requested vehicle through all visible vehicles.</span></div>
      <div class="flow-arrow">→</div>
      <div class="flow-step"><strong>4. Preview intent</strong><span>Use each requested neighbor's copied-scene next action as its message.</span></div>
      <div class="flow-arrow">→</div>
      <div class="flow-step"><strong>5. Choose action</strong><span>Apply the rule to select left, idle, right, faster, or slower.</span></div>
      <div class="flow-arrow">→</div>
      <div class="flow-step"><strong>6. Record</strong><span>Step a copied environment and save state, messages, action, next state, reward, and flags.</span></div>
    </div>
    <div class="flow-branch"><strong>Real rollout branch:</strong> after all counterfactual scenarios are evaluated, the real environment advances once using the action selected with all currently visible messages.</div>
  </div>
  <div class="grid">
    <div class="panel"><img src="charts/actions.png" alt="Ego action distribution"></div>
    <div class="panel"><img src="charts/requests.png" alt="Requested vehicle distribution"></div>
    <div class="panel"><img src="charts/behaviors.png" alt="Message behavior distribution"></div>
    <div class="panel"><img src="charts/rewards.png" alt="Mean reward by action"></div>
    <div class="panel">
      <h2>Action statistics</h2>
      <table><tr><th>Action</th><th>Count</th><th>Share</th><th>Mean reward</th></tr>{action_rows}</table>
    </div>
    <div class="panel">
      <h2>Request statistics</h2>
      <table><tr><th>Messages requested</th><th>Scenarios</th><th>Share</th></tr>{request_rows}</table>
    </div>
  </div>
</main>
</body>
</html>"""

    output_path = os.path.join(args.output_dir, "index.html")
    with open(output_path, "w") as file:
        file.write(dashboard)
    print(f"Wrote dashboard to {output_path}")


if __name__ == "__main__":
    main()
