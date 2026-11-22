"""Export training CSV plots into the documentation for an experiment."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from scripts.utils.experiment_utils import add_experiment_argument


def latest_log(root, model, filename):
    paths = sorted((root / model / "experiments").glob(f"*/{filename}"))
    if not paths:
        raise FileNotFoundError(f"No {model} training log found under {root}")
    return paths[-1]


def read_rows(path):
    with path.open() as file:
        return list(csv.DictReader(file))


def load_dqn_history(log_path, explicit_logs=None):
    """Follow unambiguous resume parents; reject unrelated runs and duplicates."""
    def encoding(path):
        config_path = path.with_name("config.json")
        if config_path.exists():
            config = json.loads(config_path.read_text())
            return (config.get("request_action_encoding", "compressed_v0"),
                    config.get('request_state_encoding', 'kinematics_v1'))
        checkpoint_path = path.with_name("best.pt")
        if not checkpoint_path.exists():
            raise ValueError(f"Cannot determine request-action encoding for {path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        return (checkpoint.get('request_action_encoding', 'compressed_v0'),
                checkpoint.get('request_state_encoding', 'kinematics_v1'))

    def follow(path, visited):
        if path in visited:
            raise ValueError("Cycle in DQN resume history")
        visited = visited | {path}
        rows = read_rows(path)
        config_path = path.with_name("config.json")
        config = json.loads(config_path.read_text()) if config_path.exists() else {}
        start = int(config.get("start_episode", 0))
        if not start:
            return rows, [path], []
        source = Path(config["resume_from"])
        direct = source.with_name("request_dqn_training.csv")
        candidates = []
        for parent in path.parent.parent.glob("*/request_dqn_training.csv"):
            if parent.parent.name >= path.parent.name:
                continue
            parent_rows = read_rows(parent)
            if any(int(row["episode"]) == start for row in parent_rows) and encoding(parent) == encoding(path):
                candidates.append(parent)
        if direct in candidates:
            candidates = [direct]
        if len(candidates) != 1:
            raise ValueError(f"Ambiguous or missing parent for {path}; pass --dqn-logs explicitly")
        earlier, paths, boundaries = follow(candidates[0], visited)
        return ([row for row in earlier if int(row["episode"]) <= start] + rows,
                paths + [path], [boundary for boundary in boundaries if boundary < start] + [start])

    if explicit_logs:
        paths = [Path(path) for path in explicit_logs]
        if len({encoding(path) for path in paths}) != 1:
            raise ValueError("Cannot combine DQN runs with different action encodings or state encodings")
        segments = [read_rows(path) for path in paths]
        rows = [row for segment in segments for row in segment]
        boundaries = [int(segment[0]["episode"]) - 1 for segment in segments[1:] if segment]
    else:
        rows, paths, boundaries = follow(log_path, set())
    episodes = [int(row["episode"]) for row in rows]
    if not episodes or episodes != list(range(1, episodes[-1] + 1)):
        raise ValueError("Combined DQN history must contain consecutive episodes starting at 1, without duplicates")
    return rows, paths, boundaries


def rolling(values, window=50):
    values = np.asarray(values, dtype=float)
    return np.asarray([
        np.nanmean(values[max(0, index - window + 1):index + 1])
        if np.isfinite(values[max(0, index - window + 1):index + 1]).any()
        else np.nan
        for index in range(len(values))
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_experiment_argument(parser)
    parser.add_argument("--dqn-logs", nargs="+", help="Explicit chronological training CSVs if resume ancestry is ambiguous")
    parser.add_argument('--mlpx-log', type=Path, help='Explicit WM log; useful when an experiment has several candidates')
    parser.add_argument('--skip-mlpx', action='store_true',
                        help='Export only DQN plots when reusing a world model from another experiment')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    results = root / "results" / f"experiment_{args.experiment}"
    output = root / "docs" / "training_graphs"
    output.mkdir(parents=True, exist_ok=True)
    plt.rcdefaults()

    mlpx_log = None
    if not args.skip_mlpx:
        mlpx_log = args.mlpx_log or latest_log(results, "MLPx", "mlpx_training.csv")
        rows = read_rows(mlpx_log)
        epochs = [int(row["epoch"]) for row in rows]
        fig, axis = plt.subplots(figsize=(10, 6))
        for key in ["train_loss", "test_loss", "state_loss", "reward_loss"]:
            axis.plot(epochs, [float(row[key]) for row in rows], label=key)
        axis.set(title=f"MLPx training · Experiment {args.experiment}",
                 xlabel="Epoch", ylabel="Loss")
        axis.grid(True)
        axis.legend()
        fig.tight_layout()
        fig.savefig(output / f"experiment_{args.experiment}_mlpx_training.png", dpi=150)
        plt.close(fig)

    dqn_log = latest_log(results, "RequestDQN", "request_dqn_training.csv")
    rows, dqn_logs, boundaries = load_dqn_history(dqn_log, args.dqn_logs)
    episodes = [int(row["episode"]) for row in rows]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for axis, key, title, ylabel, color in [
        (axes[0, 0], "mean_loss", "DQN mean loss", "Loss", "#c51b1d"),
        (axes[0, 1], "return", "Episode return (50-episode average)", "Return", "#214ce5"),
        (axes[1, 0], "requests", "Requests per episode", "Requests", "#7024e8"),
    ]:
        values = [float(row[key]) if row[key] else np.nan for row in rows]
        axis.plot(episodes, values, color=color, alpha=.2, linewidth=1.5)
        axis.plot(episodes, rolling(values), color=color, linewidth=2)
        axis.set(title=title, ylabel=ylabel)
    axis = axes[1, 1]
    axis.plot(episodes, [float(row["epsilon"]) for row in rows],
              label="epsilon", color="#009e73")
    axis.plot(episodes, rolling([float(row["crashed"]) for row in rows]),
              label="crash flag average", color="#e82424")
    axis.set(title="Exploration and crash signal", ylabel="Value")
    axis.legend()
    for axis in axes.flat:
        for boundary in boundaries:
            axis.axvline(boundary + .5, color="#64748b", linestyle="--", alpha=.8)
        axis.set_xlabel("Episode")
        axis.grid(alpha=.3)
    note = " · dashed line: resume" if boundaries else ""
    fig.suptitle(f"Request DQN training · Experiment {args.experiment}{note}")
    fig.tight_layout()
    fig.savefig(output / f"experiment_{args.experiment}_dqn_training.png", dpi=160)
    plt.close(fig)
    sources = sorted({row.get('scenario', '') for row in rows} - {''})
    if len(sources) > 1:
        fig, axes = plt.subplots(2, 2, figsize=(12, 7))
        for source in sources:
            selected = [row for row in rows if row['scenario'] == source]
            x = [int(row['episode']) for row in selected]
            for axis, key, title in zip(axes.flat, ['return', 'requests', 'crashed', 'mean_loss'],
                                         ['Return by traffic type', 'Requests by traffic type',
                                          'Training crash fraction by traffic type', 'Loss by traffic type']):
                values = [float(row[key]) if row[key] else np.nan for row in selected]
                axis.plot(x, rolling(values, 25), label=source)
                axis.set_title(title)
                axis.set_xlabel('Overall episode')
                axis.grid(alpha=.3)
                axis.legend()
        fig.suptitle(f'Request DQN · Experiment {args.experiment} · 25 source-episode rolling means')
        fig.tight_layout()
        fig.savefig(output / f'experiment_{args.experiment}_dqn_by_scenario.png', dpi=160)
        plt.close(fig)
    source = mlpx_log if mlpx_log is not None else 'DQN log only'
    print(f"Exported Experiment {args.experiment} plots from {source}")
    print(f"DQN: episodes {episodes[0]}–{episodes[-1]} from {len(dqn_logs)} related logs")
    for path in dqn_logs:
        print(f"  {path}")


if __name__ == "__main__":
    main()
