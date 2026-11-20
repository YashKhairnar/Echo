"""Export training CSV plots into the documentation for an experiment."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.utils.experiment_utils import add_experiment_argument


def latest_log(root, model, filename):
    paths = sorted((root / model / "experiments").glob(f"*/{filename}"))
    if not paths:
        raise FileNotFoundError(f"No {model} training log found under {root}")
    return paths[-1]


def read_rows(path):
    with path.open() as file:
        return list(csv.DictReader(file))


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
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    results = root / "results" / f"experiment_{args.experiment}"
    output = root / "docs" / "training_graphs"
    output.mkdir(parents=True, exist_ok=True)
    plt.rcdefaults()

    mlpx_log = latest_log(results, "MLPx", "mlpx_training.csv")
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
    rows = read_rows(dqn_log)
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
        axis.set_xlabel("Episode")
        axis.grid(alpha=.3)
    fig.suptitle(f"Request DQN training · Experiment {args.experiment}")
    fig.tight_layout()
    fig.savefig(output / f"experiment_{args.experiment}_dqn_training.png", dpi=160)
    plt.close(fig)
    print(f"Exported Experiment {args.experiment} plots from {mlpx_log} and {dqn_log}")


if __name__ == "__main__":
    main()
