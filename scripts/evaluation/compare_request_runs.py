"""Paired comparison of two request-policy evaluations on the same seeds.

Reads two request_dqn_evaluation.csv files from evaluate_request_dqn, pairs the
rows by (seed, hidden_mode), and prints the mean difference (B minus A) in
cost-adjusted return and in crashes with a 95% bootstrap interval. The
bootstrap resamples whole seeds (scenes), so the stay and merge trials of one
scene stay together.

Run from the project root with:

    .venv/bin/python -m scripts.evaluation.compare_request_runs \
        --a results/.../request_dqn_evaluation.csv --a-policy none \
        --b results/.../request_dqn_evaluation.csv --b-policy dqn
"""

import argparse
import csv
from collections import defaultdict

import numpy as np


def load_rows(path, policy):
    with open(path, newline='') as file:
        rows = [row for row in csv.DictReader(file) if row['policy'] == policy]
    if not rows:
        raise ValueError(f'No rows for policy {policy} in {path}')
    return {(int(row['seed']), row['hidden_mode']): row for row in rows}


def paired_differences(rows_a, rows_b, field):
    if set(rows_a) != set(rows_b):
        raise ValueError('The two evaluations do not cover the same seeds and modes')
    by_seed = defaultdict(list)
    for key in rows_a:
        by_seed[key[0]].append(float(rows_b[key][field]) - float(rows_a[key][field]))
    return [np.asarray(values) for _, values in sorted(by_seed.items())]


def clustered_bootstrap(clusters, samples=10000, seed=0):
    """Mean per-trial difference and a 95% interval, resampling whole clusters."""
    rng = np.random.default_rng(seed)
    sums = np.asarray([cluster.sum() for cluster in clusters])
    counts = np.asarray([len(cluster) for cluster in clusters])
    mean = float(sums.sum() / counts.sum())
    picks = rng.integers(0, len(clusters), size=(samples, len(clusters)))
    means = sums[picks].sum(axis=1) / counts[picks].sum(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return mean, float(low), float(high)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--a', required=True)
    parser.add_argument('--a-policy', required=True)
    parser.add_argument('--b', required=True)
    parser.add_argument('--b-policy', required=True)
    parser.add_argument('--samples', type=int, default=10000)
    args = parser.parse_args()
    rows_a = load_rows(args.a, args.a_policy)
    rows_b = load_rows(args.b, args.b_policy)
    for field in ('return', 'crashed', 'requests'):
        mean, low, high = clustered_bootstrap(paired_differences(rows_a, rows_b, field), args.samples)
        print(f'{field:8s} B-A = {mean:+.4f}  95% CI [{low:+.4f}, {high:+.4f}]  (trials={len(rows_a)})')


if __name__ == '__main__':
    main()
