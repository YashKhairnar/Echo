"""Paired comparison of two request-policy evaluations on the same seeds.

Reads request_dqn_evaluation.csv files from evaluate_request_dqn (several
files per side are joined, for runs split into parts), pairs the rows by
(seed, hidden_mode), and prints:

- the mean difference (B minus A) in cost-adjusted return, crashes and
  requests, with a 95% bootstrap interval that resamples whole seeds
  (scenes), so the stay and merge trials of one scene stay together;
- an exact two-sided McNemar test on the paired crash outcomes;
- the mean ego speed per hidden mode for A and B.

Run from the project root with:

    .venv/bin/python -m scripts.evaluation.compare_request_runs \
        --a results/.../request_dqn_evaluation.csv --a-policy none \
        --b results/.../request_dqn_evaluation.csv --b-policy dqn
"""

import argparse
import csv
import math
import os
from collections import defaultdict

import numpy as np


def load_rows(paths, policy):
    """Rows of one policy from one or more CSV files, keyed by (seed, mode)."""
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]
    rows = {}
    for path in paths:
        with open(path, newline='') as file:
            for row in csv.DictReader(file):
                if row['policy'] != policy:
                    continue
                key = (int(row['seed']), row['hidden_mode'])
                if key in rows:
                    raise ValueError(f'Trial {key} appears twice for policy {policy}')
                rows[key] = row
    if not rows:
        raise ValueError(f'No rows for policy {policy} in {paths}')
    return rows


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


def exact_mcnemar(only_a, only_b):
    """Two-sided exact McNemar p-value from the two discordant counts."""
    total = only_a + only_b
    if total == 0:
        return 1.0
    tail = sum(math.comb(total, k) for k in range(min(only_a, only_b) + 1)) / 2 ** total
    return min(1.0, 2 * tail)


def crash_table(rows_a, rows_b):
    """Counts of (A crashed, B crashed) over the paired trials."""
    counts = {'both': 0, 'only_a': 0, 'only_b': 0, 'neither': 0}
    for key in rows_a:
        a, b = int(rows_a[key]['crashed']), int(rows_b[key]['crashed'])
        counts['both' if a and b else 'only_a' if a else 'only_b' if b else 'neither'] += 1
    return counts


def speed_by_mode(rows):
    groups = defaultdict(list)
    for (_, mode), row in rows.items():
        groups[mode].append(float(row['mean_speed']))
    return {mode: float(np.mean(values)) for mode, values in sorted(groups.items())}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--a', nargs='+', required=True)
    parser.add_argument('--a-policy', required=True)
    parser.add_argument('--b', nargs='+', required=True)
    parser.add_argument('--b-policy', required=True)
    parser.add_argument('--samples', type=int, default=10000)
    args = parser.parse_args()
    rows_a = load_rows(args.a, args.a_policy)
    rows_b = load_rows(args.b, args.b_policy)
    for field in ('return', 'crashed', 'requests'):
        mean, low, high = clustered_bootstrap(paired_differences(rows_a, rows_b, field), args.samples)
        print(f'{field:8s} B-A = {mean:+.4f}  95% CI [{low:+.4f}, {high:+.4f}]  (trials={len(rows_a)})')
    table = crash_table(rows_a, rows_b)
    print(f"crashes  A={table['both'] + table['only_a']} B={table['both'] + table['only_b']} "
          f"both={table['both']} only_A={table['only_a']} only_B={table['only_b']} "
          f"exact McNemar p={exact_mcnemar(table['only_a'], table['only_b']):.3g}")
    for name, rows in (('A', rows_a), ('B', rows_b)):
        speeds = ' '.join(f'{mode}={speed:.2f}' for mode, speed in speed_by_mode(rows).items())
        print(f'speed {name} (m/s): {speeds}')


if __name__ == '__main__':
    main()
