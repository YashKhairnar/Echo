import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.evaluation.compare_request_runs import (
    clustered_bootstrap,
    exact_mcnemar,
    load_rows,
    paired_differences,
)


def write_rows(path, policy, returns):
    with open(path, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['policy', 'seed', 'hidden_mode', 'return', 'crashed'])
        writer.writeheader()
        for (seed, mode), value in returns.items():
            writer.writerow({'policy': policy, 'seed': seed, 'hidden_mode': mode,
                             'return': value, 'crashed': 0})


class CompareRequestRunsTests(unittest.TestCase):
    def test_pairs_by_scene_and_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory) / 'a.csv', Path(directory) / 'b.csv'
            write_rows(a, 'none', {(1, 'stay'): 1.0, (1, 'merge'): 2.0, (2, 'stay'): 3.0})
            write_rows(b, 'dqn', {(1, 'stay'): 1.5, (1, 'merge'): 2.0, (2, 'stay'): 2.0})
            clusters = paired_differences(load_rows(a, 'none'), load_rows(b, 'dqn'), 'return')
            np.testing.assert_allclose(clusters[0], [0.5, 0.0])
            np.testing.assert_allclose(clusters[1], [-1.0])
            write_rows(b, 'dqn', {(1, 'stay'): 1.5})
            with self.assertRaises(ValueError):
                paired_differences(load_rows(a, 'none'), load_rows(b, 'dqn'), 'return')

    def test_exact_mcnemar_matches_binomial_test(self):
        self.assertAlmostEqual(exact_mcnemar(0, 0), 1.0)
        # 11 discordant pairs, all on one side: 2 * 0.5**11.
        self.assertAlmostEqual(exact_mcnemar(11, 0), 2 / 2 ** 11)
        # 6 vs 1: 2 * (1 + 7) / 2**7 = 0.125.
        self.assertAlmostEqual(exact_mcnemar(6, 1), 0.125)
        self.assertAlmostEqual(exact_mcnemar(1, 6), 0.125)
        self.assertEqual(exact_mcnemar(3, 3), 1.0)

    def test_parts_are_joined_and_duplicates_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            first, second = Path(directory) / 'a.csv', Path(directory) / 'b.csv'
            write_rows(first, 'none', {(1, 'stay'): 1.0})
            write_rows(second, 'none', {(2, 'stay'): 2.0})
            self.assertEqual(set(load_rows([first, second], 'none')), {(1, 'stay'), (2, 'stay')})
            with self.assertRaises(ValueError):
                load_rows([first, first], 'none')

    def test_bootstrap_mean_is_per_trial_and_constant_gives_point_interval(self):
        clusters = [np.asarray([1.0, 1.0]), np.asarray([1.0])]
        mean, low, high = clustered_bootstrap(clusters, samples=200)
        self.assertAlmostEqual(mean, 1.0)
        self.assertAlmostEqual(low, 1.0)
        self.assertAlmostEqual(high, 1.0)
        mean, _, _ = clustered_bootstrap([np.asarray([0.0, 0.0]), np.asarray([3.0])], samples=10)
        self.assertAlmostEqual(mean, 1.0)


if __name__ == '__main__':
    unittest.main()
