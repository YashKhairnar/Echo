import unittest
from types import SimpleNamespace

import numpy as np

from scripts.evaluation.investigate_planner import controller_alias_probes, real_search
from scripts.evaluation.diagnose_driving import select_driving


class ToyEnvironment:
    def __init__(self):
        self.action_space = SimpleNamespace(n=2)
        self.depth = 0

    def step(self, action):
        self.depth += 1
        if action == 0:
            return None, 1., True, False, {}
        return None, (2. if self.depth == 1 else 10.), False, False, {}

    def close(self):
        pass


class InvestigationTests(unittest.TestCase):
    def test_oracle_stops_terminal_branch_and_discounts_future(self):
        environment = ToyEnvironment()
        values = real_search(environment, 2)
        self.assertEqual(values[0], {'value': 1., 'sequence': [0]})
        self.assertAlmostEqual(values[1]['value'], 11.9)
        self.assertEqual(values[1]['sequence'], [1, 1])
        self.assertEqual(environment.depth, 0)

    def test_controller_memory_is_not_in_observation(self):
        probes = controller_alias_probes()
        self.assertTrue(all(probe['input_equal'] for probe in probes))
        self.assertGreater(probes[1]['next_speed'] - probes[0]['next_speed'], 8.)
        self.assertFalse(np.allclose(probes[0]['next_state'], probes[2]['next_state']))

    def test_optional_speed_mask_preserves_raw_values(self):
        environment = SimpleNamespace(vehicle=SimpleNamespace(target_speed=30.))
        planner = SimpleNamespace(action_values=lambda *_: [0., 2., 1., 3., 0.])
        action, _, info, values = select_driving('wm_none', environment, {}, None,
                                                planner, np.zeros(55), None,
                                                max_target_speed=30.)
        self.assertEqual(action, 1)
        self.assertTrue(info['speed_cap_blocked'])
        self.assertEqual(values[3], 3.)
        action, _, info, _ = select_driving('wm_none', environment, {}, None,
                                           planner, np.zeros(55), None)
        self.assertEqual(action, 3)
        self.assertFalse(info['speed_cap_blocked'])


if __name__ == '__main__':
    unittest.main()
