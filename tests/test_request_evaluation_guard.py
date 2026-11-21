import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from scripts.evaluation.evaluate_request_dqn import evaluate_policy
from src.cooperative_driving.environments.customEnv import CustomEnv


class IdleModel:
    def observe(self, environment):
        return environment.observation_type.observe()

    def predict_step(self, state, action, message=None):
        return torch.as_tensor(state).reshape(-1), float(action == 1)


class GuardedEvaluationTests(unittest.TestCase):
    def test_paired_scenes_cost_and_progress_with_complete_guard(self):
        args = SimpleNamespace(seed=100220, episodes=1, planning_horizon=1,
                               visibility=60., side_window=10., safety_guard=True,
                               min_clearance=8., min_ttc=3., lane_change_guard=True,
                               max_target_speed=30., communication_cost=.05)
        with patch('scripts.evaluation.evaluate_request_dqn.CustomEnv',
                   side_effect=lambda: CustomEnv(config={'duration': 2})):
            none = evaluate_policy('none', None, IdleModel(), args)[0]
            all_requests = evaluate_policy('all', None, IdleModel(), args)[0]
        self.assertEqual(none['initial_scene_sha256'], all_requests['initial_scene_sha256'])
        self.assertEqual(none['steps'], 2)
        self.assertEqual(none['requests'], 0)
        self.assertGreater(all_requests['requests'], 0)
        self.assertAlmostEqual(all_requests['return'],
                               all_requests['driving_reward'] - .05 * all_requests['requests'])
        self.assertGreater(none['mean_speed'], 20.)
        self.assertEqual(none['completed_lane_changes'], 0)


if __name__ == '__main__':
    unittest.main()
