import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from scripts.training.train_request_dqn import (
    ASK_NOBODY,
    ScenarioReplay,
    redundant_request_penalty,
    request_state,
)
from src.cooperative_driving.environments.intent_scenarios import matched_intent_pair
from src.cooperative_driving.models.MLPx.mlpx import MLPxWorldModel
from src.cooperative_driving.policies.request_dqn import RequestDQN
from src.cooperative_driving.planner.MCP import MCPPlanner
from scripts.data_collection.show_visible_neighbors import visible_neighbors


class MixedRequestTests(unittest.TestCase):
    def test_teacher_penalizes_only_redundant_requests(self):
        self.assertEqual(redundant_request_penalty(ASK_NOBODY, 1, 1, .1), 0.)
        self.assertEqual(redundant_request_penalty(2, 1, 0, .1), 0.)
        self.assertAlmostEqual(redundant_request_penalty(2, 1, 1, .1), -.1)

    def test_replay_balances_unequal_sources_and_restores(self):
        replay = ScenarioReplay(100)
        for source, count in [('ordinary', 40), ('intent', 8)]:
            for index in range(count):
                replay.append((np.zeros(45), index, float(source == 'intent'), np.zeros(45), False, [8], source))
        random.seed(9)
        batch = replay.sample(16)
        self.assertEqual(sum(item[2] for item in batch), 8.)
        self.assertEqual(len(batch), 16)
        restored = ScenarioReplay(100)
        restored.extend(list(replay))
        random.seed(12)
        first = [item[1] for item in replay.sample(16)]
        random.seed(12)
        self.assertEqual(first, [item[1] for item in restored.sample(16)])

    def test_control_state_encoding_is_public_and_versioned(self):
        environments, _ = matched_intent_pair(920000)
        try:
            states = []
            for environment in environments.values():
                visible = visible_neighbors(environment, 60., 10.)
                old = request_state(environment, visible)
                new = request_state(environment, visible, 'ego_control_v1')
                self.assertEqual(new.size, 45)
                np.testing.assert_array_equal(old[3:5], [0., 0.])
                self.assertAlmostEqual(float(new[3]), 25 / 30, places=6)
                states.append(new)
            np.testing.assert_array_equal(*states)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory)/'dqn.pt'
                torch.save({'model_state_dict': RequestDQN().state_dict(),
                            'request_action_encoding': 'spatial_v1',
                            'request_state_encoding': 'ego_control_v1'}, path)
                self.assertEqual(RequestDQN.load(path).request_state_encoding, 'ego_control_v1')
        finally:
            for environment in environments.values():
                environment.close()

    def test_batched_planner_matches_recursive_values(self):
        environments, _ = matched_intent_pair(920001)
        try:
            model = MLPxWorldModel(state_size=57)
            environment = environments['stay']
            planner = MCPPlanner(model, planning_horizon=3)
            observation = model.observe(environment)
            np.testing.assert_allclose(planner.action_values(observation),
                                       planner.action_values_batched(observation), atol=1e-6)
        finally:
            for environment in environments.values():
                environment.close()


if __name__ == '__main__':
    unittest.main()
