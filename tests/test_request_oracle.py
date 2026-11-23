import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from scripts.data_collection.show_visible_neighbors import visible_neighbors
from scripts.policy.request_oracle import (
    agrees_with_oracle,
    oracle_choice,
    oracle_request_values,
    oracle_target_distribution,
)
from scripts.training.train_request_dqn import (
    ASK_NOBODY,
    balanced_ask_weight,
    behaviour_cloning_loss,
    demo_margin_loss,
    load_oracle_demos,
    pretrain_behaviour_cloning,
    sample_demo_indices,
)
from src.cooperative_driving.environments.intent_scenarios import intent_episode
from src.cooperative_driving.models.MLPx.mlpx import MLPxWorldModel
from src.cooperative_driving.planner.MCP import MCPPlanner
from src.cooperative_driving.policies.request_dqn import RequestDQN


def values_with(entries):
    values = np.full(9, -np.inf)
    for index, value in entries.items():
        values[index] = value
    return values


class OracleChoiceTests(unittest.TestCase):
    def test_tie_with_ask_nobody_picks_ask_nobody(self):
        values = values_with({0: 1.0, 3: 1.0, ASK_NOBODY: 1.0})
        self.assertEqual(oracle_choice(values), ASK_NOBODY)
        np.testing.assert_array_equal(np.flatnonzero(oracle_target_distribution(values)), [ASK_NOBODY])

    def test_strictly_better_request_is_chosen(self):
        values = values_with({2: 1.2, 5: 1.2, ASK_NOBODY: 1.0})
        self.assertEqual(oracle_choice(values), 2)
        target = oracle_target_distribution(values)
        np.testing.assert_allclose(target[[2, 5]], [.5, .5])
        self.assertAlmostEqual(float(target.sum()), 1.0)

    def test_redundant_request_disagrees_with_oracle(self):
        values = values_with({1: 0.95, ASK_NOBODY: 1.0})
        self.assertFalse(agrees_with_oracle(1, values))
        self.assertTrue(agrees_with_oracle(ASK_NOBODY, values))


class OracleRolloutTests(unittest.TestCase):
    def test_values_use_real_rollouts_and_leave_scene_unchanged(self):
        torch.manual_seed(0)
        model = MLPxWorldModel(state_size=57)
        environment = intent_episode(1300003, 'merge')
        try:
            planner = MCPPlanner(model, action_size=environment.action_space.n)
            visible = visible_neighbors(environment, 60., 10.)
            before = [vehicle.position.copy() for vehicle in environment.road.vehicles]
            controls = dict(safety_guard=True, lane_change_guard=True, max_target_speed=30., batched=True)
            values, first_actions = oracle_request_values(
                environment, visible, planner, model, controls, lookahead=3, communication_cost=.05)
            after = [vehicle.position for vehicle in environment.road.vehicles]
            for old, new in zip(before, after):
                np.testing.assert_array_equal(old, new)
            self.assertTrue(np.isfinite(values[ASK_NOBODY]))
            for option, action in first_actions.items():
                if option != ASK_NOBODY and action == first_actions[ASK_NOBODY]:
                    self.assertAlmostEqual(values[option], values[ASK_NOBODY] - .05)
            invalid = [index for index in range(9) if index not in first_actions]
            self.assertTrue(np.all(np.isneginf(values[invalid])))
        finally:
            environment.close()

    def test_comfort_controller_needs_state_and_keeps_it_unchanged(self):
        torch.manual_seed(0)
        model = MLPxWorldModel(state_size=57)
        environment = intent_episode(1300004, 'stay')
        try:
            with self.assertRaises(ValueError):
                oracle_request_values(environment, {}, None, None, {'comfort_ttc': 6.})
            planner = MCPPlanner(model, action_size=environment.action_space.n)
            visible = visible_neighbors(environment, 60., 10.)
            controls = dict(safety_guard=True, lane_change_guard=True, max_target_speed=30.,
                            comfort_ttc=6., batched=True)
            state = {'lane_change_cooldown': 2, 'acceleration_cooldown': 1}
            values, _ = oracle_request_values(environment, visible, planner, model, controls,
                                              lookahead=3, controller_state=state)
            self.assertEqual(state, {'lane_change_cooldown': 2, 'acceleration_cooldown': 1})
            self.assertTrue(np.isfinite(values[ASK_NOBODY]))
        finally:
            environment.close()


def synthetic_demos(count=64):
    rng = np.random.default_rng(0)
    states = rng.normal(size=(count, 45)).astype(np.float32)
    valid = np.zeros((count, 9), dtype=bool)
    valid[:, [0, 3, ASK_NOBODY]] = True
    target = np.zeros((count, 9), dtype=np.float32)
    asks = states[:, 0] > 0
    target[asks, 3] = 1.
    target[~asks, ASK_NOBODY] = 1.
    return {
        'state': states, 'target': target, 'valid': valid, 'best': target > 0,
        'choice': np.where(asks, 3, ASK_NOBODY), 'reward': np.zeros(count, dtype=np.float32),
        'next_state': states, 'done': np.ones(count, dtype=bool), 'next_valid': valid,
        'source': np.asarray(['intent'] * (count // 4) + ['ordinary'] * (count - count // 4)),
    }


class OracleTrainingTests(unittest.TestCase):
    def test_margin_loss_is_zero_only_when_oracle_option_leads(self):
        valid = torch.tensor([[True, True, False]])
        best = torch.tensor([[True, False, False]])
        ahead = torch.tensor([[1.0, 0.9, 5.0]])
        behind = torch.tensor([[1.0, 0.99, 5.0]])
        self.assertAlmostEqual(float(demo_margin_loss(ahead, best, valid, .05)), 0.)
        self.assertAlmostEqual(float(demo_margin_loss(behind, best, valid, .05)), .04, places=5)

    def test_bc_loss_ignores_invalid_slots(self):
        model = RequestDQN()
        states = torch.zeros(2, 45)
        valid = torch.zeros(2, 9, dtype=torch.bool)
        valid[:, [1, ASK_NOBODY]] = True
        target = torch.zeros(2, 9)
        target[:, ASK_NOBODY] = 1.
        with torch.no_grad():
            model.network[-1].bias[:] = 0.
            model.network[-1].bias[4] = 100.
        with torch.no_grad():
            loss = float(behaviour_cloning_loss(model, states, target, valid))
        self.assertTrue(np.isfinite(loss))
        self.assertLess(loss, 2.)

    def test_ask_weight_scales_only_ask_records(self):
        model = RequestDQN()
        states = torch.zeros(2, 45)
        valid = torch.ones(2, 9, dtype=torch.bool)
        target = torch.zeros(2, 9)
        target[0, ASK_NOBODY] = 1.
        target[1, 2] = 1.
        with torch.no_grad():
            model.network[-1].weight.zero_()
            model.network[-1].bias.zero_()
            model.network[-1].bias[ASK_NOBODY] = 3.
            plain = float(behaviour_cloning_loss(model, states, target, valid))
            weighted = float(behaviour_cloning_loss(model, states, target, valid, ask_weight=3.))
        self.assertGreater(weighted, plain)
        demos = {'target': np.zeros((10, 9), dtype=np.float32)}
        demos['target'][:8, ASK_NOBODY] = 1.
        demos['target'][8:, 1] = 1.
        self.assertAlmostEqual(balanced_ask_weight(demos), 4.)

    def test_demo_sampling_balances_sources(self):
        demos = synthetic_demos()
        np.random.seed(3)
        indices = sample_demo_indices(demos, 20)
        sources = demos['source'][indices]
        self.assertEqual(int(np.sum(sources == 'intent')), 10)
        self.assertEqual(int(np.sum(sources == 'ordinary')), 10)

    def test_behaviour_cloning_learns_oracle_choice(self):
        torch.manual_seed(0)
        np.random.seed(0)
        demos = synthetic_demos(256)
        model = RequestDQN()
        history = pretrain_behaviour_cloning(model, demos, epochs=30, batch_size=64, lr=1e-2)
        self.assertLess(history[-1], history[0])
        with torch.no_grad():
            scores = model(torch.tensor(demos['state'])).masked_fill(~torch.tensor(demos['valid']), -1e9)
        accuracy = float(np.mean(scores.argmax(1).numpy() == demos['choice']))
        self.assertGreater(accuracy, .9)

    def test_demo_files_must_match_state_encoding(self):
        demos = synthetic_demos(8)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'labels.npz'
            arrays = {key: value for key, value in demos.items() if key != 'best'}
            np.savez(path, metadata=json.dumps({'request_state_encoding': 'ego_control_v1'}), **arrays)
            loaded = load_oracle_demos([path], 'ego_control_v1')
            np.testing.assert_array_equal(loaded['best'], demos['best'])
            with self.assertRaises(ValueError):
                load_oracle_demos([path], 'kinematics_v1')


if __name__ == '__main__':
    unittest.main()
