import unittest
import random
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from data.combined_world_model import encode_messages
from scripts.training.train_request_dqn import (
    ASK_NOBODY, optimize, valid_actions, requested_message,
    resolve_requested_vehicle, select_action,
)
from scripts.data_collection.show_visible_neighbors import SLOT_ORDER
from scripts.evaluation.evaluate_request_dqn import choose_request
from src.cooperative_driving.policies.request_dqn import RequestDQN
from scripts.evaluation.evaluate_mlpx_rollout import evaluate_state
from src.cooperative_driving.models.MLPx.mlpx import MLPxWorldModel
from src.cooperative_driving.planner.MCP import MCPPlanner


class AlignmentTests(unittest.TestCase):
    def test_each_action_retains_its_spatial_slot_with_gaps(self):
        for index, slot in enumerate(SLOT_ORDER):
            vehicle = SimpleNamespace(vehicle_id=100 + index)
            visible = {slot: {"vehicle": vehicle}}
            self.assertEqual(valid_actions(visible), [index, ASK_NOBODY])
            self.assertIs(resolve_requested_vehicle(visible, index), vehicle)
            env = SimpleNamespace(request_neighbor_action=lambda vid: {"vehicle_id": vid})
            bundle = requested_message(env, visible, index)["messages_by_slot"]
            self.assertEqual(bundle[index], {"vehicle_id": 100 + index})
            self.assertEqual(sum(item is not None for item in bundle), 1)
            if index:
                with self.assertRaises(ValueError):
                    requested_message(env, visible, 0)

    def test_other_neighbors_do_not_change_request_meaning(self):
        vehicle = SimpleNamespace(vehicle_id=7)
        visible = {SLOT_ORDER[7]: {"vehicle": vehicle}}
        self.assertIs(resolve_requested_vehicle(visible, 7), vehicle)
        visible[SLOT_ORDER[0]] = {"vehicle": SimpleNamespace(vehicle_id=99)}
        self.assertIs(resolve_requested_vehicle(visible, 7), vehicle)
        self.assertEqual(valid_actions({}), [ASK_NOBODY])
        self.assertIsNone(resolve_requested_vehicle({}, ASK_NOBODY))

    def test_selection_masks_empty_slots_for_exploration_and_greedy(self):
        model = torch.nn.Linear(45, 9)
        with torch.no_grad():
            model.weight.zero_()
            model.bias.fill_(10000)
            model.bias[7] = 2
            model.bias[8] = 1
        valid = [7, 8]
        self.assertEqual(select_action(model, np.zeros(45, dtype=np.float32), valid, 0, random.Random(0)), 7)
        rng = random.Random(0)
        for _ in range(50):
            self.assertIn(select_action(model, np.zeros(45, dtype=np.float32), valid, 1, rng), valid)

    def test_random_baseline_requests_occupied_spatial_slot(self):
        visible = {SLOT_ORDER[7]: {"vehicle": SimpleNamespace(vehicle_id=17)}}
        env = SimpleNamespace(request_neighbor_action=lambda vid: {"vehicle_id": vid})
        action, bundle = choose_request("random", None, env, visible, random.Random(0))
        self.assertEqual(action, 7)
        self.assertEqual(bundle["messages_by_slot"][7]["vehicle_id"], 17)

    def test_checkpoint_versions_preserve_legacy_and_new_semantics(self):
        model = RequestDQN()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.pt"
            torch.save({"model_state_dict": model.state_dict()}, path)
            with self.assertWarnsRegex(UserWarning, "Legacy DQN"):
                legacy = RequestDQN.load(path)
            visible = {SLOT_ORDER[7]: {"vehicle": SimpleNamespace(vehicle_id=17)}}
            self.assertEqual(valid_actions(visible, legacy.request_action_encoding), [0, 8])
            self.assertEqual(resolve_requested_vehicle(visible, 0, legacy.request_action_encoding).vehicle_id, 17)
            torch.save({"model_state_dict": model.state_dict(),
                        "request_action_encoding": "spatial_v1"}, path)
            fixed = RequestDQN.load(path)
            self.assertEqual(valid_actions(visible, fixed.request_action_encoding), [7, 8])

    def test_sparse_message_matches_inference_encoding(self):
        message = {"behavior_type": "aggressive", "target_lane": ["0", "1", 2],
                   "target_speed": 27, "acceleration": -2, "steering": .1}
        record = {"messages": {"7": message},
                  "visible_vehicle_slots": [None, None, None, 7, None, None, None, None]}
        trained = torch.tensor(encode_messages(record))
        inferred = MLPxWorldModel._message_vector({
            "messages_by_slot": [None, None, None, message, None, None, None, None]})
        torch.testing.assert_close(trained, inferred)
        self.assertEqual(trained[:21].count_nonzero(), 0)

    def test_legacy_messages_fail_instead_of_silently_shifting(self):
        with self.assertRaisesRegex(ValueError, "Legacy packed"):
            encode_messages({"messages": {"7": {}}, "visible_vehicle_slots": [7]})
        self.assertEqual(encode_messages({"messages": None}), [0.] * 56)

    def test_invalid_target_does_not_inflate_loss(self):
        online = torch.nn.Linear(1, 9)
        target = torch.nn.Linear(1, 9)
        with torch.no_grad():
            online.weight.zero_()
            online.bias.zero_()
            target.weight.zero_()
            target.bias.fill_(10000)
            target.bias[8] = 2
        optimizer = torch.optim.SGD(online.parameters(), lr=0.)
        # Only ask-nobody is available at the next state.
        replay = [(np.zeros(1), 8, 0., np.zeros(1), False, [8])]
        loss = optimize(online, target, optimizer, replay, 1, .5)
        self.assertAlmostEqual(loss, .5)

    def test_terminal_candidates_remain_in_evaluation(self):
        class Env:
            class action_space:
                n = 2
            def step(self, action):
                return np.zeros(55), 1., True, False, {}
            def close(self):
                pass
        model = MLPxWorldModel()
        with patch.object(model, "predict_step", return_value=(torch.zeros(55), 1.)):
            result = evaluate_state(model, Env(), np.zeros(55), None, 3, "constant")
        self.assertEqual(result["crashed_candidates"], 2)
        self.assertAlmostEqual(result["reward_mae"], 2.)

    def test_sequence_evaluation_matches_planner_first_action(self):
        class Env:
            class action_space:
                n = 5
            def step(self, action):
                return np.zeros(55), 1., False, False, {}
            def close(self):
                pass
        torch.manual_seed(10)
        model = MLPxWorldModel().eval()
        observation = np.zeros(55)
        expected = MCPPlanner(model, planning_horizon=2).plan(observation)
        metrics = evaluate_state(model, Env(), observation, None, 2, "planner")
        self.assertEqual(metrics["predicted_best_action"], expected)


if __name__ == "__main__":
    unittest.main()
