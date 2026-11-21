import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
from collections import Counter, defaultdict

import numpy as np
import torch

from data.combined_world_model import CombinedWorldModelDataset, episode_group, heldout_episode_groups, encode_messages
from scripts.data_collection.collect_all_action_dataset import action_records, request_scenarios, collect_episode
from scripts.evaluation.evaluate_mlpx_actions import accumulate
from scripts.training.train_mlpx_wm import balanced_sampler
from src.cooperative_driving.environments.customEnv import CustomEnv


class ActionCoverageTests(unittest.TestCase):
    def test_compact_training_records_preserve_encoded_tensors(self):
        record = {"state": [[0.] * 5] * 11, "next_state": [[1.] * 5] * 11,
                  "ego_action": 2, "reward": .5, "episode": 3,
                  "visible_vehicle_slots": [None, 7] + [None] * 6,
                  "messages": {"7": {"vehicle_id": 7, "target_speed": 27.}}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.jsonl"
            path.write_text(json.dumps(record) + "\n")
            full = CombinedWorldModelDataset(path)
            compact = CombinedWorldModelDataset(path, compact_records=True)
            for a, b in zip(full[0], compact[0]):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            self.assertEqual(full.groups, compact.groups)
            self.assertEqual(compact.records, [{"ego_action": 2, "messages": True}])

    def test_request_modes_remove_duplicates_and_handle_no_neighbors(self):
        self.assertEqual(request_scenarios([7], ["none", "single", "all"]), [(), (7,)])
        self.assertEqual(request_scenarios([], ["all"]), [()])
        self.assertEqual(len(request_scenarios([1, 2, 3], ["none", "single", "all", "subsets"])), 8)

    def test_seed_local_episode_is_reproducible_independent_of_call_order(self):
        config = dict(seed=200000, steps=1, message_modes=["none", "single", "all"],
                      rollout_policy="mixed", visibility=60., side_window=10.)
        first = collect_episode((0, config))
        collect_episode((1, config))
        self.assertEqual(first, collect_episode((0, config)))

    def test_all_actions_start_from_identical_scene_without_advancing_rollout(self):
        environment = CustomEnv(config={"duration": 1, "initial_lane_id": 0})
        try:
            observation, _ = environment.reset(seed=10)
            original_positions = [vehicle.position.copy() for vehicle in environment.road.vehicles]
            original_step = CustomEnv.step
            with patch.object(CustomEnv, "step", autospec=True, side_effect=original_step) as step:
                records = list(action_records(environment, observation, 0, 10, 0,
                                              ["none", "single", "all"], 60., 10.))
            self.assertEqual(step.call_count, 5)
            self.assertEqual(environment.time, 0.)
            for original, vehicle in zip(original_positions, environment.road.vehicles):
                np.testing.assert_array_equal(original, vehicle.position)
            groups = defaultdict(list)
            for row in records:
                groups[tuple(row["requested_vehicle_ids"])].append(row)
                np.testing.assert_array_equal(row["state"], observation)
                self.assertTrue(row["truncated"])
                self.assertEqual(len(encode_messages(row)), 56)
                self.assertEqual(len(row["visible_vehicle_slots"]), 8)
            for rows in groups.values():
                self.assertEqual(Counter(row["ego_action"] for row in rows), Counter(range(5)))
                self.assertTrue(rows[0]["boundary_lane_command"])
            for action in range(5):
                copies = [row for row in records if row["ego_action"] == action]
                for row in copies[1:]:
                    np.testing.assert_array_equal(row["next_state"], copies[0]["next_state"])
                    self.assertEqual(row["reward"], copies[0]["reward"])
        finally:
            environment.close()

    def test_balancing_uses_only_training_strata(self):
        records = [{"ego_action": 0, "messages": None}] * 9 + [
            {"ego_action": 0, "messages": {"1": {"vehicle_id": 1}}},
            {"ego_action": 2, "messages": None},
            {"ego_action": 4, "messages": None},  # held out
        ]
        sampler = balanced_sampler(records, list(range(11)), "action-message", 7)
        mass = defaultdict(float)
        for index, weight in enumerate(sampler.weights):
            row = records[index]
            mass[(row["ego_action"], bool(row["messages"]))] += float(weight)
        self.assertEqual(len(mass), 3)
        for value in mass.values():
            self.assertAlmostEqual(value, 1.)
        self.assertEqual(len(sampler.weights), 11)
        self.assertIsNone(balanced_sampler(records, [0, 1], "none", 7))

    def test_action_evaluator_reports_collision_reward_bias(self):
        class ZeroModel(torch.nn.Module):
            def forward(self, inputs):
                return torch.zeros((len(inputs), 55)), torch.zeros((len(inputs), 1))
        records = [{"state": np.zeros((11, 5)), "next_state": np.ones((11, 5)),
                    "ego_action": 2, "messages": None, "reward": -2., "terminated": True}]
        totals = defaultdict(lambda: defaultdict(float))
        accumulate(ZeroModel(), records, totals)
        group = totals[(2, "no_messages", "crashed")]
        self.assertEqual(group["records"], 1)
        self.assertEqual(group["state_mae_sum"], 1)
        self.assertEqual(group["reward_mae_sum"], 2)
        self.assertEqual(group["reward_bias_sum"], 2)

    def test_counterfactuals_share_episode_holdout(self):
        a = {"dataset_source": "all_action_counterfactual", "episode": 5, "ego_action": 0}
        b = {**a, "ego_action": 2, "messages": {"7": {}}}
        self.assertEqual(episode_group(a), episode_group(b))
        groups = [episode_group(a), episode_group(b), ("other", 1)]
        self.assertEqual(heldout_episode_groups(groups), set(sorted(set(groups), key=str)[::5]))


if __name__ == "__main__":
    unittest.main()
