import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from data.combined_world_model import CombinedWorldModelDataset, episode_group
from scripts.data_collection.collect_intent_world_model_dataset import tree
from scripts.training.train_mlpx_wm import balanced_sampler
from scripts.evaluation.evaluate_mlpx_rollout import evaluate_state
from src.cooperative_driving.environments.customEnv import CustomEnv
from src.cooperative_driving.models.MLPx.mlpx import MLPxWorldModel


class FakeTreeEnvironment:
    def __init__(self):
        self.depth = 0
        self.observation_type = self
        self.vehicle = SimpleNamespace(target_speed=25., target_lane_index=('0','1',1))

    def observe(self):
        return np.full((11, 5), self.depth, dtype=np.float32)

    def step(self, action):
        self.depth += 1
        return self.observe(), float(action), False, self.depth == 3, {}

    def close(self):
        pass


class EgoControlWorldModelTests(unittest.TestCase):
    def test_rollout_evaluator_handles_extended_state_and_terminal_candidates(self):
        environment = CustomEnv(config={'duration': 1})
        try:
            model = MLPxWorldModel(state_size=57)
            metrics = evaluate_state(model, environment, environment.observation_type.observe(),
                                     None, 2, 'constant')
            self.assertTrue(np.isfinite(metrics['state_mae']))
        finally:
            environment.close()

    def test_stage_balancing_preserves_early_decision_mass(self):
        records = [{'ego_action': 1, 'messages': True, 'dataset_source': 'matched_intent', 'depth': 0}]
        records += [{'ego_action': 1, 'messages': True, 'dataset_source': 'matched_intent', 'depth': 2}] * 20
        sampler = balanced_sampler(records, list(range(len(records))), 'action-message-stage', 0)
        self.assertAlmostEqual(float(sampler.weights[0]), float(sampler.weights[1:].sum()))

    def test_control_memory_is_encoded_and_legacy_input_stays_unchanged(self):
        environment = CustomEnv()
        try:
            old, new = MLPxWorldModel(), MLPxWorldModel(state_size=57)
            before55, before57 = old.observe(environment), new.observe(environment)
            environment.vehicle.target_speed += 5.
            np.testing.assert_array_equal(old.observe(environment), before55)
            self.assertNotEqual(new.observe(environment)[-2], before57[-2])
            self.assertEqual(new.observe(environment).size, 57)
            self.assertEqual(new.INPUT_SIZE, 118)
        finally:
            environment.close()

    def test_versioned_and_legacy_checkpoint_load(self):
        with tempfile.TemporaryDirectory() as directory:
            for size in [55,57]:
                model = MLPxWorldModel(state_size=size)
                checkpoint = {'model_state_dict': model.state_dict()}
                if size == 57:
                    checkpoint['state_size'] = 57
                path = Path(directory) / f'{size}.pt'
                torch.save(checkpoint, path)
                loaded = MLPxWorldModel.load(path)
                state, reward = loaded.predict_step(np.zeros(size), 1)
                self.assertEqual(state.numel(), size)
                self.assertTrue(np.isfinite(reward))

    def test_complete_tree_and_dataset_keep_one_source_group(self):
        records = []
        tree(FakeTreeEnvironment(), 0, 'matched_intent', 7, 500007,
             [None] * 8, {'1': {'vehicle_id': 1}}, 'stay', records)
        self.assertEqual(len(records), (5+25+125)*2)
        rows = [json.loads(line) for line in records]
        self.assertEqual({episode_group(row) for row in rows}, {('matched_intent',7)})
        self.assertEqual(len(rows[0]['state']), 57)
        # Use no-message rows: nonempty messages require a real visible slot.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'dataset.jsonl'
            path.write_text(''.join(records[::2]))
            dataset = CombinedWorldModelDataset(path, compact_records=True)
            self.assertEqual(dataset.inputs.shape[1], 118)
            self.assertEqual(dataset.next_states.shape[1], 57)


if __name__ == '__main__':
    unittest.main()
