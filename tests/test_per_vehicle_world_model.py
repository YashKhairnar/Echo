import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from data.combined_world_model import CombinedWorldModelDataset
from scripts.data_collection.show_visible_neighbors import SLOT_ORDER, visible_neighbors
from scripts.training.train_mlpx_wm import compute_losses, training_loss
from src.cooperative_driving.environments.intent_scenarios import matched_intent_pair
from src.cooperative_driving.models.MLPx.mlpx import MLPxWorldModel
from src.cooperative_driving.models.MLPx.per_vehicle import (
    PerVehicleMLPx, load_world_model, match_messages, row_slots, X_SCALE, Y_SCALE)
from src.cooperative_driving.planner.MCP import MCPPlanner


def car_row(dx, dy, present=1.0):
    return [present, dx / X_SCALE, dy / Y_SCALE, 0.0, 0.0]


def message(lane=1):
    return {'behavior_type': 'aggressive', 'target_lane': ('0', '1', lane),
            'target_speed': 25., 'acceleration': 0.5, 'steering': 0.1, 'crashed': False}


class RowSlotTests(unittest.TestCase):
    def test_rows_get_the_visible_neighbors_slots(self):
        rows = torch.tensor([[car_row(20, -4), car_row(3, -4), car_row(-20, -4),
                              car_row(15, 0), car_row(-15, 0),
                              car_row(20, 4), car_row(-5, 4), car_row(-20, 4),
                              car_row(20, 8), car_row(80, 0)]])
        slots, _ = row_slots(rows)
        # Rows 8 (two lanes away) and 9 (beyond 60 m) are not visible.
        self.assertEqual(slots[0].tolist(), [0, 1, 2, 3, 4, 5, 6, 7, -1, -1])

    def test_closest_row_in_a_slot_gets_the_message(self):
        rows = torch.tensor([[car_row(40, 0), car_row(15, 0)] + [car_row(0, 0, 0.0)] * 8])
        messages = torch.zeros(1, 8, 7)
        messages[0, 3, 0] = 1.0   # ego-lane front
        messages[0, 0, 0] = 1.0   # left front: no row there
        assignment, unmatched = match_messages(rows, messages)
        self.assertEqual(assignment[0, 3].tolist(), [0., 1.] + [0.] * 8)
        self.assertEqual(unmatched[0].tolist(), [1.] + [0.] * 7)

    def test_geometry_matches_visible_neighbors_in_challenge_scenes(self):
        for seed in range(400000, 400005):
            environments, metadata = matched_intent_pair(seed, 3)
            try:
                environment = environments['merge']
                visible = visible_neighbors(environment, 60., 10.)
                rows = torch.as_tensor(environment.observation_type.observe()[1:]).unsqueeze(0)
                slots, _ = row_slots(rows.float())
                self.assertEqual(sorted(s for s in slots[0].tolist() if s >= 0),
                                 sorted(SLOT_ORDER.index(slot) for slot in visible))
            finally:
                for environment in environments.values():
                    environment.close()


class PerVehicleModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = PerVehicleMLPx(hidden_size=32).eval()

    def state_with_neighbor(self):
        rows = [[1.0, 0.1, 0.25, 0.3, 0.0], car_row(12, -4)] + [[0.0] * 5] * 9
        return np.concatenate([np.asarray(rows, dtype=np.float32).reshape(-1), [25 / 30, 1 / 3]]).astype(np.float32)

    def test_same_interface_as_mlpx(self):
        state = self.state_with_neighbor()
        next_state, reward = self.model.predict_step(state, 1)
        self.assertEqual(next_state.shape, (57,))
        self.assertTrue(np.isfinite(reward))
        states, rewards = self.model.predict_batch(np.stack([state] * 5), list(range(5)))
        self.assertEqual(tuple(states.shape), (5, 57))
        self.assertEqual(tuple(rewards.shape), (5,))
        crash = self.model.predict_crash(np.stack([state] * 5), list(range(5)))
        self.assertTrue(torch.all((crash > 0) & (crash < 1)))
        values = MCPPlanner(self.model).action_values_batched(state)
        self.assertEqual(len(values), 5)

    def test_message_reaches_its_own_car_row(self):
        state = self.state_with_neighbor()
        bundle_stay = {'messages_by_slot': [message(0)] + [None] * 7}
        bundle_merge = {'messages_by_slot': [message(1)] + [None] * 7}
        stay, _ = self.model.predict_step(state, 1, bundle_stay)
        merge, _ = self.model.predict_step(state, 1, bundle_merge)
        none, _ = self.model.predict_step(state, 1)
        self.assertFalse(torch.allclose(stay, merge))
        self.assertFalse(torch.allclose(stay, none))

    def test_empty_scene_is_finite(self):
        outputs = self.model.forward_all(torch.zeros(3, self.model.INPUT_SIZE))
        for output in outputs:
            self.assertTrue(torch.isfinite(output).all())

    def test_checkpoint_selects_architecture(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'per_vehicle.pt'
            torch.save({'model_state_dict': self.model.state_dict(), 'state_size': 57,
                        'architecture': 'per_vehicle', 'hidden_size': 32}, path)
            loaded = load_world_model(path)
            self.assertIsInstance(loaded, PerVehicleMLPx)
            state = self.state_with_neighbor()
            torch.testing.assert_close(loaded.predict_step(state, 2)[0], self.model.predict_step(state, 2)[0])
            flat_path = Path(directory) / 'flat.pt'
            torch.save({'model_state_dict': MLPxWorldModel(state_size=57).state_dict(), 'state_size': 57}, flat_path)
            self.assertIs(type(load_world_model(flat_path)), MLPxWorldModel)

    def test_training_losses_include_crash_for_per_vehicle_only(self):
        batch = (torch.zeros(4, 118), torch.zeros(4, 57), torch.zeros(4, 1), torch.ones(4, 1))
        losses = compute_losses(self.model, batch)
        self.assertIn('crash_loss', losses)
        training_loss(losses, 0.1).backward()
        self.assertNotIn('crash_loss', compute_losses(MLPxWorldModel(state_size=57), batch))


class DatasetCrashLabelTests(unittest.TestCase):
    def test_dataset_returns_terminated_as_crash(self):
        rows = [{'state': [0.0] * 57, 'next_state': [0.0] * 57, 'ego_action': 1, 'reward': 0.1,
                 'terminated': terminated, 'state_encoding': 'kinematics_ego_control_v1',
                 'messages': {}, 'episode': 0} for terminated in (False, True)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'dataset.jsonl'
            path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            dataset = CombinedWorldModelDataset(path, compact_records=True)
            self.assertEqual(dataset[0][3].item(), 0.0)
            self.assertEqual(dataset[1][3].item(), 1.0)


if __name__ == '__main__':
    unittest.main()
