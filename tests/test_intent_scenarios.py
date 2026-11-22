import json
import unittest

import numpy as np

from scripts.data_collection.enumerate_message_requests import jsonable
from scripts.data_collection.show_visible_neighbors import visible_neighbors
from scripts.evaluation.evaluate_intent_scenarios import evaluate_pair
from scripts.training.train_request_dqn import request_state
from src.cooperative_driving.environments.intent_scenarios import matched_intent_pair


class IntentScenarioTests(unittest.TestCase):
    def test_hidden_intent_changes_dynamics_without_changing_public_input(self):
        environments, metadata = matched_intent_pair(400000)
        try:
            stay, merge = environments['stay'], environments['merge']
            np.testing.assert_array_equal(stay.observation_type.observe(), merge.observation_type.observe())
            states = [request_state(env, visible_neighbors(env, 60., 10.)) for env in [stay, merge]]
            np.testing.assert_array_equal(*states)
            messages = []
            for env in [stay, merge]:
                before = json.dumps(jsonable(env.get_all_vehicle_states()), sort_keys=True)
                messages.append(env.request_neighbor_action(metadata['neighbor_id']))
                self.assertEqual(before, json.dumps(jsonable(env.get_all_vehicle_states()), sort_keys=True))
                self.assertIn(metadata['neighbor_id'], [c['vehicle'].vehicle_id for c in visible_neighbors(env,60.,10.).values()])
                env.step(1)
            self.assertNotEqual(messages[0]['target_lane'], messages[1]['target_lane'])
            stay_y = stay.get_vehicle(metadata['neighbor_id']).position[1]
            merge_y = merge.get_vehicle(metadata['neighbor_id']).position[1]
            ego_y = stay.vehicle.position[1]
            self.assertLess(abs(merge_y - ego_y), abs(stay_y - ego_y))
        finally:
            for env in environments.values():
                env.close()

    def test_information_oracles_respect_shared_blind_action_and_request_cost(self):
        result = evaluate_pair(400000, 3, .05)
        self.assertGreaterEqual(result['gross_information_value'], -1e-10)
        modes = result['policies']
        self.assertEqual(modes['stay']['blind_oracle']['first_action'], modes['merge']['blind_oracle']['first_action'])
        self.assertEqual(modes['stay']['selective_oracle']['requests'], modes['merge']['selective_oracle']['requests'])
        for mode in modes.values():
            self.assertAlmostEqual(mode['informed_rule']['return'], mode['informed_rule']['reward'] - .05)
            self.assertEqual(mode['geometry']['requests'], 0)
        informed = sum(modes[m]['informed_oracle']['return'] for m in modes) / 2
        blind = sum(modes[m]['blind_oracle']['return'] for m in modes) / 2
        self.assertAlmostEqual(informed - blind, result['net_information_value'])

    def test_slot_eligibility_for_close_neighbor(self):
        environments, metadata = matched_intent_pair(300012)
        try:
            for env in environments.values():
                visible = visible_neighbors(env, 60., 10.)
                self.assertLessEqual(len(visible), 8)
                self.assertIn(metadata['neighbor_id'], [c['vehicle'].vehicle_id for c in visible.values()])
        finally:
            for env in environments.values():
                env.close()


if __name__ == '__main__':
    unittest.main()
