import random
import unittest

import torch

from scripts.training.train_request_dqn import ASK_NOBODY, select_action


class FixedScores(torch.nn.Module):
    """A stand-in request network that returns fixed scores."""

    def __init__(self, scores):
        super().__init__()
        self.scores = torch.tensor(scores, dtype=torch.float32)

    def forward(self, states):
        return self.scores.unsqueeze(0).expand(states.shape[0], -1)


def scores(ask_slot, ask_score, nobody_score):
    values = [-5.0] * 9
    values[ask_slot] = ask_score
    values[ASK_NOBODY] = nobody_score
    return values


class AskMarginTests(unittest.TestCase):
    def choose(self, model, valid, margin):
        return select_action(model, [0.0] * 45, valid, epsilon=0.0, rng=random.Random(0),
                             ask_margin=margin)

    def test_zero_margin_keeps_greedy_choice(self):
        model = FixedScores(scores(2, 1.0, 0.9))
        self.assertEqual(self.choose(model, [2, ASK_NOBODY], 0.0), 2)

    def test_small_lead_falls_back_to_ask_nobody(self):
        model = FixedScores(scores(2, 1.0, 0.9))
        self.assertEqual(self.choose(model, [2, ASK_NOBODY], 0.5), ASK_NOBODY)

    def test_large_lead_still_asks(self):
        model = FixedScores(scores(2, 3.0, 0.9))
        self.assertEqual(self.choose(model, [2, ASK_NOBODY], 0.5), 2)

    def test_margin_does_not_change_ask_nobody_choice(self):
        model = FixedScores(scores(2, 0.0, 1.0))
        self.assertEqual(self.choose(model, [2, ASK_NOBODY], 0.5), ASK_NOBODY)

    def test_invalid_slots_are_ignored(self):
        model = FixedScores(scores(2, 3.0, 0.9))
        self.assertEqual(self.choose(model, [ASK_NOBODY], 0.5), ASK_NOBODY)


class DaggerCollectionTests(unittest.TestCase):
    def test_driver_choice_drives_and_oracle_choice_labels(self):
        import argparse

        from scripts.data_collection.collect_oracle_requests import oracle_episode
        from src.cooperative_driving.environments.intent_scenarios import intent_episode
        from src.cooperative_driving.models.MLPx.mlpx import MLPxWorldModel

        torch.manual_seed(0)
        args = argparse.Namespace(planning_horizon=1, visibility=60.0, side_window=10.0,
                                  request_state_encoding='ego_control_v1', lookahead=1,
                                  communication_cost=0.05, oracle_drive_probability=0.0,
                                  drive_ask_margin=0.0)
        controls = dict(safety_guard=True, lane_change_guard=True, max_target_speed=30.0,
                        min_clearance=8.0, min_ttc=3.0, batched=True)
        driver = FixedScores(scores(0, -9.0, 9.0))  # always asks nobody
        environment = intent_episode(1300005, 'merge')
        records = oracle_episode(environment, MLPxWorldModel(state_size=57), args, controls,
                                 'intent', 1300005, 'merge', driver, random.Random(0))
        environment.close()
        self.assertTrue(records)
        for record in records:
            self.assertEqual(record['driven'], ASK_NOBODY)
            self.assertEqual(record['choice'], int(record['target'].argmax())
                             if record['target'][ASK_NOBODY] < 1 else ASK_NOBODY)


if __name__ == '__main__':
    unittest.main()
