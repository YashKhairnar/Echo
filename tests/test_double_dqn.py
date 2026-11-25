import unittest

import torch
from torch import nn

from scripts.training.train_request_dqn import td_loss
from src.cooperative_driving.policies import RequestDDQN, RequestDQN, load_request_policy


class FixedQ(nn.Module):
    def __init__(self, values):
        super().__init__()
        self.values = nn.Parameter(torch.tensor(values, dtype=torch.float32))

    def forward(self, states):
        return self.values.unsqueeze(0).expand(states.shape[0], -1)


class DoubleDQNTests(unittest.TestCase):
    def test_dqn_and_ddqn_are_separate_policy_classes(self):
        self.assertIsNot(RequestDQN, RequestDDQN)
        self.assertEqual(RequestDQN.POLICY_TYPE, "request_dqn")
        self.assertEqual(RequestDDQN.POLICY_TYPE, "request_ddqn")
        self.assertEqual(RequestDQN.TD_TARGET, "dqn")
        self.assertEqual(RequestDDQN.TD_TARGET, "double-dqn")

    def test_loader_infers_legacy_double_dqn_checkpoint(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy_ddqn.pt"
            torch.save({
                "model_state_dict": RequestDDQN().state_dict(),
                "td_target": "double-dqn",
                "request_action_encoding": "spatial_v1",
            }, path)
            self.assertIsInstance(load_request_policy(path), RequestDDQN)

    def test_double_dqn_selects_online_and_evaluates_target(self):
        online = FixedQ([5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        target = FixedQ([1.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        states = torch.zeros((1, 45))
        actions = torch.tensor([1])
        rewards = torch.tensor([0.0])
        dones = torch.tensor([0.0])
        valid = torch.ones((1, 9), dtype=torch.bool)

        dqn = td_loss(
            online, target, states, actions, rewards, states, dones, valid, 1.0,
            "dqn",
        )
        double = td_loss(
            online, target, states, actions, rewards, states, dones, valid, 1.0,
            "double-dqn",
        )

        self.assertGreater(float(dqn.detach()), float(double.detach()))
        self.assertAlmostEqual(float(double.detach()), 0.5)

    def test_double_dqn_respects_next_action_mask(self):
        online = FixedQ([5.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        target = FixedQ([9.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        states = torch.zeros((1, 45))
        valid = torch.tensor([[False, True, False, False, False, False, False, False, False]])
        loss = td_loss(
            online, target, states, torch.tensor([1]), torch.tensor([0.0]),
            states, torch.tensor([0.0]), valid, 1.0, "double-dqn",
        )
        self.assertAlmostEqual(float(loss.detach()), 1.5)


if __name__ == "__main__":
    unittest.main()
