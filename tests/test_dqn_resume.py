import csv
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from scripts.training import train_request_dqn as trainer


class TinyEnvironment:
    """One decision per episode to exercise real optimization and checkpoint IO."""
    config = {"duration": 1}
    action_space = SimpleNamespace(n=5)
    vehicle = SimpleNamespace(lane_index=("0", "1", 1), speed=25.)

    @property
    def observation_type(self):
        return self

    def observe(self):
        return np.zeros(55, dtype=np.float32)

    def reset(self, seed):
        self.reward = float(np.random.default_rng(seed).uniform())
        return np.zeros(55, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(55, dtype=np.float32), self.reward, False, True, {}

    def close(self):
        pass


class ResumeTests(unittest.TestCase):
    def test_split_training_matches_uninterrupted_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wm = root / "MLPx"
            wm.mkdir()
            (wm / "best.pt").write_bytes(b"unchanged test world model")

            def run(name, episodes, resume=False, mixed=False):
                output = root / name / "best.pt"
                command = ["train", "--experiment", "3", "--episodes", str(episodes),
                           "--batch-size", "1", "--seed", "5", "--save-every", "1",
                           "--output", str(output)]
                if resume:
                    command += ["--resume", str(output.with_name("latest.pt"))]
                if mixed:
                    command += ['--scenario-mix', 'mixed', '--ordinary-seed', '1000', '--challenge-seed', '2000']
                def intent_factory(seed, mode):
                    environment = TinyEnvironment()
                    environment.reset(seed)
                    return environment
                with patch("sys.argv", command), \
                     patch.object(trainer, "CustomEnv", TinyEnvironment), \
                     patch.object(trainer, 'intent_episode', side_effect=intent_factory), \
                     patch.object(trainer, "visible_neighbors", return_value={}), \
                     patch.object(trainer.MLPxWorldModel, "load", return_value=None), \
                     patch.object(trainer, "MCPPlanner", return_value=SimpleNamespace(plan=lambda *args: 1)), \
                     patch.object(trainer, "experiment_checkpoint_dir", side_effect=lambda exp, model: wm if model == "MLPx" else root / name), \
                     patch.object(trainer, "experiment_results_dir", return_value=root / name / "results"):
                    trainer.main()
                return torch.load(output.with_name("latest.pt"), weights_only=False)

            full = run("full", 4)
            run("split", 2)
            resumed = run("split", 2, resume=True)
            self.assertEqual(resumed["episodes"], 4)
            self.assertEqual(resumed["epsilon"], full["epsilon"])
            self.assertEqual(resumed["returns"], full["returns"])
            self.assertEqual(len(resumed["replay"]), 4)
            for key in ("model_state_dict", "target_state_dict"):
                for parameter in full[key]:
                    torch.testing.assert_close(full[key][parameter], resumed[key][parameter], rtol=0, atol=0)
            for state_id, state in full["optimizer_state_dict"]["state"].items():
                for key, value in state.items():
                    torch.testing.assert_close(value, resumed["optimizer_state_dict"]["state"][state_id][key], rtol=0, atol=0)
            logs = sorted((root / "split" / "results" / "experiments").glob("*/request_dqn_training.csv"))
            self.assertEqual(len(logs), 2)
            with logs[-1].open() as file:
                self.assertEqual([int(row["episode"]) for row in csv.DictReader(file)], [3, 4])

            mixed_full = run('mixed_full', 8, mixed=True)
            run('mixed_split', 4, mixed=True)
            mixed_resumed = run('mixed_split', 4, resume=True)
            self.assertEqual(mixed_full['returns'], mixed_resumed['returns'])
            self.assertEqual(mixed_resumed['training_config']['scenario_mix'], 'mixed')
            self.assertEqual(mixed_resumed['training_config']['ordinary_seed'], 1000)
            for key in ('model_state_dict', 'target_state_dict'):
                for parameter in mixed_full[key]:
                    torch.testing.assert_close(mixed_full[key][parameter], mixed_resumed[key][parameter], rtol=0, atol=0)

    def test_weights_only_checkpoint_keeps_episode_and_epsilon(self):
        online = trainer.RequestDQN()
        target = trainer.RequestDQN()
        optimizer = torch.optim.Adam(online.parameters())
        replay = trainer.deque(maxlen=10)
        checkpoint = {"model_state_dict": online.state_dict(), "episodes": 1000,
                      "request_action_encoding": "spatial_v1"}
        with self.assertWarnsRegex(UserWarning, "Weights-only"):
            completed, epsilon, returns = trainer.restore_training(
                checkpoint, online, target, optimizer, replay, random.Random(0)
            )
        self.assertEqual((completed, epsilon, returns), (1000, .05, []))
        self.assertEqual(len(replay), 0)
        with self.assertRaisesRegex(ValueError, "compressed-action"):
            trainer.restore_training({**checkpoint, "request_action_encoding": "compressed_v0"},
                                     online, target, optimizer, replay, random.Random(0))


if __name__ == "__main__":
    unittest.main()
