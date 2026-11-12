import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class WorldModelDataset(Dataset):
    """Dataset for training a world model from collected transitions."""

    NUM_ACTIONS = 5

    def __init__(self, csv_path):
        dataframe = pd.read_csv(csv_path)

        self.inputs = []
        self.next_states = []
        self.rewards = []
        self.episode_ids = []

        for _, row in dataframe.iterrows():
            observation = self._observation_vector(row["observation_t"])
            action = self._action_vector(row["action_t"])
            message = self._message_vector(row["message_t"])
            next_state = self._observation_vector(row["observation_t+1"])

            self.inputs.append(np.concatenate([observation, action, message]))
            self.next_states.append(next_state)
            self.rewards.append(float(row["reward_t+1"]))
            self.episode_ids.append(int(row["episode_id"]))

        self.inputs = torch.tensor(np.asarray(self.inputs), dtype=torch.float32)
        self.next_states = torch.tensor(
            np.asarray(self.next_states), dtype=torch.float32
        )
        self.rewards = torch.tensor(
            np.asarray(self.rewards).reshape(-1, 1), dtype=torch.float32
        )
        self.episode_ids = torch.tensor(self.episode_ids, dtype=torch.long)

    @staticmethod
    def _observation_vector(value):
        """Parse and flatten one Kinematics observation from the CSV."""
        observation = json.loads(value) if isinstance(value, str) else value
        return np.asarray(observation, dtype=np.float32).reshape(-1)

    @classmethod
    def _action_vector(cls, value):
        """Convert a discrete action into a one-hot vector."""
        action = json.loads(value) if isinstance(value, str) else value
        action = int(action)

        if not 0 <= action < cls.NUM_ACTIONS:
            raise ValueError(f"Action {action} is outside [0, {cls.NUM_ACTIONS - 1}]")

        one_hot = np.zeros(cls.NUM_ACTIONS, dtype=np.float32)
        one_hot[action] = 1.0
        return one_hot

    @staticmethod
    def _message_vector(value):
        """Convert the shared vehicle message into ten numeric values."""
        message = json.loads(value) if isinstance(value, str) else value

        # pandas reads empty/None CSV cells as NaN floats.
        if message is None or (
            isinstance(message, float) and np.isnan(message)
        ):
            return np.zeros(10, dtype=np.float32)

        return np.asarray(
            [
                *message["position"],
                *message["velocity"],
                message["action"]["acceleration"],
                message["action"]["steering"],
                message["intent"]["intent_acceleration"],
                message["intent"]["intent_steering"],
                message["intent"]["intent_lane_change"],
                message["vehicle_id"],
            ],
            dtype=np.float32,
        )

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return (
            self.inputs[index],
            self.next_states[index],
            self.rewards[index],
        )


if __name__ == "__main__":
    dataset = WorldModelDataset("data/randomPolicyData.csv")
    print("Samples:", len(dataset))
    print("Input shape:", dataset[0][0].shape)
    print("Next-state shape:", dataset[0][1].shape)
    print("Reward shape:", dataset[0][2].shape)
