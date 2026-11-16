"""Training data for MLPx, read from the CSV written by collect_multi_request_data.py."""

import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

NUM_ACTIONS = 5


class MultiRequestWorldModelDataset(Dataset):
    """One item per recorded step:

    inputs       115 numbers: observation, one-hot action, request mask, messages
    next_state   the observation after the step
    reward       the step's driving reward
    crash        1 if this step ended in a crash
    crash_soon   1 if the episode crashes within ``crash_horizon`` steps
                 (this step included)
    """

    def __init__(self, csv_path, crash_horizon=3):
        table = pd.read_csv(csv_path)

        inputs, next_states = [], []
        for observation, action, message, next_observation in zip(
            table["observation_t"], table["action_t"], table["message_t"], table["observation_t+1"]
        ):
            one_hot = np.zeros(NUM_ACTIONS, dtype=np.float32)
            one_hot[int(action)] = 1.0
            message = json.loads(message)
            inputs.append(np.concatenate([
                np.asarray(json.loads(observation), dtype=np.float32).reshape(-1),
                one_hot,
                np.asarray(message["request_mask"], dtype=np.float32),
                np.asarray(message["messages"], dtype=np.float32).reshape(-1),
            ]))
            next_states.append(np.asarray(json.loads(next_observation), dtype=np.float32).reshape(-1))

        crashes = table["terminated"].astype(bool).to_numpy()
        self.inputs = torch.tensor(np.asarray(inputs))
        self.next_states = torch.tensor(np.asarray(next_states))
        self.rewards = torch.tensor(table["driving_reward_t+1"].to_numpy(), dtype=torch.float32)[:, None]
        self.crashes = torch.tensor(crashes, dtype=torch.float32)[:, None]
        self.crash_soon = torch.tensor(
            crash_soon_labels(table["episode_id"], table["timestep"], crashes, crash_horizon),
            dtype=torch.float32,
        )[:, None]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return (
            self.inputs[index],
            self.next_states[index],
            self.rewards[index],
            self.crashes[index],
            self.crash_soon[index],
        )


def crash_soon_labels(episode_ids, timesteps, crashes, horizon):
    """1 for the last ``horizon`` steps of every episode that ends in a crash."""
    crash_step = {episode: step for episode, step, crashed in zip(episode_ids, timesteps, crashes) if crashed}
    return [
        float(episode in crash_step and 0 <= crash_step[episode] - step < horizon)
        for episode, step in zip(episode_ids, timesteps)
    ]
