"""Simple MLP world model for the combined driving dataset.

Input: current observation, ego action, and requested vehicle messages.
Output: next observation and reward.

Legacy observations contain 55 kinematic values (116 total inputs).
Versioned ego-control states append target speed and lane: 57 state values
and 118 inputs. Eight message slots contain seven features each. Checkpoints
record the state size; old checkpoints retain their original representation.
"""

from __future__ import annotations
from collections.abc import Mapping
import torch
import torch.nn as nn


class MLPxWorldModel(nn.Module):
    """A small baseline world model with shared hidden layers."""

    STATE_SIZE = 11 * 5
    ACTION_SIZE = 5
    MAX_MESSAGE_SLOTS = 8
    MESSAGE_FEATURES = 7
    MESSAGE_SIZE = MAX_MESSAGE_SLOTS * MESSAGE_FEATURES
    INPUT_SIZE = STATE_SIZE + ACTION_SIZE + MESSAGE_SIZE

    def __init__(self, hidden_size=256, state_size=55):
        super().__init__()
        if state_size not in (55, 57):
            raise ValueError('Supported state sizes are 55 (legacy) and 57 (ego controls)')
        self.STATE_SIZE = state_size
        self.INPUT_SIZE = state_size + self.ACTION_SIZE + self.MESSAGE_SIZE
        self.state_encoding = 'kinematics_ego_control_v1' if state_size == 57 else 'kinematics_v1'
        self.encoder = nn.Sequential(
            nn.Linear(self.INPUT_SIZE, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
        )
        self.next_state_head = nn.Linear(128, self.STATE_SIZE)
        self.reward_head = nn.Linear(128, 1)

    def forward(self, inputs):
        hidden = self.encoder(inputs)
        return (
            self.next_state_head(hidden),
            self.reward_head(hidden),
        )

    @classmethod
    def load(cls, path):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = cls(state_size=checkpoint.get('state_size', 55))
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model

    def observe(self, environment):
        """Encode real state with the checkpoint's versioned representation."""
        import numpy as np
        state = np.asarray(environment.observation_type.observe(), dtype=np.float32).reshape(-1)
        if self.STATE_SIZE == 57:
            state = np.concatenate([state, [environment.vehicle.target_speed / 30.,
                                           environment.vehicle.target_lane_index[2] / 3.]]).astype(np.float32)
        return state

    @staticmethod
    def _action_one_hot(action):
        result = torch.zeros(MLPxWorldModel.ACTION_SIZE, dtype=torch.float32)
        result[int(action)] = 1.0
        return result

    @staticmethod
    def _encode_message(message):
        behavior_code = {
            "aggressive": 1.0,
            "defensive": -1.0,
            "idm": 0.0,
        }.get(message.get("behavior_type", "unknown"), 0.0)
        target_lane = message.get("target_lane")
        target_lane_id = float(target_lane[2]) / 3.0 if target_lane is not None else 0.0
        return [
            1.0,
            behavior_code,
            target_lane_id,
            float(message.get("target_speed", 0.0)) / 30.0,
            float(message.get("acceleration", 0.0)) / 6.0,
            float(message.get("steering", 0.0)),
            float(bool(message.get("crashed", False))),
        ]

    @classmethod
    def _message_vector(cls, message):
        """Convert a message bundle or vehicle-message mapping to 56 values."""
        if message is None:
            return torch.zeros(cls.MESSAGE_SIZE, dtype=torch.float32)

        if "message_vector" in message:
            values = torch.as_tensor(message["message_vector"], dtype=torch.float32).reshape(-1)
            if values.numel() != cls.MESSAGE_SIZE:
                raise ValueError(f"message_vector must contain {cls.MESSAGE_SIZE} values")
            return values

        if "messages_by_slot" in message:
            values = []
            for item in message["messages_by_slot"][: cls.MAX_MESSAGE_SLOTS]:
                values.extend(cls._encode_message(item) if item else [0.0] * cls.MESSAGE_FEATURES)
            values.extend([0.0] * (cls.MESSAGE_SIZE - len(values)))
            return torch.tensor(values, dtype=torch.float32)

        if "messages" in message:
            values = torch.as_tensor(message["messages"], dtype=torch.float32).reshape(-1)
            if values.numel() == cls.MESSAGE_SIZE:
                return values

        values = []
        for vehicle_id in sorted(message, key=lambda value: int(value)):
            item = message[vehicle_id]
            if isinstance(item, Mapping):
                values.extend(cls._encode_message(item))
            if len(values) >= cls.MESSAGE_SIZE:
                break
        values = values[: cls.MESSAGE_SIZE]
        values.extend([0.0] * (cls.MESSAGE_SIZE - len(values)))
        return torch.tensor(values, dtype=torch.float32)

    @torch.no_grad()
    def predict_step(self, observation, action, message=None):
        observation = torch.as_tensor(observation, dtype=torch.float32).reshape(-1)
        if observation.numel() != self.STATE_SIZE:
            raise ValueError(f"observation must contain {self.STATE_SIZE} values")
        inputs = torch.cat((observation, self._action_one_hot(action), self._message_vector(message)))
        next_state, reward = self(inputs.unsqueeze(0))
        return (
            next_state.squeeze(0),
            float(reward.squeeze()),
        )

    @torch.no_grad()
    def predict_batch(self, observations, actions, message=None):
        observations = torch.as_tensor(observations, dtype=torch.float32).reshape(-1, self.STATE_SIZE)
        actions = torch.as_tensor(actions, dtype=torch.long).reshape(-1)
        if observations.shape[0] != actions.shape[0]:
            raise ValueError("observations and actions must have the same batch size")
        action_one_hot = torch.zeros(observations.shape[0], self.ACTION_SIZE)
        action_one_hot[torch.arange(observations.shape[0]), actions] = 1.0
        message_vector = self._message_vector(message).expand(observations.shape[0], -1)
        next_state, reward = self(
            torch.cat((observations, action_one_hot, message_vector), dim=1)
        )
        return (
            next_state,
            reward.squeeze(1),
        )
