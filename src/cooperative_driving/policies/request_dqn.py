"""DQN network for choosing which visible vehicle to ask."""

import torch
import torch.nn as nn
import warnings


class RequestDQN(nn.Module):
    """Scores eight spatial request slots plus ask-nobody."""

    STATE_SIZE = 5 + 8 * 5
    ACTION_SIZE = 9

    def __init__(self, hidden_size=128):
        super().__init__()
        self.request_action_encoding = "spatial_v1"
        self.request_state_encoding = 'kinematics_v1'
        self.network = nn.Sequential(
            nn.Linear(self.STATE_SIZE, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.ACTION_SIZE),
        )

    def forward(self, state):
        return self.network(state)

    @classmethod
    def load(cls, path):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = cls()
        model.load_state_dict(checkpoint["model_state_dict"])
        model.request_action_encoding = checkpoint.get("request_action_encoding", "compressed_v0")
        model.request_state_encoding = checkpoint.get('request_state_encoding', 'kinematics_v1')
        model.mlpx_checkpoint = checkpoint.get('mlpx_checkpoint')
        model.training_config = checkpoint.get('training_config', {})
        if model.request_state_encoding not in ('kinematics_v1', 'ego_control_v1'):
            raise ValueError('Unknown request state encoding')
        if model.request_action_encoding not in ("spatial_v1", "compressed_v0"):
            raise ValueError(f"Unknown request action encoding: {model.request_action_encoding}")
        if model.request_action_encoding == "compressed_v0":
            warnings.warn(
                "Legacy DQN checkpoint: preserving compressed request-action semantics. "
                "Retrain DQN to use fixed spatial actions; the MLPx checkpoint can be reused.",
                UserWarning, stacklevel=2,
            )
        model.eval()
        return model
