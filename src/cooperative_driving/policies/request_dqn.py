"""DQN network for choosing which visible vehicle to ask."""

import torch
import torch.nn as nn


class RequestDQN(nn.Module):
    """Scores eight spatial request slots plus ask-nobody."""

    STATE_SIZE = 5 + 8 * 5
    ACTION_SIZE = 9

    def __init__(self, hidden_size=128):
        super().__init__()
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
        model.eval()
        return model
