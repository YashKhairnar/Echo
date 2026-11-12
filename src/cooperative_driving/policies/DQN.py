import torch
import torch.nn as nn


class DQNPolicy(nn.Module):
    def __init__(self, observation_size=90, action_size=11):
        super().__init__()
        self.q_network = nn.Sequential(
            nn.Linear(observation_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_size), # 10 vehicles + None
        )

    def forward(self, observation):
        observation = torch.as_tensor(observation, dtype=torch.float32)
        if observation.ndim <= 2:
            # A single observation can be shaped (10, 9) or already be flat.
            observation = observation.reshape(1, -1)
        else:
            # A batch of observations is shaped (batch_size, 10, 9).
            observation = observation.reshape(observation.size(0), -1)
        return self.q_network(observation)

    # which visible vehicle to request information from
    def act(self, observation, valid_actions=None):
        with torch.no_grad():
            q_values = self(observation)

        if valid_actions is not None:
            masked_q_values = torch.full_like(q_values, float("-inf"))
            masked_q_values[:, valid_actions] = q_values[:, valid_actions]
            q_values = masked_q_values

        return int(q_values.argmax(dim=1).item())


# Keep the old name working for existing imports.
DQNpolicy = DQNPolicy

