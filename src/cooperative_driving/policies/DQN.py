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
        # Flatten the 10x9 kinematics observation before estimating Q-values.
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
        # Mask cars that are not visible so the policy cannot request them.
        with torch.no_grad():
            q_values = self(observation)

        if valid_actions is not None:
            masked_q_values = torch.full_like(q_values, float("-inf"))
            masked_q_values[:, valid_actions] = q_values[:, valid_actions]
            q_values = masked_q_values

        return int(q_values.argmax(dim=1).item())


class MultiRequestDQNPolicy(nn.Module):
    """DQN policy with an independent request/no-request decision per slot.

    The output has shape ``(batch, 10, 2)``. For each visible-neighbor slot,
    index 0 means do not request and index 1 means request. Invalid slots are
    forced to the no-request choice by ``act``.
    """

    MAX_REQUEST_SLOTS = 10

    def __init__(self, observation_size=90):
        super().__init__()
        self.q_network = nn.Sequential(
            nn.Linear(observation_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.MAX_REQUEST_SLOTS * 2),
        )

    def forward(self, observation):
        # Each visible slot gets two Q-values: no-request and request.
        observation = torch.as_tensor(observation, dtype=torch.float32)
        if observation.ndim <= 2:
            observation = observation.reshape(1, -1)
        else:
            observation = observation.reshape(observation.size(0), -1)
        return self.q_network(observation).reshape(
            observation.shape[0],
            self.MAX_REQUEST_SLOTS,
            2,
        )

    def act(self, observation, visible_slots):
        """Return the visible slots whose request Q-value beats no-request."""
        # Return local slots; the caller maps them to actual vehicle IDs.
        with torch.no_grad():
            q_values = self(observation)[0]

        requested_slots = []
        for slot in visible_slots:
            if q_values[slot, 1] > q_values[slot, 0]:
                requested_slots.append(int(slot))
        return requested_slots


# Keep the old name working for existing imports.
DQNpolicy = DQNPolicy

