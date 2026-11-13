import torch
import torch.nn as nn


class MLPxWorldModel(nn.Module):
    """World model that accepts information from multiple requested cars.

    Input = observation (60) + driving action (5) + request mask (10)
    + ten message slots (4 values each) = 115 values.
    """

    OBSERVATION_SIZE = 60
    ACTION_SIZE = 5
    REQUEST_MASK_SIZE = 10
    MESSAGE_FEATURES = 4
    MESSAGE_SIZE = REQUEST_MASK_SIZE * MESSAGE_FEATURES
    INPUT_SIZE = OBSERVATION_SIZE + ACTION_SIZE + REQUEST_MASK_SIZE + MESSAGE_SIZE
    NEXT_STATE_SIZE = 60

    def __init__(self):
        super().__init__()
        # The shared trunk encodes ego state, driving action, request mask,
        # and all padded requested-car messages.
        self.shared = nn.Sequential(
            nn.Linear(self.INPUT_SIZE, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.next_state_head = nn.Linear(128, self.NEXT_STATE_SIZE)
        self.reward_head = nn.Linear(128, 1)

    def forward(self, inputs):
        # Two heads predict the next observation and the reward separately.
        hidden = self.shared(inputs)
        return self.next_state_head(hidden), self.reward_head(hidden)

    def predict_step(self, observation, action, request_mask=None, messages=None):
        """Predict one step for a multi-request message.

        ``request_mask`` may also be a dictionary containing ``request_mask``
        and ``messages`` so the existing MCP planner can pass one message
        object through its API.
        """
        if isinstance(request_mask, dict):
            messages = request_mask.get("messages", messages)
            request_mask = request_mask.get("request_mask")

        observation = torch.as_tensor(observation, dtype=torch.float32).reshape(-1)
        action_one_hot = torch.zeros(self.ACTION_SIZE, dtype=torch.float32)
        action_one_hot[int(action)] = 1.0
        if request_mask is None:
            request_mask = torch.zeros(self.REQUEST_MASK_SIZE, dtype=torch.float32)
        else:
            request_mask = torch.as_tensor(request_mask, dtype=torch.float32).reshape(-1)
        if messages is None:
            messages = torch.zeros(self.MESSAGE_SIZE, dtype=torch.float32)
        else:
            messages = torch.as_tensor(messages, dtype=torch.float32).reshape(-1)
        inputs = torch.cat(
            (observation, action_one_hot, request_mask, messages)
        ).unsqueeze(0)
        next_observation, reward = self(inputs)
        return next_observation.squeeze(0), reward.squeeze(0).squeeze(-1)
