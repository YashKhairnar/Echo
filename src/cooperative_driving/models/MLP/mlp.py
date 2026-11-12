# A simple multilayer perceptron model for 
# predicting future state and reward
# This model is used for learning the dynamics of the environment
# and the reward function
# It is used for model-based reinforcement learning


import torch
import torch.nn as nn

class MLP_worldModel(nn.Module):
    """
    Input:
    Observation: 10 × 9 = 90 values
    Action:      5 discrete actions → one-hot vector of size 5
    Message:     10 values

    Output:
    Next State: 90 values
    Reward: 1 value
    """
    def __init__(self):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(105, 256),
            nn.ReLU(),
            nn.Linear(256,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU()
        )

        self.next_state_head = nn.Linear(128, 90)
        self.reward_head = nn.Linear(128, 1)   

    def forward(self, inputs):
        hidden = self.shared(inputs)
        next_state = self.next_state_head(hidden)
        reward = self.reward_head(hidden)
        return next_state, reward

    def predict_step(self, observation, action, message=None):
        """Predict one next observation and reward for a candidate action.

        This is used by the planner during evaluation. It does not add
        predicted transitions to the DQN replay buffer.
        """
        observation = torch.as_tensor(observation, dtype=torch.float32).reshape(-1)

        action = int(action)
        action_one_hot = torch.zeros(5, dtype=torch.float32)
        action_one_hot[action] = 1.0

        if message is None:
            message = torch.zeros(10, dtype=torch.float32)
        else:
            message = torch.as_tensor(message, dtype=torch.float32).reshape(-1)

        inputs = torch.cat((observation, action_one_hot, message)).unsqueeze(0)
        next_observation, reward = self(inputs)
        return next_observation.squeeze(0), reward.squeeze(0).squeeze(-1)

    def make_planner(self, action_size=5, planning_horizon=5, discount_factor=0.99):
        """Create an MCP planner that uses this world model for evaluation."""
        from ...planner.MCP import MCPPlanner

        return MCPPlanner(
            world_model=self,
            action_size=action_size,
            planning_horizon=planning_horizon,
            discount_factor=discount_factor,
        )
