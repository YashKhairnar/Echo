# A simple multilayer perceptron model for 
# predicting future state and reward
# This model is used for learning the dynamics of the environment
# and the reward function
# It is used for model-based reinforcement learning


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