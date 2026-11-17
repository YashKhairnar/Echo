"""MLPx: the world model ("the car's imagination").

Input (115 numbers):
    observation       60  = 10 rows x [presence, x, y, vx, vy, vehicle_id]
    driving action     5  one-hot
    request mask      10  1 = that slot's car was asked
    messages          40  = 10 slots x 4 message values (zeros if not asked)

Outputs:
    next observation      60 numbers
    reward                driving reward of this step, assuming no crash
    crash logit           will this step end in a crash?
    crash-soon logit      will the episode crash within the next 3 steps?

How it works ("per-vehicle"): each neighbour car is encoded *together with
its own message* by one shared small network. The car encodings are pooled
into a scene summary, and the outputs are predicted from that summary.
"""

import torch
import torch.nn as nn


class MLPxWorldModel(nn.Module):
    OBSERVATION_SIZE = 60
    ACTION_SIZE = 5
    REQUEST_MASK_SIZE = 10
    MESSAGE_FEATURES = 4
    MESSAGE_SIZE = REQUEST_MASK_SIZE * MESSAGE_FEATURES
    INPUT_SIZE = OBSERVATION_SIZE + ACTION_SIZE + REQUEST_MASK_SIZE + MESSAGE_SIZE

    ROW_FEATURES = 6        # presence, x, y, vx, vy, vehicle_id
    KINEMATICS = 5          # the same without vehicle_id (just a label)
    NEIGHBOURS = 9          # observation rows 1..9; row k pairs with message slot k-1
    CAR_SIZE = 64
    SCENE_SIZE = 128
    # Rough scales that bring message values to about [-1, 1].
    MESSAGE_SCALE = (1.0, 1 / 5.0, 1.0, 1 / 10.0)

    def __init__(self):
        super().__init__()
        # One car = its kinematics, asked flag, message, ego kinematics, action.
        car_input = self.KINEMATICS + 1 + self.MESSAGE_FEATURES + self.KINEMATICS + self.ACTION_SIZE
        self.vehicle_encoder = nn.Sequential(
            nn.Linear(car_input, 128), nn.ReLU(),
            nn.Linear(128, self.CAR_SIZE), nn.ReLU(),
        )
        # Scene = ego kinematics, action, sum and max of all car encodings.
        scene_input = self.KINEMATICS + self.ACTION_SIZE + 2 * self.CAR_SIZE
        self.scene = nn.Sequential(
            nn.Linear(scene_input, 256), nn.ReLU(),
            nn.Linear(256, self.SCENE_SIZE), nn.ReLU(),
        )
        self.ego_head = nn.Linear(self.SCENE_SIZE, self.ROW_FEATURES)
        self.neighbour_head = nn.Sequential(
            nn.Linear(self.CAR_SIZE + self.SCENE_SIZE, 64), nn.ReLU(),
            nn.Linear(64, self.ROW_FEATURES),
        )
        self.reward_head = nn.Linear(self.SCENE_SIZE, 1)
        self.crash_head = nn.Linear(self.SCENE_SIZE, 1)
        self.crash_soon_head = nn.Linear(self.SCENE_SIZE, 1)
        self.register_buffer("message_scale", torch.tensor(self.MESSAGE_SCALE))

    def forward(self, inputs):
        """inputs: (batch, 115) -> next observation, reward, crash logit, crash-soon logit."""
        observation, action, request_mask, messages = torch.split(
            inputs, [self.OBSERVATION_SIZE, self.ACTION_SIZE, self.REQUEST_MASK_SIZE, self.MESSAGE_SIZE],
            dim=-1,
        )
        batch, n = inputs.shape[0], self.NEIGHBOURS
        rows = observation.reshape(batch, 10, self.ROW_FEATURES)[:, :, : self.KINEMATICS]
        ego, cars = rows[:, 0], rows[:, 1:]
        present = cars[:, :, :1]
        asked = request_mask[:, :n].unsqueeze(-1)
        car_messages = messages.reshape(batch, 10, self.MESSAGE_FEATURES)[:, :n] * self.message_scale

        # 1. Encode every car together with its own message.
        ego_and_action = torch.cat((ego, action), dim=-1)
        car_inputs = torch.cat(
            (cars, asked, car_messages, ego_and_action.unsqueeze(1).expand(batch, n, -1)), dim=-1
        )
        car_codes = self.vehicle_encoder(car_inputs) * present  # empty slots -> 0

        # 2. Summarise the scene.
        scene = self.scene(torch.cat(
            (ego_and_action, car_codes.sum(dim=1), car_codes.max(dim=1).values), dim=-1
        ))

        # 3. Predict.
        next_ego = self.ego_head(scene).unsqueeze(1)
        next_cars = self.neighbour_head(
            torch.cat((car_codes, scene.unsqueeze(1).expand(batch, n, -1)), dim=-1)
        )
        next_observation = torch.cat((next_ego, next_cars), dim=1).reshape(batch, self.OBSERVATION_SIZE)
        return next_observation, self.reward_head(scene), self.crash_head(scene), self.crash_soon_head(scene)

    @classmethod
    def load(cls, path):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = cls()
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model

    @torch.no_grad()
    def predict_batch(self, observations, actions, message=None):
        """Imagine one step for many (observation, action) pairs at once.

        ``observations``: (N, 60) tensor, ``actions``: N ints, ``message``: one
        message dictionary shared by all pairs (or None). Returns tensors:
        next observations (N, 60), rewards (N,), crash and crash-soon
        probabilities (N,).
        """
        count = observations.shape[0]
        action_one_hot = torch.zeros(count, self.ACTION_SIZE)
        action_one_hot[torch.arange(count), torch.as_tensor(actions)] = 1.0
        if message is None:
            request_mask, messages = torch.zeros(self.REQUEST_MASK_SIZE), torch.zeros(self.MESSAGE_SIZE)
        else:
            request_mask = torch.as_tensor(message["request_mask"], dtype=torch.float32)
            messages = torch.as_tensor(message["messages"], dtype=torch.float32).reshape(-1)
        inputs = torch.cat(
            (observations, action_one_hot, request_mask.expand(count, -1), messages.expand(count, -1)), dim=1
        )
        next_observations, rewards, crash_logits, crash_soon_logits = self(inputs)
        return (
            next_observations,
            rewards.squeeze(1),
            torch.sigmoid(crash_logits).squeeze(1),
            torch.sigmoid(crash_soon_logits).squeeze(1),
        )

    @torch.no_grad()
    def predict_step(self, observation, action, message=None):
        """Imagine one step.

        ``message`` is the dictionary from ``message_bundle`` (request mask +
        messages), or None for "nobody asked".
        Returns (next observation, reward, crash probability, crash-soon probability).
        """
        action_one_hot = torch.zeros(self.ACTION_SIZE)
        action_one_hot[int(action)] = 1.0
        if message is None:
            request_mask, messages = torch.zeros(self.REQUEST_MASK_SIZE), torch.zeros(self.MESSAGE_SIZE)
        else:
            request_mask = torch.as_tensor(message["request_mask"], dtype=torch.float32)
            messages = torch.as_tensor(message["messages"], dtype=torch.float32).reshape(-1)
        observation = torch.as_tensor(observation, dtype=torch.float32).reshape(-1)
        inputs = torch.cat((observation, action_one_hot, request_mask, messages)).unsqueeze(0)

        next_observation, reward, crash_logit, crash_soon_logit = self(inputs)
        return (
            next_observation.squeeze(0),
            float(reward),
            float(torch.sigmoid(crash_logit)),
            float(torch.sigmoid(crash_soon_logit)),
        )
