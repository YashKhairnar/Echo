"""Per-vehicle, message-aware world model.

The flat MLPx reads the state, the action and the eight message slots as one
long vector. This model keeps the car structure instead:

1. One shared MLP encodes each observed car row (rows 1 to 10).
2. Each car row is joined with the message from that same car. The message
   slot (left/ego/right lane, front/beside/rear) is matched to the car row by
   position, with the same geometry as ``visible_neighbors``.
3. A message that matches no car row becomes its own token.
4. Attention (with the ego as the query) and max pooling join all tokens.
5. Each row gets its own next-row prediction (a residual on the current row).
   Shared heads predict the reward and a "crash soon" logit.

The input is the same flat vector as MLPx (state, action one-hot, 56 message
values), so the training data, the planner and the evaluation scripts do not
change. The public methods are the same as ``MLPxWorldModel``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .mlpx import MLPxWorldModel

# Kinematics observation normalization of HighwayEnv 1.12.1 with four lanes:
# x in [-5 * MAX_SPEED, 5 * MAX_SPEED], y in [-4 * lanes, 4 * lanes].
X_SCALE = 5.0 * 40.0
Y_SCALE = 4.0 * 4
LANE_WIDTH = 4.0
VISIBILITY = 60.0
SIDE_WINDOW = 10.0
VEHICLE_ROWS = 11
ROW_FEATURES = 5

# Index of (lane delta, longitudinal slot) in SLOT_ORDER of
# scripts/data_collection/show_visible_neighbors.py.
FRONT, BESIDE, REAR = 0, 1, 2
SLOT_INDEX = {(-1, FRONT): 0, (-1, BESIDE): 1, (-1, REAR): 2,
              (0, FRONT): 3, (0, REAR): 4,
              (1, FRONT): 5, (1, BESIDE): 6, (1, REAR): 7}


def row_slots(rows):
    """Return the spatial message slot of each neighbour row, or -1.

    ``rows`` has shape (batch, 10, 5) with normalized, ego-relative
    presence, x, y, vx and vy. Also returns the distance to the ego.
    """
    present = rows[..., 0] > 0.5
    dx = rows[..., 1] * X_SCALE
    dy = rows[..., 2] * Y_SCALE
    lane = torch.round(dy / LANE_WIDTH)
    distance = torch.sqrt(dx ** 2 + dy ** 2)
    beside = (dx.abs() <= SIDE_WINDOW) & (lane != 0)
    longitudinal = torch.where(beside, torch.full_like(dx, BESIDE),
                               torch.where(dx > 0, torch.full_like(dx, FRONT), torch.full_like(dx, REAR)))
    slots = torch.full_like(dx, -1, dtype=torch.long)
    for (lane_delta, position), index in SLOT_INDEX.items():
        slots[(lane == lane_delta) & (longitudinal == position)] = index
    visible = present & (distance <= VISIBILITY) & (lane.abs() <= 1)
    slots[~visible] = -1
    return slots, distance


def match_messages(rows, messages):
    """Join each message slot with the closest car row in that slot.

    Returns ``assignment`` (batch, 8, 10), 1 where slot s belongs to row r,
    and ``unmatched`` (batch, 8), 1 where a message has no matching row.
    """
    slots, distance = row_slots(rows)
    slot_ids = torch.arange(MLPxWorldModel.MAX_MESSAGE_SLOTS, device=rows.device)
    in_slot = slots.unsqueeze(1) == slot_ids.view(1, -1, 1)          # (B, 8, 10)
    cost = torch.where(in_slot, distance.unsqueeze(1), torch.full_like(in_slot, float('inf'), dtype=rows.dtype))
    best_cost, best_row = cost.min(dim=2)                             # (B, 8)
    found = torch.isfinite(best_cost)
    assignment = torch.zeros_like(cost)
    assignment.scatter_(2, best_row.unsqueeze(2), 1.0)
    assignment = assignment * found.unsqueeze(2)
    has_message = messages[..., 0] > 0.5
    assignment = assignment * has_message.unsqueeze(2)
    unmatched = (has_message & ~found).to(rows.dtype)
    return assignment, unmatched


class PerVehicleMLPx(MLPxWorldModel):
    """Shared per-car encoder with car-matched messages and attention pooling."""

    architecture = 'per_vehicle'
    TOKEN_FEATURES = ROW_FEATURES + MLPxWorldModel.MESSAGE_FEATURES + MLPxWorldModel.MAX_MESSAGE_SLOTS

    def __init__(self, hidden_size=128, state_size=57):
        nn.Module.__init__(self)
        if state_size not in (55, 57):
            raise ValueError('Supported state sizes are 55 (legacy) and 57 (ego controls)')
        self.STATE_SIZE = state_size
        self.INPUT_SIZE = state_size + self.ACTION_SIZE + self.MESSAGE_SIZE
        self.state_encoding = 'kinematics_ego_control_v1' if state_size == 57 else 'kinematics_v1'
        self.hidden_size = hidden_size
        extra = state_size - VEHICLE_ROWS * ROW_FEATURES
        self.extra_size = extra
        h = hidden_size
        self.ego_encoder = nn.Sequential(
            nn.Linear(ROW_FEATURES + extra + self.ACTION_SIZE, h), nn.ReLU(),
            nn.Linear(h, h), nn.ReLU())
        self.token_encoder = nn.Sequential(
            nn.Linear(self.TOKEN_FEATURES + h, h), nn.ReLU(),
            nn.Linear(h, h), nn.ReLU())
        self.query = nn.Linear(h, h)
        self.key = nn.Linear(h, h)
        self.value = nn.Linear(h, h)
        self.trunk = nn.Sequential(nn.Linear(3 * h, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU())
        self.ego_head = nn.Sequential(nn.Linear(2 * h, h), nn.ReLU(), nn.Linear(h, ROW_FEATURES + extra))
        self.car_head = nn.Sequential(nn.Linear(2 * h, h), nn.ReLU(), nn.Linear(h, ROW_FEATURES))
        self.reward_head = nn.Linear(h, 1)
        self.crash_head = nn.Linear(h, 1)

    def split_inputs(self, inputs):
        state = inputs[:, :self.STATE_SIZE]
        action = inputs[:, self.STATE_SIZE:self.STATE_SIZE + self.ACTION_SIZE]
        messages = inputs[:, self.STATE_SIZE + self.ACTION_SIZE:].reshape(
            -1, self.MAX_MESSAGE_SLOTS, self.MESSAGE_FEATURES)
        rows = state[:, :VEHICLE_ROWS * ROW_FEATURES].reshape(-1, VEHICLE_ROWS, ROW_FEATURES)
        extra = state[:, VEHICLE_ROWS * ROW_FEATURES:]
        return state, action, messages, rows, extra

    def forward_all(self, inputs):
        """Return the next state, the reward and the crash-soon logit."""
        state, action, messages, rows, extra = self.split_inputs(inputs)
        batch = inputs.shape[0]
        cars = rows[:, 1:]
        assignment, unmatched = match_messages(cars, messages)          # (B,8,10), (B,8)
        slot_eye = torch.eye(self.MAX_MESSAGE_SLOTS, dtype=inputs.dtype, device=inputs.device)
        car_messages = assignment.transpose(1, 2) @ messages            # (B,10,7)
        car_slots = assignment.transpose(1, 2) @ slot_eye               # (B,10,8)
        car_tokens = torch.cat((cars, car_messages, car_slots), dim=2)
        message_tokens = torch.cat((
            torch.zeros(batch, self.MAX_MESSAGE_SLOTS, ROW_FEATURES, dtype=inputs.dtype, device=inputs.device),
            messages * unmatched.unsqueeze(2),
            slot_eye.expand(batch, -1, -1) * unmatched.unsqueeze(2)), dim=2)
        tokens = torch.cat((car_tokens, message_tokens), dim=1)           # (B,18,20)
        mask = torch.cat(((cars[..., 0] > 0.5) | (car_messages[..., 0] > 0.5), unmatched > 0.5), dim=1)

        ego = self.ego_encoder(torch.cat((rows[:, 0], extra, action), dim=1))   # (B,h)
        hidden = self.token_encoder(torch.cat(
            (tokens, ego.unsqueeze(1).expand(-1, tokens.shape[1], -1)), dim=2))  # (B,18,h)

        # Attention with a zero "null" slot, so a scene with no cars is valid.
        scores = (self.key(hidden) @ self.query(ego).unsqueeze(2)).squeeze(2) / self.hidden_size ** 0.5
        scores = scores.masked_fill(~mask, float('-inf'))
        scores = torch.cat((scores, torch.zeros(batch, 1, dtype=inputs.dtype, device=inputs.device)), dim=1)
        weights = torch.softmax(scores, dim=1)[:, :-1]
        attended = (weights.unsqueeze(2) * self.value(hidden)).sum(dim=1)
        pooled_max = hidden.masked_fill(~mask.unsqueeze(2), 0.0).max(dim=1).values
        context = self.trunk(torch.cat((ego, attended, pooled_max), dim=1))

        ego_delta = self.ego_head(torch.cat((ego, context), dim=1))
        car_hidden = hidden[:, :VEHICLE_ROWS - 1]
        car_delta = self.car_head(torch.cat(
            (car_hidden, context.unsqueeze(1).expand(-1, VEHICLE_ROWS - 1, -1)), dim=2))
        delta = torch.cat((ego_delta[:, :ROW_FEATURES], car_delta.reshape(batch, -1),
                           ego_delta[:, ROW_FEATURES:]), dim=1)
        return state + delta, self.reward_head(context), self.crash_head(context)

    def forward(self, inputs):
        next_state, reward, _ = self.forward_all(inputs)
        return next_state, reward

    @classmethod
    def load(cls, path):
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        model = cls(hidden_size=checkpoint.get('hidden_size', 128), state_size=checkpoint.get('state_size', 57))
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model

    @torch.no_grad()
    def predict_crash(self, observations, actions, message=None):
        """Return the crash-soon probability for each (observation, action)."""
        observations = torch.as_tensor(observations, dtype=torch.float32).reshape(-1, self.STATE_SIZE)
        actions = torch.as_tensor(actions, dtype=torch.long).reshape(-1)
        action_one_hot = torch.zeros(observations.shape[0], self.ACTION_SIZE)
        action_one_hot[torch.arange(observations.shape[0]), actions] = 1.0
        message_vector = self._message_vector(message).expand(observations.shape[0], -1)
        _, _, crash = self.forward_all(torch.cat((observations, action_one_hot, message_vector), dim=1))
        return torch.sigmoid(crash.squeeze(1))


ARCHITECTURES = {'flat': MLPxWorldModel, 'per_vehicle': PerVehicleMLPx}


def load_world_model(path):
    """Load a world model checkpoint with the architecture it was saved with."""
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    return ARCHITECTURES[checkpoint.get('architecture', 'flat')].load(path)
