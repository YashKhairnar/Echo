import torch
import torch.nn as nn

from ..communication.request_utils import MAX_REQUEST_SLOTS, NO_REQUEST_ACTION


class RequestDQNPolicy(nn.Module):
    """Q-network for the request policy: ask one visible car, or nobody.

    Actions ``0..9`` ask the car in that observation slot, and action
    ``NO_REQUEST_ACTION`` (10) asks nobody, so at most one request is sent
    per step.

    The network is per-vehicle, like MLPx: the same small network scores
    every neighbour from ``[its kinematics, ego kinematics, scene summary]``.
    "Ask the car behind me in the target lane" is then one rule that works in
    any slot, instead of something a flat network must learn separately for
    each of the ten slots.
    """

    ROW_FEATURES = 6         # presence, x, y, vx, vy, vehicle_id
    KINEMATIC_FEATURES = 5   # vehicle_id is an arbitrary label, so not used
    NEIGHBOUR_ROWS = 9
    ACTION_COUNT = MAX_REQUEST_SLOTS + 1
    HIDDEN = 64

    def __init__(self):
        super().__init__()
        k = self.KINEMATIC_FEATURES
        self.vehicle_encoder = nn.Sequential(
            nn.Linear(2 * k, 128), nn.ReLU(), nn.Linear(128, self.HIDDEN), nn.ReLU()
        )
        self.scene = nn.Sequential(
            nn.Linear(k + 2 * self.HIDDEN, 128), nn.ReLU(), nn.Linear(128, self.HIDDEN), nn.ReLU()
        )
        self.ask_head = nn.Sequential(
            nn.Linear(2 * self.HIDDEN, 64), nn.ReLU(), nn.Linear(64, 1)
        )
        self.no_request_head = nn.Sequential(
            nn.Linear(self.HIDDEN, 64), nn.ReLU(), nn.Linear(64, 1)
        )

    def forward(self, observation):
        """Return Q-values of shape (batch, 11). Slots with no car get -1e9."""
        rows = torch.as_tensor(observation, dtype=torch.float32).reshape(
            -1, 10, self.ROW_FEATURES
        )[:, :, : self.KINEMATIC_FEATURES]
        batch = rows.shape[0]
        ego = rows[:, 0]
        neighbours = rows[:, 1 : self.NEIGHBOUR_ROWS + 1]
        presence = neighbours[:, :, 0]

        ego_per_car = ego.unsqueeze(1).expand(-1, self.NEIGHBOUR_ROWS, -1)
        embeddings = self.vehicle_encoder(torch.cat((neighbours, ego_per_car), dim=-1))
        embeddings = embeddings * presence.unsqueeze(-1)
        scene = self.scene(
            torch.cat((ego, embeddings.sum(dim=1), embeddings.max(dim=1).values), dim=-1)
        )

        scene_per_car = scene.unsqueeze(1).expand(-1, self.NEIGHBOUR_ROWS, -1)
        ask_q = self.ask_head(torch.cat((embeddings, scene_per_car), dim=-1)).squeeze(-1)
        ask_q = ask_q.masked_fill(presence <= 0, -1e9)

        q_values = torch.full((batch, self.ACTION_COUNT), -1e9, device=rows.device)
        q_values[:, : self.NEIGHBOUR_ROWS] = ask_q
        q_values[:, NO_REQUEST_ACTION] = self.no_request_head(scene).squeeze(-1)
        return q_values

    @torch.no_grad()
    def act(self, observation, valid_actions):
        """Greedy action among ``valid_actions``."""
        q_values = self(observation)[0]
        return max(valid_actions, key=lambda action: float(q_values[action]))
