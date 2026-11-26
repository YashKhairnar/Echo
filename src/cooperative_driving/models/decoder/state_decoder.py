"""State decoders: turn a 55-value state into a 64 x 256 grayscale frame.

Two architectures, both output one logit per pixel; ``sigmoid(logit) * 255``
is the gray value.

``StateDecoder`` (default, "vehicle"): a per-vehicle encoder-decoder.
    The camera is fixed (the ego sits at 30% of the width and 50% of the
    height), so every pixel of a 32 x 128 grid has a known position in metres
    relative to the ego. For each vehicle and pixel, the network gets the
    offset from the car to the pixel (in car lengths and widths) and the car
    velocity, and learns "what does this pixel look like near this car".
    The vehicle features are max-pooled over vehicles, joined with road
    features from the pixel's road y, and a small conv decoder up-samples
    them to 64 x 256.

``MLPDeconvDecoder`` (baseline, "mlp_deconv"): an MLP maps the whole state to
    a latent vector; transposed convolutions grow it into an image. It must
    learn to place every car from a flat vector, which is hard for small cars.
"""

import numpy as np
import torch
import torch.nn as nn

from ...visualization.frames import (
    CAR_LENGTH, CAR_WIDTH, FRAME_HEIGHT, FRAME_WIDTH, LANE_WIDTH, STATE_X_RANGE, STATE_Y_RANGE,
    pixel_offsets_m,
)
from ...visualization.state_renderer import FEATURES, ROWS

STATE_SIZE = ROWS * FEATURES
HEIGHT, WIDTH = FRAME_HEIGHT, FRAME_WIDTH


class _FrameDecoder(nn.Module):
    """Shared helpers: drawing one state and loading checkpoints."""

    STATE_SIZE = STATE_SIZE
    HEIGHT = HEIGHT
    WIDTH = WIDTH
    ARCHITECTURE = None

    @torch.no_grad()
    def draw(self, state):
        """Return the predicted frame of one state as a (64, 256) uint8 array."""
        device = next(self.parameters()).device
        state = torch.as_tensor(np.asarray(state), dtype=torch.float32, device=device).reshape(1, -1)
        frame = torch.sigmoid(self(state))[0, 0] * 255
        return frame.round().byte().cpu().numpy()

    @torch.no_grad()
    def draw_batch(self, states, batch_size=128):
        """Return predicted frames of many states as an (N, 64, 256) uint8 array."""
        device = next(self.parameters()).device
        states = torch.as_tensor(np.asarray(states), dtype=torch.float32)
        frames = []
        for start in range(0, len(states), batch_size):
            logits = self(states[start:start + batch_size].to(device))
            frames.append((torch.sigmoid(logits[:, 0]) * 255).round().byte().cpu())
        return torch.cat(frames).numpy()

    @staticmethod
    def load(path):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        model_class = ARCHITECTURES[checkpoint.get("architecture", "mlp_deconv")]
        model = model_class(**checkpoint.get("model_kwargs", {}))
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model


class StateDecoder(_FrameDecoder):
    ARCHITECTURE = "vehicle"
    GRID = (32, 128)  # feature grid, up-sampled 2x to the frame size

    def __init__(self, hidden=32, features=16):
        super().__init__()
        # Camera geometry is computed, not learned: keep it out of checkpoints.
        self.register_buffer("pixels_m", torch.from_numpy(pixel_offsets_m(*self.GRID)), persistent=False)
        self.register_buffer("car_size", torch.tensor([CAR_LENGTH, CAR_WIDTH]), persistent=False)
        self.register_buffer("state_scale", torch.tensor([STATE_X_RANGE, STATE_Y_RANGE]), persistent=False)
        # Per (vehicle, pixel): offset in car lengths/widths, velocity, is_ego.
        self.vehicle_mlp = nn.Sequential(
            nn.Linear(5, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, features), nn.ReLU(),
        )
        # Per pixel: road y in lane widths (lane lines and road edges).
        self.road_mlp = nn.Sequential(
            nn.Linear(1, hidden), nn.ReLU(), nn.Linear(hidden, features), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(2 * features, 32, 3, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1), nn.ReLU(),  # 64 x 256
            nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 1, 3, padding=1),
        )

    def vehicle_inputs(self, states):
        """(batch, 55) -> positions [m] rel. to ego (B, 11, 2), velocities (B, 11, 2),
        is_ego (B, 11, 1), present (B, 11, 1).

        ``present`` is the state's presence value. Real states have 0 or 1;
        a world-model prediction can be in between, and the forward pass
        treats more than 0.5 as present.
        """
        rows = states.reshape(-1, ROWS, FEATURES)
        present = rows[..., :1].clone()
        present[:, 0] = 1.0
        positions = rows[..., 1:3] * self.state_scale
        positions[:, 0] = 0.0  # the camera follows the ego
        velocities = rows[..., 3:5].clone()
        velocities[:, 1:] += velocities[:, :1]  # others are relative to the ego
        is_ego = torch.zeros_like(present)
        is_ego[:, 0] = 1.0
        return positions, velocities, is_ego, present

    def forward(self, states):
        """states: (batch, 55) -> pixel logits (batch, 1, 64, 256)."""
        batch, pixels = states.shape[0], self.pixels_m.shape[0]
        positions, velocities, is_ego, present = self.vehicle_inputs(states)
        # Car-to-pixel offset in car lengths and widths, so a car spans about 1.
        offsets = (self.pixels_m[None, None] - positions[:, :, None]) / self.car_size
        per_vehicle = torch.cat((
            offsets,                                                   # (B, 11, P, 2)
            velocities[:, :, None].expand(-1, -1, pixels, -1),
            is_ego[:, :, None].expand(-1, -1, pixels, -1),
        ), dim=-1)
        vehicle_features = self.vehicle_mlp(per_vehicle) * (present[:, :, None] > 0.5)
        vehicle_features = vehicle_features.max(dim=1).values          # (B, P, F)

        ego_y = states.reshape(-1, ROWS, FEATURES)[:, 0, 2:3] * STATE_Y_RANGE  # (B, 1) [m]
        road_y = (ego_y + self.pixels_m[None, :, 1]) / LANE_WIDTH         # (B, P)
        road = self.road_mlp(road_y[..., None])                          # (B, P, F)

        grid_h, grid_w = self.GRID
        features = torch.cat((vehicle_features, road), dim=-1)         # (B, P, 2F)
        features = features.transpose(1, 2).reshape(batch, -1, grid_h, grid_w)
        return self.decoder(features)


class MLPDeconvDecoder(_FrameDecoder):
    ARCHITECTURE = "mlp_deconv"
    SEED_SHAPE = (128, 4, 16)  # four 2x up-samplings -> 64 x 256

    def __init__(self, latent_size=512):
        super().__init__()
        channels, height, width = self.SEED_SHAPE
        self.encoder = nn.Sequential(
            nn.Linear(STATE_SIZE, 512), nn.ReLU(),
            nn.Linear(512, latent_size), nn.ReLU(),
        )
        self.project = nn.Sequential(nn.Linear(latent_size, channels * height * width), nn.ReLU())
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.ReLU(),  # 8 x 32
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(),   # 16 x 64
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1), nn.ReLU(),   # 32 x 128
            nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1),               # 64 x 256
        )

    def forward(self, states):
        seed = self.project(self.encoder(states)).reshape(-1, *self.SEED_SHAPE)
        return self.decoder(seed)


ARCHITECTURES = {model.ARCHITECTURE: model for model in (StateDecoder, MLPDeconvDecoder)}


def load_decoder(path):
    """Load a decoder checkpoint of either architecture."""
    return _FrameDecoder.load(path)
