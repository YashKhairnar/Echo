"""State decoder: turns a 55-value state into a 64 x 256 grayscale frame.

Encoder: an MLP maps the state to a latent vector.
Decoder: transposed convolutions grow the latent vector into an image.

    state (55) -> MLP -> latent (512) -> 128 x 4 x 16 -> ... -> 1 x 64 x 256

The output is a logit per pixel; ``sigmoid(logit) * 255`` is the gray value.
"""

import numpy as np
import torch
import torch.nn as nn


class StateDecoder(nn.Module):
    STATE_SIZE = 55
    HEIGHT = 64
    WIDTH = 256
    SEED_SHAPE = (128, 4, 16)  # four 2x up-samplings -> 64 x 256

    def __init__(self, latent_size=512):
        super().__init__()
        channels, height, width = self.SEED_SHAPE
        self.encoder = nn.Sequential(
            nn.Linear(self.STATE_SIZE, 512), nn.ReLU(),
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
        """states: (batch, 55) -> pixel logits (batch, 1, 64, 256)."""
        seed = self.project(self.encoder(states)).reshape(-1, *self.SEED_SHAPE)
        return self.decoder(seed)

    @torch.no_grad()
    def draw(self, state):
        """Return the predicted frame of one state as a (64, 256) uint8 array."""
        device = next(self.parameters()).device
        state = torch.as_tensor(np.asarray(state), dtype=torch.float32, device=device).reshape(1, -1)
        frame = torch.sigmoid(self(state))[0, 0] * 255
        return frame.round().byte().cpu().numpy()

    @classmethod
    def load(cls, path):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = cls(**checkpoint.get("model_kwargs", {}))
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model
