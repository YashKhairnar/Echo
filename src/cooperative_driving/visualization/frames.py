"""Shared frame settings for the state-to-frame decoder.

Frames are small grayscale versions of the CustomEnv render. The camera and
the road look the same as the normal 600 x 150 render, only scaled down.
"""

import numpy as np

from ..environments.customEnv import CustomEnv

FRAME_WIDTH = 256
FRAME_HEIGHT = 64
# Same view as the default 600 px wide render at scaling 5.5.
FRAME_SCALING = 5.5 * FRAME_WIDTH / 600

ROAD_GREY = 100
GRAY_WEIGHTS = np.array([0.299, 0.587, 0.114])


def frame_config():
    return {"screen_width": FRAME_WIDTH, "screen_height": FRAME_HEIGHT, "scaling": FRAME_SCALING}


def make_frame_environment():
    """A CustomEnv that renders frames offscreen at the frame size."""
    environment = CustomEnv(config=frame_config(), render_mode="rgb_array")
    environment.reset(seed=0)
    environment.render()  # creates the viewer
    # HighwayEnv turns drawing off when SDL uses the dummy video driver.
    environment.viewer.enabled = True
    return environment


def to_gray(rgb):
    """(H, W, 3) uint8 RGB -> (H, W) uint8 grayscale."""
    return np.clip(rgb[..., :3] @ GRAY_WEIGHTS, 0, 255).astype(np.uint8)


def vehicle_mask(gray):
    """Pixels that are neither road grey nor white lane markings."""
    gray = np.asarray(gray, dtype=np.int16)
    return (np.abs(gray - ROAD_GREY) > 20) & (gray < 230)


def iou(first, second):
    union = np.logical_or(first, second).sum()
    return float(np.logical_and(first, second).sum() / union) if union else 1.0
