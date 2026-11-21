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


LANE_WIDTH = 4.0      # [m]
LANE_LINES_Y = [-2.0 + LANE_WIDTH * k for k in range(5)]  # road edges and lane lines [m]
STATE_Y_RANGE = 16.0  # [m], normalization range of y in the state (4 lanes x 4 m)


def lane_line_rows(ego_y_normalized, height=FRAME_HEIGHT, margin=1.5):
    """Boolean (height,) mask of frame rows on a lane line or road edge.

    The camera centres the ego vertically, so a line at road y sits at row
    (y - ego_y) * scaling + height / 2.
    """
    ego_y = float(ego_y_normalized) * STATE_Y_RANGE
    rows = np.arange(height) + 0.5
    line_rows = [(y - ego_y) * FRAME_SCALING + height / 2 for y in LANE_LINES_Y]
    return np.any([np.abs(rows - line) <= margin for line in line_rows], axis=0)


def vehicle_mask(gray, ego_y_normalized=None):
    """Pixels that are neither road grey nor white lane markings.

    With ``ego_y_normalized`` (state[2]), rows on lane lines are left out.
    A learned decoder draws dashed lines as a soft grey stripe, which would
    otherwise count as vehicle pixels.
    """
    gray = np.asarray(gray, dtype=np.int16)
    mask = (np.abs(gray - ROAD_GREY) > 20) & (gray < 230)
    if ego_y_normalized is not None:
        mask &= ~lane_line_rows(ego_y_normalized, gray.shape[0])[:, None]
    return mask


def iou(first, second):
    union = np.logical_or(first, second).sum()
    return float(np.logical_and(first, second).sum() / union) if union else 1.0
