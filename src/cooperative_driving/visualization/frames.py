"""Shared frame settings, camera geometry and image metrics.

Frames are small grayscale versions of the CustomEnv render. The camera and
the road look the same as the normal 600 x 150 render, only scaled down.

All code that needs the camera, the road or the vehicle-pixel IoU uses this
module, so the values cannot drift apart.
"""

import numpy as np
from highway_env.road.lane import AbstractLane
from highway_env.vehicle.kinematics import Vehicle

from ..environments.customEnv import CustomEnv
from .state_renderer import ROWS, X_RANGE, decode_state, y_range

FRAME_WIDTH = 256
FRAME_HEIGHT = 64
# Same view as the default 600 px wide render at scaling 5.5.
FRAME_SCALING = 5.5 * FRAME_WIDTH / 600

# Camera: HighwayEnv's "centering_position" default puts the ego at 30% of the
# width and 50% of the height.
CAMERA_CENTER = (0.3, 0.5)

LANES = 4
LANE_WIDTH = AbstractLane.DEFAULT_WIDTH                       # [m]
LANE_LINES_Y = [LANE_WIDTH * (k - 0.5) for k in range(LANES + 1)]  # edges and lines [m]
STATE_Y_RANGE = y_range(LANES)                                 # [m]
STATE_X_RANGE = X_RANGE                                        # [m]
CAR_LENGTH, CAR_WIDTH = Vehicle.LENGTH, Vehicle.WIDTH          # [m]

ROAD_GREY = 100
VEHICLE_GRAY_DIFFERENCE = 20  # a vehicle pixel differs from road grey by more
LINE_GRAY = 230               # gray at or above this is a white lane marking
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


# ----------------------------------------------------------------------------
# Camera geometry
# ----------------------------------------------------------------------------

def pixel_offsets_m(grid_h, grid_w, width=FRAME_WIDTH, height=FRAME_HEIGHT, scaling=FRAME_SCALING):
    """(grid_h * grid_w, 2) position of each grid cell centre relative to the
    ego [m], for an image of ``width`` x ``height`` pixels."""
    cell = (width / grid_w) / scaling  # metres per grid cell
    columns = (np.arange(grid_w) + 0.5) * cell - CAMERA_CENTER[0] * width / scaling
    rows = (np.arange(grid_h) + 0.5) * cell - CAMERA_CENTER[1] * height / scaling
    ys, xs = np.meshgrid(rows, columns, indexing="ij")
    return np.stack((xs, ys), axis=-1).reshape(-1, 2).astype(np.float32)


def lane_line_rows(ego_y_normalized, height=FRAME_HEIGHT, scaling=FRAME_SCALING, margin=1.5):
    """Boolean (height,) mask of image rows on a lane line or road edge."""
    ego_y = float(ego_y_normalized) * STATE_Y_RANGE
    rows = np.arange(height) + 0.5
    line_rows = [(y - ego_y) * scaling + height / 2 for y in LANE_LINES_Y]
    return np.any([np.abs(rows - line) <= margin for line in line_rows], axis=0)


def near_vehicles(state, height, width, scaling=FRAME_SCALING, margin_m=1.0):
    """Boolean (height, width) mask of pixels near any vehicle in ``state``.

    A pixel is near a car when it is inside the car's box grown by
    ``margin_m`` on every side (enough for the tilt of a lane change).
    """
    offsets = pixel_offsets_m(height, width, width, height, scaling).reshape(height, width, 2)
    vehicles = decode_state(state, LANES)
    ego = vehicles[0]
    mask = np.zeros((height, width), dtype=bool)
    for vehicle in vehicles:
        dx = offsets[..., 0] - (vehicle["x"] - ego["x"])
        dy = offsets[..., 1] - (vehicle["y"] - ego["y"])
        mask |= (np.abs(dx) <= CAR_LENGTH / 2 + margin_m) & (np.abs(dy) <= CAR_WIDTH / 2 + margin_m)
    return mask


# ----------------------------------------------------------------------------
# Vehicle-pixel metric
# ----------------------------------------------------------------------------

def vehicle_mask(gray, state=None, scaling=FRAME_SCALING):
    """Pixels that are neither road grey nor white lane markings.

    With ``state`` (the real state of the frame), pixels on lane-line rows
    are left out unless they are near a car of that state. A learned decoder
    draws dashed lines as a soft grey stripe; this keeps the stripe out of the
    metric but keeps every car, also a car that crosses a lane line.
    """
    gray = np.asarray(gray, dtype=np.int16)
    mask = (np.abs(gray - ROAD_GREY) > VEHICLE_GRAY_DIFFERENCE) & (gray < LINE_GRAY)
    if state is not None:
        height, width = gray.shape
        rows = np.asarray(state, dtype=np.float32).reshape(ROWS, -1)
        on_lines = lane_line_rows(rows[0, 2], height, scaling)[:, None]
        mask &= ~on_lines | near_vehicles(state, height, width, scaling)
    return mask


def iou(first, second):
    union = np.logical_or(first, second).sum()
    return float(np.logical_and(first, second).sum() / union) if union else 1.0
