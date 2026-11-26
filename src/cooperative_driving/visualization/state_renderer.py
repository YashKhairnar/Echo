"""Draw a road image directly from one 55-value state vector.

The state is the CustomEnv observation: 11 rows x [presence, x, y, vx, vy],
normalized to [-1, 1]. Row 0 is the ego in road coordinates. Rows 1-10 are
other vehicles relative to the ego.

``StateRenderer`` turns the state back into metres, puts plain vehicles on a
copy of the CustomEnv road, and renders it with HighwayEnv's own viewer. The
image therefore looks like the real render, but it shows only what the state
contains. Use it to see MLPx predictions as pictures.

Limits of the state, and so of the drawing:
- The ego x is clipped at 200 m from the start. After that, lane stripes stop
  moving, but vehicle positions relative to the ego stay correct.
- The state has no vehicle type and no crash flag, so other vehicles all get
  one neutral colour.
- The heading is not in the state; it is computed from vx and vy. HighwayEnv
  sets velocity = speed x heading direction, so this is exact for moving
  cars. A car with speed 0.1 m/s or less (often a crashed car) is drawn with
  heading 0, and a car with negative speed is drawn turned by 180 degrees.
"""

import math

import numpy as np
from highway_env.road.lane import AbstractLane
from highway_env.vehicle.kinematics import Vehicle

from ..environments.customEnv import CustomEnv

ROWS = 11
FEATURES = 5  # presence, x, y, vx, vy

# HighwayEnv's default Kinematics normalization ranges (see
# KinematicObservation.normalize_obs). Values are mapped from [-R, R] to [-1, 1].
X_RANGE = 5.0 * Vehicle.MAX_SPEED
VELOCITY_RANGE = 2.0 * Vehicle.MAX_SPEED

EGO_COLOR = (50, 200, 0)       # same green as the real render
OTHER_COLOR = (160, 160, 160)  # type is unknown from the state


class DrawnVehicle(Vehicle):
    """A plain vehicle for drawing only.

    HighwayEnv draws tyres on objects whose type is exactly ``Vehicle``, but
    not on the real traffic vehicles. A subclass keeps the two images alike.
    """


def y_range(lanes_count):
    """The y range scales with the number of lanes next to the ego."""
    return AbstractLane.DEFAULT_WIDTH * lanes_count


def decode_state(state, lanes_count=4):
    """Return the present vehicles of ``state`` in metres, in road coordinates.

    Each item is ``{"x", "y", "vx", "vy"}``. Item 0 is the ego.
    """
    rows = np.asarray(state, dtype=np.float64).reshape(ROWS, FEATURES)
    scale = np.array([X_RANGE, y_range(lanes_count), VELOCITY_RANGE, VELOCITY_RANGE])
    ego = rows[0, 1:] * scale
    vehicles = [dict(zip(("x", "y", "vx", "vy"), ego))]
    for row in rows[1:]:
        if row[0] <= 0.5:  # presence
            continue
        values = row[1:] * scale + ego  # relative -> road coordinates
        vehicles.append(dict(zip(("x", "y", "vx", "vy"), values)))
    return vehicles


class StateRenderer:
    """Render state vectors as RGB images with the CustomEnv viewer."""

    def __init__(self, screen_width=600, screen_height=150, scaling=5.5):
        self.environment = CustomEnv(
            config={
                "screen_width": screen_width,
                "screen_height": screen_height,
                "scaling": scaling,
            },
            render_mode="rgb_array",
        )
        self.environment.reset(seed=0)
        self.environment.render()  # creates the offscreen viewer
        # HighwayEnv turns drawing off when SDL uses the dummy video driver.
        self.environment.viewer.enabled = True
        self.lanes_count = self.environment.config["lanes_count"]

    def draw(self, state):
        """Return the image of ``state`` as a (height, width, 3) uint8 array."""
        road = self.environment.road
        vehicles = []
        for index, item in enumerate(decode_state(state, self.lanes_count)):
            speed = math.hypot(item["vx"], item["vy"])
            heading = math.atan2(item["vy"], item["vx"]) if speed > 0.1 else 0.0
            vehicle = DrawnVehicle(road, [item["x"], item["y"]], heading, speed)
            vehicle.color = EGO_COLOR if index == 0 else OTHER_COLOR
            vehicles.append(vehicle)
        road.vehicles = vehicles
        self.environment.vehicle = vehicles[0]  # the camera follows the ego
        return self.environment.render()

    def close(self):
        self.environment.close()
