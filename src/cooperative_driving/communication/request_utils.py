"""Mapping between observation slots, vehicles, and request messages.

The ego observation has one row for the ego and up to nine neighbour rows.
Visible neighbours fill the rows in order, and *slot k* means the k-th visible
neighbour (observation row k + 1). Requests and messages are addressed by
slot; ``vehicle_by_id`` turns a slot back into the simulator car.
"""

import numpy as np

from .getMessage import encode_message

MAX_REQUEST_SLOTS = 10
MESSAGE_FEATURES = 4
NO_REQUEST_ACTION = MAX_REQUEST_SLOTS  # request-policy action meaning "ask nobody"
OBSERVATION_FEATURE_COUNT = 6          # presence, x, y, vx, vy, vehicle_id


def visible_neighbor_ids(observation):
    """Return the vehicle IDs of visible neighbours, in slot order."""
    rows = np.asarray(observation, dtype=np.float32).reshape(
        -1, OBSERVATION_FEATURE_COUNT
    )
    visible_ids = [
        int(round(row[5]))
        for row in rows[1:]
        if row[0] > 0 and row[5] >= 0
    ]
    return visible_ids[:MAX_REQUEST_SLOTS]


def valid_request_actions(observation):
    """Request-policy actions allowed now: each visible slot, or ask nobody."""
    return list(range(len(visible_neighbor_ids(observation)))) + [NO_REQUEST_ACTION]


def vehicle_by_id(environment, vehicle_id):
    return next(
        (vehicle for vehicle in environment.road.vehicles
         if vehicle.vehicle_id == vehicle_id),
        None,
    )


def slot_of_vehicle(observation, vehicle):
    """Slot of ``vehicle`` in this observation, or None if it is not visible."""
    if vehicle is None:
        return None
    visible_ids = visible_neighbor_ids(observation)
    return visible_ids.index(vehicle.vehicle_id) if vehicle.vehicle_id in visible_ids else None


def message_bundle(environment, observation, slots):
    """Ask the cars in ``slots`` and return the MLPx message input.

    Returns ``{"request_mask": [10], "messages": [10 x 4]}``; slots that were
    not asked (or are not visible) stay zero.
    """
    visible_ids = visible_neighbor_ids(observation)
    request_mask = [0] * MAX_REQUEST_SLOTS
    messages = [[0.0] * MESSAGE_FEATURES for _ in range(MAX_REQUEST_SLOTS)]
    for slot in slots:
        if not 0 <= slot < len(visible_ids):
            continue
        vehicle = vehicle_by_id(environment, visible_ids[slot])
        if vehicle is None:
            continue
        request_mask[slot] = 1
        messages[slot] = encode_message(vehicle)
    return {"request_mask": request_mask, "messages": messages}
