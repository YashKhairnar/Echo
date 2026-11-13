import numpy as np


NO_REQUEST_ACTION = 10
MAX_NEIGHBOR_SLOTS = 10
OBSERVATION_FEATURE_COUNT = 6


def visible_neighbor_ids(observation):
    """Return visible vehicle IDs, excluding the ego row."""
    # Row 0 is ego. Other rows are valid only when presence and ID are set.
    observation = np.asarray(observation, dtype=np.float32).reshape(
        -1,
        OBSERVATION_FEATURE_COUNT,
    )

    visible_ids = []
    for row in observation[1:]:
        if row[0] > 0 and row[5] >= 0:
            visible_ids.append(int(round(row[5])))

    return visible_ids[:MAX_NEIGHBOR_SLOTS]


def valid_request_actions(observation):
    """Return visible-neighbor slots plus the no-request action."""
    # Request actions use local observation slots, not permanent vehicle IDs.
    return list(range(len(visible_neighbor_ids(observation)))) + [
        NO_REQUEST_ACTION
    ]


def requested_vehicle(environment, observation, request_action):
    """Map a visible-neighbor slot to its current vehicle object."""
    # Convert the local slot back to the actual car object before messaging.
    if request_action == NO_REQUEST_ACTION:
        return None

    visible_ids = visible_neighbor_ids(observation)
    if not 0 <= request_action < len(visible_ids):
        return None

    requested_id = visible_ids[request_action]
    return next(
        (
            vehicle
            for vehicle in environment.road.vehicles
            if vehicle.vehicle_id == requested_id
        ),
        None,
    )
