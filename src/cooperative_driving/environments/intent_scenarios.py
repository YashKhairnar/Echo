"""Controlled matched-intent scenes for evaluating the value of a request.

These are challenge scenes, not the default traffic distribution. A stock
aggressive/defensive controller follows an assigned stay/merge target. MOBIL
retargeting is disabled for this vehicle to isolate committed intent; steering,
acceleration, collisions, rewards, and the preview API remain unchanged.
"""

import copy
import numpy as np
from highway_env.vehicle.behavior import AggressiveVehicle, DefensiveVehicle, IDMVehicle

from .customEnv import CustomEnv

SCENARIO_SPEC = {
    'version': 'committed_intent_v1', 'lanes': 4, 'ego_lanes': [1, 2],
    'ego_speed': 25., 'ego_target_speed': 25., 'neighbor_dx_uniform': [8., 22.],
    'neighbor_speed_uniform': [22., 28.], 'neighbor_kinds': ['aggressive', 'defensive'],
    'neighbor_kind_prior': [.5, .5], 'neighbor_mobil_retargeting': False,
    'rear_blocker_dx': -11., 'rear_blocker_speed': 25.,
    'background_vehicles': 3, 'hidden_intent_prior': {'stay': .5, 'merge': .5},
}


def matched_intent_pair(seed, duration=3):
    rng = np.random.default_rng(seed)
    ego_lane = int(rng.choice([1, 2]))
    side = int(rng.choice([-1, 1]))
    gap = float(rng.uniform(8., 22.))
    speed = float(rng.uniform(22., 28.))
    kind = str(rng.choice(['aggressive', 'defensive']))
    environment = CustomEnv(config={
        'vehicles_count': 0, 'aggressive_vehicles': 0, 'defensive_vehicles': 0,
        'idm_vehicles': 0, 'initial_lane_id': ego_lane, 'duration': duration,
    })
    environment.reset(seed=seed)
    neighbor_lane = ego_lane + side
    lane = environment.road.network.get_lane(('0', '1', neighbor_lane))
    cls = AggressiveVehicle if kind == 'aggressive' else DefensiveVehicle
    neighbor = cls(environment.road, lane.position(gap, 0), heading=0., speed=speed,
                   target_speed=speed, enable_lane_change=False)
    neighbor.color = (180, 180, 180)
    environment._add_vehicle(neighbor, kind)
    # Close rear vehicles prevent an immediate free lane escape. Separate rear
    # slots keep the queried neighbor visible even when it occupies "beside".
    # IDM keeps them in lane and reacts to their lead vehicle normally.
    for adjacent in [ego_lane - 1, ego_lane + 1]:
        geometry = environment.road.network.get_lane(('0', '1', adjacent))
        blocker = IDMVehicle(environment.road, geometry.position(-11., 0),
                             heading=0., speed=25., target_speed=25., enable_lane_change=False)
        environment._add_vehicle(blocker, 'idm')
    stay = environment
    merge = copy.deepcopy(environment)
    merge.get_vehicle(neighbor.vehicle_id).target_lane_index = ('0', '1', ego_lane)
    metadata = {'seed': seed, 'ego_lane': ego_lane, 'neighbor_lane': neighbor_lane,
                'neighbor_id': neighbor.vehicle_id, 'initial_dx': gap,
                'neighbor_speed': speed, 'behavior': kind, 'prior_merge': .5,
                'scenario_version': SCENARIO_SPEC['version']}
    return {'stay': stay, 'merge': merge}, metadata


def intent_episode(seed, mode, duration=3):
    """Return one hidden variant and close the unused copy."""
    if mode not in ('stay', 'merge'):
        raise ValueError('Intent mode must be stay or merge')
    environments, metadata = matched_intent_pair(seed, duration)
    environments['merge' if mode == 'stay' else 'stay'].close()
    return environments[mode]
