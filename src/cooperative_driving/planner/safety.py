"""Optional geometry guard for the first action of a WM plan."""

import math


def update_controller_state(state, executed_action, lane_change_cooldown=5):
    """Advance optional per-episode memory after one real driving action."""
    if state is None:
        return
    remaining = max(0, int(state.get('lane_change_cooldown', 0)))
    if executed_action in (0, 2):
        state['lane_change_cooldown'] = int(lane_change_cooldown)
    else:
        state['lane_change_cooldown'] = max(0, remaining - 1)
    acceleration_remaining = max(0, int(state.get('acceleration_cooldown', 0)))
    if executed_action == 4:
        state['acceleration_cooldown'] = 3
    else:
        state['acceleration_cooldown'] = max(0, acceleration_remaining - 1)


def guard_action(values, ego, visible, lanes_count, min_clearance=8., min_ttc=3.,
                 check_lane_changes=False, max_target_speed=None,
                 comfort_ttc=None, controller_state=None,
                 lane_change_cooldown=5, commit_state=True):
    """Mask unsafe actions, then brake for an unsafe visible current-lane front gap.

    Uses public geometry and longitudinal velocity only. This is a heuristic
    filter, not collision avoidance proof or a check of imagined future lanes.
    Optional lane checks require an eight-metre bumper gap (by default) to
    remain after a constant-velocity projection over min_ttc seconds. They
    also prevent new lane commands until the current maneuver has settled.
    Discrete action order: LEFT, IDLE, RIGHT, FASTER, SLOWER.
    """
    if len(values) != 5 or not all(math.isfinite(float(value)) for value in values):
        raise ValueError("Guard expects five finite action values")
    if not (math.isfinite(min_clearance) and math.isfinite(min_ttc)) or min(min_clearance, min_ttc) <= 0:
        raise ValueError("Guard thresholds must be finite and positive")
    if comfort_ttc is not None:
        if not math.isfinite(comfort_ttc) or comfort_ttc <= min_ttc:
            raise ValueError('Comfort TTC must be finite and greater than emergency TTC')
        if not check_lane_changes:
            raise ValueError('Comfort TTC requires checked lane changes')
    if lane_change_cooldown < 0:
        raise ValueError('Lane-change cooldown must be nonnegative')
    lane = int(ego.lane_index[2])
    target_lane = int(ego.target_lane_index[2])
    allowed = [0, 1, 2, 3, 4]
    # HighwayEnv applies lane-change commands relative to target_lane_index.
    if lane == 0 or target_lane == 0:
        allowed.remove(0)
    if lane == lanes_count - 1 or target_lane == lanes_count - 1:
        allowed.remove(2)
    raw_action = max(range(5), key=values.__getitem__)
    boundary_masked = raw_action not in allowed
    if max_target_speed is not None:
        if not math.isfinite(max_target_speed) or max_target_speed <= 0:
            raise ValueError('Maximum target speed must be finite and positive')
        if ego.target_speed >= max_target_speed:
            allowed.remove(3)
    acceleration_cooldown = (max(0, int(controller_state.get('acceleration_cooldown', 0)))
                             if controller_state is not None else 0)
    if acceleration_cooldown and 3 in allowed:
        allowed.remove(3)
    lane_checks = {}
    if check_lane_changes:
        # Do not stack or reverse lane commands until the vehicle has settled.
        changing = lane != target_lane or abs(float(getattr(ego, 'heading', 0.))) > .05
        if hasattr(ego, 'road'):
            target = ego.road.network.get_lane(ego.target_lane_index)
            _, lateral = target.local_coordinates(ego.position)
            changing = changing or abs(float(lateral)) > .3
        for command, destination in [(0, target_lane - 1), (2, target_lane + 1)]:
            if command not in allowed:
                continue
            conflicts = []
            for candidate in visible.values():
                vehicle = candidate['vehicle']
                if int(vehicle.lane_index[2]) != destination:
                    continue
                dx = float(vehicle.position[0] - ego.position[0])
                clearance = abs(dx) - (ego.LENGTH + vehicle.LENGTH) / 2
                closing = float(ego.velocity[0] - vehicle.velocity[0]) * (1 if dx >= 0 else -1)
                projected_clearance = clearance - max(0., closing) * min_ttc
                if projected_clearance < min_clearance:
                    conflicts.append({'vehicle_id': vehicle.vehicle_id,
                                      'position': 'front' if dx >= 0 else 'rear',
                                      'clearance': clearance, 'closing_speed': closing,
                                      'projected_clearance': projected_clearance})
            lane_checks[str(command)] = {'destination_lane': destination,
                                         'maneuver_in_progress': changing,
                                         'conflicts': conflicts}
            if changing or conflicts:
                allowed.remove(command)
    cooldown_remaining = (max(0, int(controller_state.get('lane_change_cooldown', 0)))
                          if controller_state is not None else 0)
    if cooldown_remaining:
        for command in (0, 2):
            if command in allowed:
                allowed.remove(command)
            if str(command) in lane_checks:
                lane_checks[str(command)]['cooldown_remaining'] = cooldown_remaining
    fronts = [candidate["vehicle"] for candidate in visible.values()
              if int(candidate["vehicle"].lane_index[2]) == lane
              and candidate["vehicle"].position[0] > ego.position[0]]
    front = min(fronts, key=lambda vehicle: vehicle.position[0]) if fronts else None
    clearance = closing = ttc = headway = None
    reasons = []
    if boundary_masked:
        reasons.append("boundary_lane_mask")
    if raw_action == 3 and 3 not in allowed:
        reasons.append('acceleration_cooldown' if acceleration_cooldown
                       else 'target_speed_cap')
    if check_lane_changes and str(raw_action) in lane_checks and raw_action not in allowed:
        check = lane_checks[str(raw_action)]
        if check.get('cooldown_remaining'):
            reasons.append('lane_change_cooldown')
        else:
            reasons.append('maneuver_in_progress' if check['maneuver_in_progress'] else 'destination_lane_gap')
    if front is not None:
        clearance = float(front.position[0] - ego.position[0] - (ego.LENGTH + front.LENGTH) / 2)
        closing = float(ego.velocity[0] - front.velocity[0])
        ttc = max(0., clearance) / closing if closing > 0 else None
        ego_speed = float(getattr(ego, 'speed', ego.velocity[0]))
        headway = max(0., clearance) / max(ego_speed, 1.)
        if clearance < min_clearance:
            reasons.append("front_clearance")
        if ttc is not None and ttc < min_ttc:
            reasons.append("front_ttc")
    braking = "front_clearance" in reasons or "front_ttc" in reasons
    caution_ttc = bool(comfort_ttc is not None and ttc is not None
                       and min_ttc <= ttc < comfort_ttc)
    cautious = bool(not braking and caution_ttc)
    if cautious:
        reasons.append('front_caution')
        if 3 in allowed:
            allowed.remove(3)
    masked_action = max(allowed, key=values.__getitem__)
    if braking:
        action = 4
    elif cautious:
        safe_lane_changes = [command for command in (0, 2) if command in allowed]
        useful_lane_changes = []
        for command in safe_lane_changes:
            destination = lane_checks[str(command)]['destination_lane']
            destination_fronts = [candidate['vehicle'] for candidate in visible.values()
                                  if int(candidate['vehicle'].lane_index[2]) == destination
                                  and candidate['vehicle'].position[0] > ego.position[0]]
            if destination_fronts:
                destination_front = min(destination_fronts,
                                        key=lambda vehicle: vehicle.position[0])
                destination_clearance = float(
                    destination_front.position[0] - ego.position[0]
                    - (ego.LENGTH + destination_front.LENGTH) / 2
                )
            else:
                destination_clearance = math.inf
            lane_checks[str(command)]['front_clearance'] = destination_clearance
            lane_checks[str(command)]['clearance_gain'] = destination_clearance - clearance
            if destination_clearance >= clearance:
                useful_lane_changes.append(command)
        if useful_lane_changes:
            action = max(useful_lane_changes, key=values.__getitem__)
            reasons.append('proactive_lane_change')
        elif caution_ttc:
            action = 4
            reasons.append('proactive_slowdown')
        else:
            action = 1
            reasons.append('proactive_hold')
    else:
        action = masked_action
    if commit_state:
        update_controller_state(controller_state, action, lane_change_cooldown)
    return action, {
        "raw_action": raw_action, "masked_action": masked_action,
        "executed_action": action, "overridden": action != raw_action,
        "braking_required": braking, "reasons": reasons,
        "allowed_actions": allowed, "front_vehicle_id": front.vehicle_id if front else None,
        "front_clearance": clearance, "closing_speed": closing, "ttc": ttc,
        "headway": headway,
        "min_clearance": min_clearance, "min_ttc": min_ttc,
        "lane_change_checks_enabled": check_lane_changes, "lane_change_checks": lane_checks,
        "max_target_speed": max_target_speed, "comfort_ttc": comfort_ttc,
        "front_caution": cautious,
        "lane_change_cooldown": lane_change_cooldown,
        "lane_change_cooldown_remaining": cooldown_remaining,
        "acceleration_cooldown_remaining": acceleration_cooldown,
    }
