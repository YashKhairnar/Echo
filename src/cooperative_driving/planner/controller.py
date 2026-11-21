"""Shared real first-action selection for request training and evaluation."""
import numpy as np
from .safety import guard_action


def driving_action(planner, environment, observation, message, *, safety_guard=False,
                   min_clearance=8., min_ttc=3., lane_change_guard=False,
                   max_target_speed=None, comfort_ttc=None, batched=False,
                   visible=None, controller_state=None, commit_state=True):
    if not safety_guard and not batched:
        return planner.plan(observation, message), None
    values = (planner.action_values_batched if batched else planner.action_values)(observation, message)
    if safety_guard:
        if visible is None:
            raise ValueError('Guarded controller requires the visible-neighbor mapping')
        return guard_action(values, environment.vehicle, visible,
                            environment.config['lanes_count'], min_clearance, min_ttc,
                            check_lane_changes=lane_change_guard,
                            max_target_speed=max_target_speed,
                            comfort_ttc=comfort_ttc,
                            controller_state=controller_state,
                            commit_state=commit_state)
    return int(np.argmax(values)), None
