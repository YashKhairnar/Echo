"""Shared real first-action selection for request training and evaluation."""
import numpy as np
from .safety import guard_action, merge_brake_action, update_controller_state


def driving_action(planner, environment, observation, message, *, safety_guard=False,
                   min_clearance=8., min_ttc=3., lane_change_guard=False,
                   max_target_speed=None, comfort_ttc=None, batched=False,
                   visible=None, controller_state=None, commit_state=True,
                   merge_brake=False):
    """Choose the real driving action for one step.

    ``merge_brake`` adds the message-aware brake rule after the guard: a
    requested neighbour that announces a move into the ego lane is treated
    as a front car (same ``min_clearance`` and ``min_ttc`` as the guard).
    """
    if merge_brake and not safety_guard:
        raise ValueError('merge_brake requires the safety guard')
    if not safety_guard and not batched:
        return planner.plan(observation, message), None
    values = (planner.action_values_batched if batched else planner.action_values)(observation, message)
    if safety_guard:
        if visible is None:
            raise ValueError('Guarded controller requires the visible-neighbor mapping')
        action, guard = guard_action(values, environment.vehicle, visible,
                                     environment.config['lanes_count'], min_clearance, min_ttc,
                                     check_lane_changes=lane_change_guard,
                                     max_target_speed=max_target_speed,
                                     comfort_ttc=comfort_ttc,
                                     controller_state=controller_state,
                                     commit_state=commit_state and not merge_brake)
        if merge_brake:
            action, report = merge_brake_action(action, environment, message,
                                                min_clearance, min_ttc)
            if commit_state:
                update_controller_state(controller_state, action)
            if report is not None:
                guard = dict(guard, executed_action=action,
                             overridden=action != guard['raw_action'], merge_brake=report)
        return action, guard
    return int(np.argmax(values)), None
