"""Privileged request oracle: try every request option in a copy of the road.

For one scene the oracle tries each request option (ask nobody, or ask one
visible vehicle). For each option it does these steps:

1. Get the message that this option gives.
2. Let the planner choose the driving action with that message.
3. Drive a deep copy of the real environment with that action, then drive
   blind (no messages) for the remaining look-ahead steps.
4. Add the real rewards and subtract the request cost.

The oracle is privileged because it uses the real simulator future. It is a
teacher for training and an upper reference for evaluation, not a deployable
policy.

Options that give the same first driving action have the same future, so the
oracle simulates each distinct first action only one time.

The smooth controller (``comfort_ttc``) keeps per-episode memory. The oracle
reads a copy of that memory and never changes the caller's copy.
"""

import copy

import numpy as np

from scripts.data_collection.show_visible_neighbors import visible_neighbors
from scripts.training.train_request_dqn import ASK_NOBODY, requested_message, valid_actions
from src.cooperative_driving.planner.controller import driving_action as choose_driving
from src.cooperative_driving.planner.safety import update_controller_state

ACTION_COUNT = 9
TIE_TOLERANCE = 1e-6


def planner_action(planner, world_model, environment, visible, message, controls,
                   controller_state=None):
    """Return the planner's driving action for one message (no state commit)."""
    observation = world_model.observe(environment)
    action, _ = choose_driving(planner, environment, observation, message, visible=visible,
                               controller_state=controller_state, commit_state=False, **controls)
    return int(action)


def blind_rollout_return(environment, first_action, planner, world_model, controls,
                         lookahead, visibility=60.0, side_window=10.0, gamma=1.0,
                         controller_state=None):
    """Real return of one first action, then blind driving, in a copy of the road."""
    copied = copy.deepcopy(environment)
    state = copy.deepcopy(controller_state)
    try:
        total = 0.0
        action = first_action
        for step in range(lookahead):
            update_controller_state(state, action)
            _, reward, terminated, truncated, _ = copied.step(action)
            total += (gamma ** step) * float(reward)
            if terminated or truncated:
                break
            visible = visible_neighbors(copied, visibility, side_window)
            action = planner_action(planner, world_model, copied, visible, None, controls, state)
        return total
    finally:
        copied.close()


def oracle_request_values(environment, visible, planner, world_model, controls, *,
                          lookahead=3, communication_cost=0.05, visibility=60.0,
                          side_window=10.0, gamma=1.0, controller_state=None):
    """Return a 9-value array: oracle value per request option, -inf if invalid.

    Also returns the driving action that each valid option leads to.
    """
    if controls.get('comfort_ttc') is not None and controller_state is None:
        raise ValueError('The comfort controller needs its controller_state')
    values = np.full(ACTION_COUNT, -np.inf, dtype=np.float64)
    first_actions = {}
    rollout_cache = {}
    for option in valid_actions(visible):
        message = requested_message(environment, visible, option)
        action = planner_action(planner, world_model, environment, visible, message, controls,
                                copy.deepcopy(controller_state))
        if action not in rollout_cache:
            rollout_cache[action] = blind_rollout_return(
                environment, action, planner, world_model, controls, lookahead,
                visibility, side_window, gamma, controller_state)
        cost = communication_cost if option != ASK_NOBODY else 0.0
        values[option] = rollout_cache[action] - cost
        first_actions[option] = action
    return values, first_actions


def oracle_best_set(values, tolerance=TIE_TOLERANCE):
    """All options within the tolerance of the best value."""
    best = float(np.max(values))
    return [int(index) for index in np.flatnonzero(values >= best - tolerance)]


def oracle_choice(values, tolerance=TIE_TOLERANCE):
    """Pick the best option. A tie with ask-nobody picks ask-nobody.

    Asking is chosen only if it is strictly better after the request cost.
    Among tied askers, the lowest slot index wins (deterministic).
    """
    best = oracle_best_set(values, tolerance)
    return ASK_NOBODY if ASK_NOBODY in best else best[0]


def oracle_target_distribution(values, tolerance=TIE_TOLERANCE):
    """Soft cross-entropy target over the options the oracle accepts.

    If ask-nobody is among the best options, the whole mass goes to it.
    Otherwise the mass is shared equally by every tied best asker.
    """
    target = np.zeros(ACTION_COUNT, dtype=np.float32)
    best = oracle_best_set(values, tolerance)
    if ASK_NOBODY in best:
        target[ASK_NOBODY] = 1.0
    else:
        target[best] = 1.0 / len(best)
    return target


def agrees_with_oracle(choice, values, tolerance=TIE_TOLERANCE):
    """A choice agrees if the oracle values it as well as its best option."""
    return int(choice) in oracle_best_set(values, tolerance)
