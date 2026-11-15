"""Exploration driving policies used for world-model data collection."""

import random

# DiscreteMetaAction indices in highway-env.
LANE_LEFT, IDLE, LANE_RIGHT, FASTER, SLOWER = 0, 1, 2, 3, 4
LONGITUDINAL_ACTIONS = (IDLE, FASTER, SLOWER)


class MergeExploration:
    """Random driving that reliably attempts the merge at varied moments.

    Pure uniform sampling picks LANE_LEFT on ~20% of steps, so the ego almost
    always merges in the first few seconds and the data rarely contains merges
    after slowing down or from different gaps. Instead, each episode draws a
    merge step: before it the ego only changes speed, at it the ego merges,
    and afterwards actions are uniform random.
    """

    def __init__(self, action_count, latest_merge_step=12, rng=None):
        self.action_count = action_count
        self.latest_merge_step = latest_merge_step
        self.rng = rng or random.Random()
        self.merge_step = 0

    def reset(self):
        self.merge_step = self.rng.randint(0, self.latest_merge_step)

    def act(self, timestep):
        if timestep < self.merge_step:
            return self.rng.choice(LONGITUDINAL_ACTIONS)
        if timestep == self.merge_step:
            return LANE_LEFT
        return self.rng.randrange(self.action_count)


class UniformExploration:
    """Uniformly random driving actions (legacy highway data)."""

    def __init__(self, action_count, rng=None):
        self.action_count = action_count
        self.rng = rng or random.Random()

    def reset(self):
        pass

    def act(self, timestep):
        return self.rng.randrange(self.action_count)


def make_exploration(env_name, action_count, rng=None):
    if env_name == "merge":
        return MergeExploration(action_count, rng=rng)
    return UniformExploration(action_count, rng=rng)
