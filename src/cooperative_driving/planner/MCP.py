class MCPPlanner:
    """Chooses the driving action by imagining the future with MLPx.

    It tries every sequence of ``planning_horizon`` actions (5 x 5 x 5 for 3
    steps) in the world model and returns the first action of the best one.

    The simplified MLPx predicts reward only, so planning uses discounted
    predicted rewards without a learned crash penalty.
    """

    def __init__(self, world_model, action_size=5, planning_horizon=3,
                 discount_factor=0.99, crash_reward=-2.0):
        self.world_model = world_model
        self.action_size = action_size
        self.planning_horizon = planning_horizon
        self.discount_factor = discount_factor
        self.crash_reward = crash_reward

    def plan(self, observation, message=None):
        """Return the best driving action."""
        values = self.action_values(observation, message)
        return max(range(self.action_size), key=values.__getitem__)

    def action_values(self, observation, message=None):
        """Imagined value of each first action."""
        return [self._value(observation, action, message, depth=0) for action in range(self.action_size)]

    def action_values_batched(self, observation, message=None):
        """Same values as ``action_values``, computed level by level.

        All 5 first steps, then all 25 second steps, then all 125 third steps
        go through MLPx as three batches instead of 155 single calls. Used
        where many plans are needed per step (the value-of-information rule).
        Results match ``action_values`` up to floating-point rounding.
        """
        import torch

        observations = torch.as_tensor(observation, dtype=torch.float32).reshape(1, -1)
        levels = []
        for depth in range(self.planning_horizon):
            count = observations.shape[0]
            repeated = observations.repeat_interleave(self.action_size, dim=0)
            actions = list(range(self.action_size)) * count
            next_observations, rewards = self.world_model.predict_batch(
                repeated, actions, message
            )
            levels.append(rewards)
            observations = next_observations

        future = None
        for rewards in reversed(levels):
            values = rewards if future is None else rewards + self.discount_factor * future
            future = values.reshape(-1, self.action_size).max(dim=1).values
        return values.tolist()

    def _value(self, observation, action, message, depth):
        next_observation, reward = self.world_model.predict_step(
            observation, action, message
        )
        future = 0.0
        if depth + 1 < self.planning_horizon:
            future = max(
                self._value(next_observation, next_action, message, depth + 1)
                for next_action in range(self.action_size)
            )
        return reward + self.discount_factor * future


def build_planner(mlpx_checkpoint, environment, planning_horizon=3):
    """Load MLPx and wrap it in the planner used by every script."""
    from ..models.MLPx.mlpx import MLPxWorldModel

    return MCPPlanner(
        MLPxWorldModel.load(mlpx_checkpoint),
        action_size=environment.action_space.n,
        planning_horizon=planning_horizon,
        crash_reward=environment.config["collision_reward"],
    )
