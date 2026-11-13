import torch


class MCPPlanner:
    """Simple model-predictive planner using a trained world model.

    The planner evaluates candidate action sequences with the MLP and returns
    only the first action. The real environment must execute that action.
    Predicted transitions are not used for DQN training.
    """

    def __init__(
        self,
        world_model,
        action_size=5,
        planning_horizon=5,
        discount_factor=0.99,
    ):
        self.world_model = world_model
        self.action_size = action_size
        self.planning_horizon = planning_horizon
        self.discount_factor = discount_factor

    @torch.no_grad()
    def plan(self, observation, message=None):
        """Return the first action from the best predicted action sequence."""
        observation = torch.as_tensor(observation, dtype=torch.float32).reshape(-1)
        best_action = 0
        best_return = float("-inf")

        # Evaluate candidate driving actions in the learned world model. Only
        # the first action of the best predicted sequence is executed in the
        # real environment; later actions are planning-only.
        for action in range(self.action_size):
            predicted_observation, predicted_reward = (
                self.world_model.predict_step(observation, action, message)
            )
            predicted_return = float(predicted_reward.item())

            if self.planning_horizon > 1:
                predicted_return = self._rollout(
                    predicted_observation,
                    message,
                    predicted_return,
                    depth=1,
                )

            if predicted_return > best_return:
                best_return = predicted_return
                best_action = action

        return best_action

    @torch.no_grad()
    def _rollout(self, observation, message, total_return, depth):
        if depth >= self.planning_horizon:
            return total_return

        best_future_return = float("-inf")
        for action in range(self.action_size):
            next_observation, reward = self.world_model.predict_step(
                observation,
                action,
                message,
            )
            future_return = total_return + (
                self.discount_factor ** depth
            ) * float(reward.item())
            
            future_return = self._rollout(
                next_observation,
                message,
                future_return,
                depth + 1,
            )
            best_future_return = max(best_future_return, future_return)

        return best_future_return
