"""Double-DQN request policy.

The Q-network architecture is intentionally shared with RequestDQN. Double
DQN differs in how its bootstrap target is trained: the online policy selects
the next action and the target policy evaluates it.
"""

from .request_dqn import RequestDQN


class RequestDDQN(RequestDQN):
    """Request policy trained with a Double-DQN bootstrap target."""

    POLICY_TYPE = "request_ddqn"
    TD_TARGET = "double-dqn"
