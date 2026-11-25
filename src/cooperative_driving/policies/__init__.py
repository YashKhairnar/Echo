import torch

from .request_ddqn import RequestDDQN
from .request_dqn import RequestDQN


def load_request_policy(path):
    """Load the correct request-policy class from checkpoint metadata."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    policy_type = checkpoint.get("policy_type")
    if policy_type is None:
        # Compatibility with checkpoints created before policy_type was saved.
        td_target = checkpoint.get("td_target") or checkpoint.get(
            "training_config", {}
        ).get("td_target", "dqn")
        policy_type = "request_ddqn" if td_target == "double-dqn" else "request_dqn"
    classes = {
        RequestDQN.POLICY_TYPE: RequestDQN,
        RequestDDQN.POLICY_TYPE: RequestDDQN,
    }
    if policy_type not in classes:
        raise ValueError(f"Unknown request policy type: {policy_type}")
    return classes[policy_type].from_checkpoint(checkpoint)


__all__ = ["RequestDQN", "RequestDDQN", "load_request_policy"]
