"""Dataset adapter for the merged intent-aware JSONL World Model data."""

import json

import numpy as np
import torch
from torch.utils.data import Dataset

MAX_MESSAGE_SLOTS = 8
MESSAGE_FEATURES = 7
ACTION_SIZE = 5
STATE_SIZE = 11 * 5


def episode_group(record):
    """Keep all counterfactuals from the same source episode in one split."""
    return (record.get("dataset_source", "unknown"),
            record.get("source_episode", record.get("episode", 0)))


def heldout_episode_groups(groups):
    return set(sorted(set(groups), key=str)[::5])


def encode_message(message):
    behavior = message.get("behavior_type", "unknown")
    behavior_code = {
        "aggressive": 1.0,
        "defensive": -1.0,
        "idm": 0.0,
    }.get(behavior, 0.0)
    target_lane = message.get("target_lane")
    target_lane_id = target_lane[2] / 3.0 if target_lane is not None else 0.0
    return [
        1.0,
        behavior_code,
        float(target_lane_id),
        float(message.get("target_speed", 0.0)) / 30.0,
        float(message.get("acceleration", 0.0)) / 6.0,
        float(message.get("steering", 0.0)),
        float(bool(message.get("crashed", False))),
    ]


def encode_messages(record):
    """Encode messages into the fixed left-to-right spatial slot order.

    Empty spatial positions must be preserved. Legacy packed message records
    cannot be mapped reliably to spatial positions from IDs alone.
    """
    messages = record.get("messages") or {}
    if not messages:
        return [0.0] * (MAX_MESSAGE_SLOTS * MESSAGE_FEATURES)
    by_id = {str(vehicle_id): message for vehicle_id, message in messages.items()}
    slot_ids = record.get("visible_vehicle_slots")
    if slot_ids is None or len(slot_ids) != MAX_MESSAGE_SLOTS:
        raise ValueError(
            "Legacy packed message slots found. Recollect informed data with "
            "the corrected collector (eight spatial slots including None); "
            "existing vehicle IDs do not identify missing spatial positions."
        )
    if not set(by_id).issubset({str(item) for item in slot_ids if item is not None}):
        raise ValueError("Requested message vehicle is absent from visible_vehicle_slots")

    encoded = []
    for vehicle_id in slot_ids[:MAX_MESSAGE_SLOTS]:
        message = by_id.get(str(vehicle_id)) if vehicle_id is not None else None
        encoded.extend(encode_message(message) if message is not None else [0.0] * MESSAGE_FEATURES)
    encoded.extend([0.0] * (MAX_MESSAGE_SLOTS * MESSAGE_FEATURES - len(encoded)))
    return encoded


class CombinedWorldModelDataset(Dataset):
    """Merged JSONL transitions with a fixed-size state/message representation."""

    def __init__(self, path, compact_records=False):
        # Training only needs action/message presence after encoding. Keeping
        # the full decoded JSON for every counterfactual wastes substantial RAM.
        self.records = []
        self.inputs = []
        self.next_states = []
        self.rewards = []
        self.crashes = []
        self.groups = []
        self.state_size = None
        with open(path) as file:
            for line in file:
                if line.strip():
                    self._append_record(json.loads(line), compact_records)

        if not self.records:
            raise ValueError("Dataset is empty")
        self.inputs = torch.tensor(np.asarray(self.inputs), dtype=torch.float32)
        self.next_states = torch.tensor(np.asarray(self.next_states), dtype=torch.float32)
        self.rewards = torch.tensor(self.rewards, dtype=torch.float32)[:, None]
        self.crashes = torch.tensor(self.crashes, dtype=torch.float32)[:, None]

    def _append_record(self, record, compact):
        state = np.asarray(record["state"], dtype=np.float32).reshape(-1)
        next_state = np.asarray(record["next_state"], dtype=np.float32).reshape(-1)
        if self.state_size is None:
            self.state_size = len(state)
        if self.state_size not in (55, 57) or len(state) != self.state_size or len(next_state) != self.state_size:
            raise ValueError(
                f"Expected consistent state size 55 or 57, got {len(state)} -> {len(next_state)}"
            )
        if self.state_size == 57 and record.get('state_encoding') != 'kinematics_ego_control_v1':
            raise ValueError('57-value states must declare kinematics_ego_control_v1')
        action = np.zeros(ACTION_SIZE, dtype=np.float32)
        action[int(record["ego_action"])] = 1.0
        self.inputs.append(np.concatenate([state, action, encode_messages(record)]).astype(np.float32))
        self.next_states.append(next_state)
        self.rewards.append(float(record["reward"]))
        # "terminated" is a crash: offroad_terminal is off in CustomEnv.
        self.crashes.append(float(bool(record.get("terminated", False))))
        self.groups.append(episode_group(record))
        self.records.append({"ego_action": record["ego_action"], "messages": bool(record.get("messages"))}
                            if compact else record)
        if compact:
            for key in ('dataset_source', 'depth'):
                if key in record:
                    self.records[-1][key] = record[key]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return (
            self.inputs[index],
            self.next_states[index],
            self.rewards[index],
            self.crashes[index],
        )
