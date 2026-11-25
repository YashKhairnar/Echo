# Request DQN Policy

The **Request DQN** is the learned communication policy. At each decision step, it decides **which nearby vehicle to query for private intent information**—or whether to ask nobody at all.

It does not directly control vehicle steering or throttle; instead, it decides what information to feed into the **Driving Planner**.

- **Implementation:** `src/cooperative_driving/policies/request_dqn.py`
- **Algorithm:** Deep Q-Network with dynamic action masking and target synchronization
- **Training Objective:** Huber Loss on TD-error

---

## Current recommendation

The strongest learned configuration is **Experiment 6 DQN + the smooth
controller** (`--comfort-ttc 6`). On identical seeds it improves over DQN5,
keeps ordinary-road crashes at zero, reduces intent crashes to 9/200, and
increases completed lane changes from 171 to 240 per 100 ordinary episodes.

It is the best current choice when safety and realistic lane-changing behavior
matter. It is not the highest cost-adjusted policy: asking nobody still scores
36.566 versus 35.748 on ordinary roads and 2.643 versus 2.594 in intent trials
because it pays no message cost.

---

## Component Architecture

```text
┌──────────────────────────────────────────────┐
│ Ego State Vector (5 values)                  │
│ 8 Spatial Neighbor States (8 × 5 = 40 values)│
└──────────────────────────────────────────────┘
                       │
                       ▼
             ┌───────────────────┐
             │    Request DQN    │
             │  Linear Networks  │
             └───────────────────┘
                       │
                       ▼
       ┌───────────────────────────────┐
       │   9 Q-Values (Action Scores)  │
       │  [Slot 0 ... Slot 7, Nobody]  │
       └───────────────────────────────┘
                       │
       [Mask out slots without cars]
                       │
                       ▼
     Selected Communication Request (or Ask Nobody)
                       │
                       ▼
       Neighbor returns private intent preview
                       │
                       ▼
           Passed to MLPx Driving Planner
```

---

## State Input Vector (45 Values)

The DQN observes a 45-value state vector describing the ego vehicle and up to 8 surrounding spatial slots within 60 meters:

| Section | Size | Values | Description |
| :--- | ---: | :---: | :--- |
| **Ego State** | 5 | Continuous | `[presence, current_lane / 3, speed / 30, target_speed / 30, target_lane / 3]` |
| **8 Spatial Neighbors** | 40 | Continuous | 8 neighbor slots × 5 features: `[presence, lane / 3, speed / 30, x_offset / 60, y_offset / 10]` |

---

## Action Space & Spatial Slots (9 Actions)

The ego vehicle chooses one of 9 discrete request actions:

```text
       LEFT LANE              SAME LANE             RIGHT LANE
┌─────────────────────┐┌─────────────────────┐┌─────────────────────┐
│  0: Left-Front      ││  3: Same-Lane-Front ││  5: Right-Front     │
├─────────────────────┤├─────────────────────┤├─────────────────────┤
│  1: Left-Beside     ││      [ EGO CAR ]    ││  6: Right-Beside    │
├─────────────────────┤├─────────────────────┤├─────────────────────┤
│  2: Left-Rear       ││  4: Same-Lane-Rear  ││  7: Right-Rear      │
└─────────────────────┘└─────────────────────┘└─────────────────────┘
                             [ 8: Ask Nobody ]
```

### Dynamic Action Masking
If a spatial slot contains no vehicle (e.g., no car in `Left-Beside`), that action is masked out. The DQN only selects among visible cars or Action 8 (*Ask Nobody*), preventing wasted queries to empty road space.

---

## Reward Formulation & Trade-offs

The Request DQN is trained to maximize driving return while minimizing communication cost:

```text
R_DQN = R_driving − (c × N_requests)
```

- `R_driving`: Immediate driving reward from HighwayEnv (speed, staying in lane, avoiding crashes).
- `c`: Communication cost penalty (default: **0.05** per requested message).
- `N_requests`: Number of vehicles queried (1 if asking a neighbor, 0 if asking nobody).

### The Communication Trade-Off
- **On open highways:** Standard drivers are predictable from kinematics alone. Asking nobody achieves high driving rewards and zero communication penalties.
- **In ambiguous cut-in scenes:** Asking the neighbor reveals their private intended lane change early, allowing the ego car to brake and avoid high-penalty crashes (-1.0 to -2.0) that far outweigh the 0.05 request cost.

---

## Python API Usage

```python
import random

from scripts.data_collection.show_visible_neighbors import visible_neighbors
from scripts.training.train_request_dqn import (
    request_state, requested_message, select_action, valid_actions,
)
from src.cooperative_driving.policies.request_dqn import RequestDQN

# 1. Load trained policy checkpoint
policy = RequestDQN.load("checkpoints/experiment_6/RequestDQN/best.pt")

# 2. Extract 45-value DQN observation and available slot mask
visible = visible_neighbors(environment, 60.0, 10.0)
dqn_state = request_state(environment, visible, policy.request_state_encoding)
available = valid_actions(visible, policy.request_action_encoding)

# 3. Select request action (greedy)
request_action = select_action(
    policy, dqn_state, available, epsilon=0.0, rng=random.Random(0)
)

# 4. Resolve action: if 8 -> ask nobody; if 0-7 -> query spatial slot
message = requested_message(
    environment, visible, request_action, policy.request_action_encoding
)
```

---

## Double-DQN Target (Experiment 8)

Experiment 8 adds an optional `--td-target double-dqn` training mode. Standard DQN uses the target network both to select and evaluate the next request. Double DQN instead:

1. Uses the online policy to select the best valid next request.
2. Uses the target network to evaluate that selected request.
3. Preserves the same empty-slot action mask in both operations.

The policy can also start from an existing request checkpoint with `--init-checkpoint`. This imports policy weights without restoring the previous optimizer or replay buffer.

The policy implementations are kept separate under `src/cooperative_driving/policies/`:

- `request_dqn.py` defines `RequestDQN` for standard DQN checkpoints.
- `request_ddqn.py` defines `RequestDDQN` for Double-DQN checkpoints.
- `load_request_policy()` reads checkpoint metadata and restores the matching class automatically. Older Experiment 8 checkpoints are recognized from their saved `td_target` field.

### Related results

For the latest selective-request training curves and evaluation, see
[Experiment 6](experiment_6.html). Earlier learned-policy baselines are in
[Experiment 5](experiment_5.html) and [Experiment 3](experiment_3.html). The
Double-DQN feasibility run and its controlled Experiment 6 comparison are in
[Experiment 8](experiment_8.html).
