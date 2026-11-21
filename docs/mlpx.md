# MLPx World Model

**MLPx** is the learned driving world model. It simulates the road environment one step into the future, predicting the **next vehicle kinematics** and **immediate driving reward** resulting from candidate ego actions.

- **Implementation:** `src/cooperative_driving/models/MLPx/mlpx.py`
- **Model Type:** Multi-Layer Perceptron (MLP) with dual output heads
- **Primary Method:** `model.predict_step(state, action, messages)`

---

## Component Architecture

MLPx takes the current road state, ego internal control targets, candidate ego action, and received neighbor messages to jointly predict the next state and driving reward:

```text
┌────────────────────────────────────────────────────────┐
│ Current State: 55 values (11 vehicles × 5 kinematics)  │
│ Ego Internal Controls: 2 values (Target Speed, Lane)   │───┐
│ Ego Candidate Action: 5 values (One-Hot)               │   │
│ Neighbor Messages: 56 values (8 slots × 7 features)    │   │
└────────────────────────────────────────────────────────┘   │
                                                             ▼
                                                    ┌─────────────────┐
                                                    │   MLPx Model    │
                                                    │ 256 → 256 → 128 │
                                                    └─────────────────┘
                                                             │
                              ┌──────────────────────────────┴──────────────────────────────┐
                              ▼                                                             ▼
                ┌───────────────────────────┐                                 ┌───────────────────────────┐
                │   Next State (57 values)  │                                 │      Reward (1 scalar)    │
                └───────────────────────────┘                                 └───────────────────────────┘
```

---

## Input Vector Specification (118 Values)

The input vector contains 118 normalized continuous and one-hot values:

| Field | Dimension | Normalized Range | Description |
| :--- | ---: | :---: | :--- |
| **Vehicle Kinematics** | 55 | `[0, 1]` or `[-1, 1]` | 11 vehicles (ego + 10 nearest neighbors) × 5 features: `[presence, x, y, vx, vy]` |
| **Ego Controls** | 2 | `[0, 1]` | Internal controller state: `target_speed / 30.0` and `target_lane / 3.0` |
| **Ego Action** | 5 | `{0, 1}` | One-hot action vector: `LANE_LEFT`, `IDLE`, `LANE_RIGHT`, `FASTER`, `SLOWER` |
| **Neighbor Messages** | 56 | Varied | 8 fixed spatial slots × 7 message features |

### Spatial Message Slots

Messages are addressed by 8 fixed spatial slots surrounding the ego car within 60 meters:
```text
[Left-Front,  Left-Beside,  Left-Rear]
[Same-Front,                Same-Rear]
[Right-Front, Right-Beside, Right-Rear]
```

Each spatial message slot contains 7 features:
1. `presence`: 1.0 if a vehicle occupies this slot, 0.0 otherwise.
2. `behavior_type`: Categorical label (aggressive = 1.0, defensive = -1.0, neutral/none = 0.0).
3. `target_lane`: The neighbor's intended future lane index (normalized).
4. `target_speed`: The neighbor's cruising target speed.
5. `acceleration`: The neighbor's current longitudinal acceleration.
6. `steering`: The neighbor's current steering angle.
7. `crashed`: 1.0 if the neighbor vehicle has crashed.

*Note:* If a slot is empty or the ego chooses not to query that vehicle, all 7 values are zero-padded.

---

## Output Heads (57 + 1 Values)

MLPx splits its final 128-neuron hidden representation into two specialized linear heads:

1. **Next-State Head (57 values):**
   Predicts the normalized kinematic positions and velocities of all vehicles, plus the ego's updated control state.
2. **Reward Head (1 scalar):**
   Predicts the scalar HighwayEnv driving reward for the transition:
   ```text
   R_driving = w₁·R_speed + w₂·R_right_lane + w₃·R_collision
   ```

---

## Why a 3-Step Rollout Horizon?

When the driving planner simulates future paths, it repeatedly feeds MLPx's predicted next state back into itself:

- **1 Step:** Highly accurate (MAE 0.068), but lacks the foresight needed to plan lane changes or prepare for distant braking.
- **3 Steps (Selected Standard):** Balances strong predictive agreement (86.3% action match with ground-truth simulator) with multi-step foresight.
- **5 Steps:** Open-loop errors compound significantly across multiple recurrent iterations (MAE rises above 0.120; agreement drops below 74%).

---

## Python API Usage

```python
from src.cooperative_driving.models.MLPx.mlpx import MLPx

# 1. Load trained model checkpoint
model = MLPx.load("checkpoints/experiment_5/MLPx/intent_stage_best.pt")

# 2. Predict one-step transition
predicted_next_state, predicted_reward = model.predict_step(
    current_state=state_tensor,     # [1, 57]
    ego_action=action_idx,          # Integer 0-4
    messages=message_tensor         # [1, 56]
)
```

---

### Related results

For training loss curves, held-out validation benchmarks, and the
architectural evolution from WM1 to WM5, see the
[Experiments Overview](experiments.html).
