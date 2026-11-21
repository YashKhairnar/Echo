# Driving Planner & Safety Guard

The **Planner** chooses the ego vehicle's driving action at every step. It uses the learned **MLPx World Model** to simulate the future consequences of candidate action sequences and picks the move that yields the highest cumulative predicted reward.

- **Planner Implementation:** `src/cooperative_driving/planner/MCP.py`
- **Safety Guard Implementation:** `src/cooperative_driving/planner/safety.py`

---

## Component Architecture

At each decision step, the planner performs a tree search over future candidate trajectories:

```text
Current State + Messages
  │
  ├── Try Action 1 ──> Imagine Step 1 ──> Imagine Step 2 ──> Imagine Step 3 ──> Sum Returns
  ├── Try Action 2 ──> Imagine Step 1 ──> Imagine Step 2 ──> Imagine Step 3 ──> Sum Returns
  ├── Try Action 3 ──> ...
  ├── Try Action 4 ──> ...
  └── Try Action 5 ──> ...
  │
  └── Best Action Candidate
           │
           ▼
  ┌─────────────────────────────────────────────────────────┐
  │ Executive Safety Guard (safety.py)                      │
  │ • Speed Cap: Max 30 m/s                                 │
  │ • Road Boundary Check: Mask invalid lane commands       │
  │ • Front Clearance: Force SLOWER if TTC < 3s or gap < 8m │
  │ • Destination Gap: Block lane change if target occupied │
  └─────────────────────────────────────────────────────────┘
           │
           ▼
     Execute Action in HighwayEnv
```

---

## Planning Algorithm

1. **Candidate Actions:** The ego evaluates 5 discrete actions: `LANE_LEFT`, `IDLE`, `LANE_RIGHT`, `FASTER`, `SLOWER`.
2. **Horizon Rollout (H = 3):**
   The planner simulates all 5³ = 125 possible 3-step action sequences using MLPx:
   ```text
   V(a₁, a₂, a₃) = r₁ + γ·r₂ + γ²·r₃   (with discount factor γ = 0.99)
   ```
3. **Action Selection:** The planner identifies the sequence with the highest total discounted return and selects its first action `a₁*`.
4. **Receding Horizon:** After the first action is executed and the simulator advances, the entire planning cycle repeats with updated observations.

---

## Executive Safety Guards (`safety.py`)

Raw neural world models can occasionally make small optimistic errors (e.g., underestimating the closing speed of a vehicle 3 steps ahead). To guarantee safety, the executive **Safety Guard** filters the planner's selected action using deterministic physical checks:

### Safety and comfort rules

1. **30 m/s Speed Cap:**
   Blocks the `FASTER` command if target speed is already ≥ 30 m/s (speed reward maxes out at 30 m/s; preventing runaway acceleration).
2. **Road Boundary Protection:**
   Blocks `LANE_LEFT` when in Lane 0 (left edge) and `LANE_RIGHT` when in Lane 3 (right edge).
3. **Front Braking Guard:**
   Automatically forces `SLOWER` if bumper clearance to the lead vehicle in the current lane is under **8 meters** or Time-To-Collision (TTC) is under **3 seconds**.
4. **Destination-Lane Clearance Guard:**
   Blocks lane changes if the adjacent destination lane lacks at least 8 m front/rear clearance, or if a previous lane maneuver is still settling.
5. **Optional Proactive Controller:**
   With `comfort_ttc=6`, detects a closing front vehicle before the 3-second
   emergency threshold. It prefers a safe adjacent lane with at least equal
   forward clearance; otherwise it slows. A five-step lane-change cooldown
   prevents left/right ping-pong, and a three-step acceleration cooldown after
   braking prevents immediate `SLOWER → FASTER` reversals.

---

## Python API Usage

```python
from src.cooperative_driving.planner.MCP import MCPPlanner
from src.cooperative_driving.planner.safety import guard_action

# 1. Initialize planner with frozen world model
planner = MCPPlanner(world_model=mlpx_model, action_size=5, planning_horizon=3)

# 2. Score candidate actions and get preferred first action
action_scores = planner.action_values(current_state, messages)

# 3. Pass through executive safety guard
controller_state = {}
safe_action, guard_info = guard_action(
    action_scores,
    env.vehicle,
    visible_neighbors,
    env.config["lanes_count"],
    max_target_speed=30.0,
    min_clearance=8.0,
    min_ttc=3.0,
    check_lane_changes=True,
    comfort_ttc=6.0,
    controller_state=controller_state,
)

# 4. Step environment with the safe action
next_obs, reward, done, truncated, info = env.step(safe_action)
```

---

### Related results

To see how the safety guards reduced crashes from 97% to 0% across 100
validation seeds, and to read the root-cause analysis of raw neural planning,
see [Experiment 4: Safety Guards](experiment_4.html).
