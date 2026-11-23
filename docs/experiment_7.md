# Experiment 7: Oracle-Taught Request Policy

**Experiment 7** introduces an **offline oracle teacher** to guide the Request DQN. Instead of relying purely on trial-and-error reinforcement learning, the ego vehicle learns communication decisions directly from a privileged simulation teacher that knows when asking genuinely changes the outcome.

- **Request DQN Checkpoint:** `checkpoints/experiment_7/RequestDQN/best.pt`
- **World Model Used:** `checkpoints/experiment_5/MLPx/intent_stage_best.pt` (frozen WM5)
- **Teacher Implementation:** `scripts/policy/request_oracle.py`

---

## Objective & Hypothesis

In earlier experiments (DQN5 and DQN6), the agent frequently asked redundant questions on open roads, or missed subtle, dangerous cut-ins. 

Can a **privileged simulation oracle**—which simulates candidate actions 5 steps ahead in a parallel copy of the road—generate high-quality demonstration labels so that the Request DQN learns to ask only when a message prevents a collision?

---

## How the Oracle Teacher Works

```text
Current Traffic Scene
         │
         ▼
[Fork Simulation into Parallel Worlds]
         │
   ├── Try "Ask Nobody" ──> Planner Rollout ──> Record 5-step return
   ├── Try Ask Slot 0   ──> Planner Rollout ──> Record (Return - 0.05 Cost)
   ├── Try Ask Slot 1   ──> Planner Rollout ──> Record (Return - 0.05 Cost)
   └── ...
         │
         ▼
[Compare Outcomes]
• Did asking improve the net return after paying the 0.05 request cost?
  ├── YES ──> Label = "Ask this vehicle"
  └── NO  ──> Label = "Ask Nobody"
         │
         ▼
Train Request DQN via Behavior Cloning (Supervised Imitation)
```

1. **Privileged Evaluation:** For every decision step, the oracle tests every request option in a sandboxed simulator copy.
2. **Net Return Scoring:** The oracle computes the true 5-step trajectory return minus the 0.05 communication penalty.
3. **Conservative Default:** If asking doesn't clearly beat "Ask Nobody", the label defaults to asking nobody.
4. **Behavior Cloning:** The deployed Request DQN is trained on these labels using class-balanced imitation learning.

---

## Benchmark Results (2,000 New Cut-In Trials)

Across 2,000 randomized cut-in challenge scenes (seeds 1600000–1600999), Experiment 7 significantly outperformed all previous learned policies and blind baselines:

| Policy | Cut-in Crashes | Crash Rate | Requests / Trial | Net Return | Mean Speed |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Ask Nobody (Blind)** | 179 | 8.95% | **0.00** | 2.616 | 27.3 m/s |
| **DQN6 (Experiment 6)** | 124 | 6.20% | 0.97 | 2.575 | 27.0 m/s |
| **Experiment 7 (Oracle-Taught)** | **75** | **3.75%** | **0.75** | **2.631** | **27.0 m/s** |
| *Privileged Oracle (Upper Bound)* | *30* | *1.50%* | *0.17* | *2.718* | *27.8 m/s* |

---

## Policy Driving Videos

### 1. Ordinary Highway Driving
The oracle-trained DQN navigating 4-lane highway traffic smoothly with the executive safety guard active:

![Experiment 7 Ordinary Driving](images/dqn_experiment_7_policy.mp4)

### 2. Cut-In Comparison
When a neighboring car abruptly swerves into the ego lane, blind policies crash while the oracle-taught DQN asks early, predicts the maneuver, and brakes safely:

| Scenario | Blind (Ask Nobody) | Learned Policy (Experiment 7) |
| :--- | :--- | :--- |
| **Seed 1300051** | Crashes at step 2 | ![Safe Braking](images/dqn_experiment_7_intent_1300051_merge_dqn.mp4) |
| **Seed 1300005** | ![Blind Crash](images/dqn_experiment_7_intent_1300005_merge_none.mp4) | ![Safe Braking](images/dqn_experiment_7_intent_1300005_merge_dqn.mp4) |
| **Seed 1300061** | ![Blind Crash](images/dqn_experiment_7_intent_1300061_merge_none.mp4) | ![Safe Braking](images/dqn_experiment_7_intent_1300061_merge_dqn.mp4) |

---

## Key Findings & Conclusions

1. **40% Fewer Crashes:** In 2,000 cut-in trials, Experiment 7 slashed crashes from 124 down to 75 compared to DQN6.
2. **First Positive Cost-Adjusted Return:** Because it avoids asking in obvious scenes, the net reward gained from avoiding crashes exceeds the 0.05 query cost, outperforming DQN6 in net return (+0.056).
3. **Selective Querying:** Requests dropped by 23% compared to DQN6 (0.75 vs 0.97 per trial).
4. **Highway Cruising Speed Maintained:** Cruising speed remains rock-solid at 27.0 m/s with no sluggish hesitation.
