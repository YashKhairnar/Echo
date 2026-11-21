# Intent Scenarios Benchmark

The **Intent Scenarios** benchmark evaluates whether exchanging private intent messages actually helps an autonomous vehicle make safer and more profitable driving decisions.

- **Scene Generator:** `src/cooperative_driving/environments/intent_scenarios.py`
- **Evaluation Script:** `scripts/evaluation/evaluate_intent_scenarios.py`

---

## The Core Problem

On ordinary highways, neighboring vehicles follow predictable traffic models (like IDM). Kinematics alone (position and speed) are usually enough to drive safely.

To rigorously test cooperative communication, we need scenarios with **perceptual ambiguity**: situations where two vehicles have **identical positions and speeds**, but opposite private intentions.

---

## How Matched-Intent Scenes Work

For every test seed, we generate two parallel worlds:

```text
World A: Neighbor STAYS in its lane       World B: Neighbor CUTS INTO ego lane
┌─────────────────────────────────┐       ┌─────────────────────────────────┐
│ [Neighbor Car] ──> Keeps Lane   │       │ [Neighbor Car] ──> Cut-in       │
│                                 │       │                ↘                │
│ [Ego Car]                       │       │ [Ego Car]                       │
└─────────────────────────────────┘       └─────────────────────────────────┘
     Sensors: EXACT SAME SCENE                 Sensors: EXACT SAME SCENE
```

1. **Identical Kinematics:** To an external camera or radar, World A and World B appear identical at the decision step.
2. **Diverging Futures:** In World A, the neighbor continues straight. In World B, the neighbor aggressively merges directly into the ego vehicle's path.
3. **The Intent Message:** Only when asked does the neighbor reveal its private `target_lane`.

---

## A Concrete Example (Seed 400011)

The ego is in Lane 2. A neighboring vehicle is 10.25 meters ahead in Lane 3, traveling at 24.65 m/s.

| Real Neighbor Intent | Ego Action Chosen | Outcome | Reward |
| :--- | :--- | :---: | :---: |
| **World A (Neighbor Stays)** | `FASTER` (Speed up) | Safe Pass | **2.91** |
| | `IDLE` (Coast) | Safe, but slow | 2.53 |
| **World B (Neighbor Cuts in)** | `FASTER` (Speed up) | **CRASH** | 1.03 |
| | `IDLE` / `SLOWER` (Brake) | **Safe Avoidance** | **2.29** |

*The Dilemma:* If the ego car cannot communicate, it must play it safe and choose `IDLE` or brake in both worlds. If it can ask, it accelerates in World A and brakes in World B, maximizing efficiency while completely avoiding collisions.

---

## Benchmark Results (100 Paired Scenes / 200 Trials)

We evaluated 100 paired challenge scenes (seeds 400000–400099) across different policy options:

| Strategy | Requests / Trial | Crashes / 200 | Net Return (After Cost) |
| :--- | :---: | :---: | :---: |
| **Blind Acceleration Rule** | 0.0 | 16 | 2.61 |
| **Conservative Braking Rule** | 0.0 | **0** | 2.13 |
| **Message-Conditioned Rule** | 1.0 | **0** | 2.45 |
| **Blind Simulator Oracle** | 0.0 | **0** | 2.69 |
| **Always-Informed Oracle** | 1.0 | **0** | 2.67 |
| **Selective Simulator Oracle** | **0.16** | **0** | **2.71** |

---

## Key Takeaways

1. **Communication Genuinely Avoids Crashes:** The message-conditioned rule eliminated all 16 crashes that occurred under the blind policy.
2. **In 16% of Scenes, the Message is Critical:** In 16 out of 100 scenes, knowing the hidden intent justified paying the 0.05 communication cost.
3. **Asking Every Time Lowers Net Return:** In the remaining 84 scenes, knowing the intent didn't change the action, meaning the 0.05 request cost was wasted. Always-informed return (2.67) was slightly lower than blind return (2.69).
4. **The Ideal Policy is Selective:** Asking only in the critical 16% of scenes achieved the highest overall return (**2.71**), establishing the target for reinforcement learning.

The learned policies have not yet matched this oracle. DQN5 reduced crashes
from 20 to 18 across one 200-trial set but requested too frequently. Experiment
6 added a training-only penalty for requests that do not change the planner's
move. On identical fresh seeds, DQN6 improved over DQN5: requests 1.88→1.04,
return 2.563→2.597, and crashes 10→9. Asking nobody still had the highest
cost-adjusted return. See the [Experiment 6 report](experiment_6.html).

---

## How to Run

```bash
.venv/bin/python -m scripts.evaluation.evaluate_intent_scenarios \
  --experiment 5 \
  --pairs 100 \
  --seed 400000 \
  --horizon 3 \
  --communication-cost 0.05 \
  --mlpx-checkpoint checkpoints/experiment_5/MLPx/intent_stage_best.pt
```
