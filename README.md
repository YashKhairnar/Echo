# Echo: Cooperative Driving

A self-driving vehicle learns **when to ask nearby cars for information, and whom to ask**. Because communication incurs a cost, the ego vehicle must learn to ask only when a neighbor's hidden intent can change driving safety and rewards. Built on [HighwayEnv](https://github.com/Farama-Foundation/HighwayEnv).

---

## How It Works

```text
┌────────────────┐     1. Request Policy (DQN)       ┌────────────────────────┐
│  Ego Vehicle   │ ────────────────────────────────> │ Neighbor Car (Slot 0-7)│
│ (CustomEnv, 4L)│ <──────────────────────────────── │ (Replies with Intent)  │
└────────────────┘       2. Private Intent Message   └────────────────────────┘
        │
        ▼ 3. 3-Step Simulation
┌─────────────────────────────────┐
│     MLPx World Model            │
│ (Predicts Next State & Reward)  │
└─────────────────────────────────┘
        │
        ▼ 4. Action Optimization
┌─────────────────────────────────┐
│     Model-Predictive Planner    │ ──> [Executes Safe Action: Lane / Speed]
│    + Executive Safety Guard     │
└─────────────────────────────────┘
```

---

## Core System Modules

| Module | Purpose | Code |
| :--- | :--- | :--- |
| **CustomEnv** | 4-lane highway with mixed aggressive/defensive traffic and 8 spatial observation slots | `src/cooperative_driving/environments/customEnv.py` |
| **Request DQN** | Learned communication policy: selects which vehicle slot to ask or asks nobody | `src/cooperative_driving/policies/request_dqn.py` |
| **MLPx World Model** | Neural network simulating 3-step future state and driving reward | `src/cooperative_driving/models/MLPx/mlpx.py` |
| **Driving Planner** | Receding-horizon tree search ($5^3=125$ trajectories) picking the best driving action | `src/cooperative_driving/planner/MCP.py` |
| **Safety Guard** | Executive speed cap (30 m/s) and lane-clearance filter eliminating crashes | `src/cooperative_driving/planner/safety.py` |
| **Intent Scenarios** | Challenge scenarios with hidden driver intentions (stay vs. cut-in) | `src/cooperative_driving/environments/intent_scenarios.py` |

---

## Key Findings

1. **Highway Cruising Doesn't Need Communication:** On standard open highways, standard vehicle movements are predictable from position and speed alone. Asking nobody achieves **0% crashes** at 27.4 m/s while avoiding unnecessary communication costs.
2. **Intent Communication Shines in Ambiguous Cut-ins:** In paired challenge scenarios where driver intentions are hidden until the last moment, intent communication **cuts crashes in half** (from 12 down to 6).
3. **Safety Guards are Essential:** Unbounded neural planners tend to spam acceleration commands up to 50 m/s. Pairing MLPx with an executive speed cap (30 m/s) and destination-lane gap checks reduced crash rates from **97% down to 0%**.
4. **Selective Requests Improve the Learned Policy:** Experiment 6 approximately
   halved DQN5's message count on identical seeds and improved intent crashes
   from 10/200 to 9/200. Asking nobody still has the best cost-adjusted return,
   so selective communication remains an open learning problem.
5. **Oracle-Taught Requests Improve Cut-In Safety:** Experiment 7 uses privileged
   simulation labels to train the request DQN, reducing cut-in crashes from
   124/2,000 with DQN6 to 75/2,000 while making fewer requests (0.75 vs. 0.97
   per trial).
6. **Best Current Learned Configuration:** Experiment 7 is the strongest
   learned policy for ambiguous cut-ins; DQN6 with the smooth controller
   (`--comfort-ttc 6`) remains the best-documented ordinary-highway behavior
   configuration, with zero crashes.

---

## Project Timeline

<a href="https://yashkhairnar.github.io/Echo/timeline.html"><img src="docs/images/project_timeline.svg" width="100%" alt="Echo project timeline"></a>

[Open the interactive timeline](https://yashkhairnar.github.io/Echo/timeline.html) directly.

---

## Quickstart

```bash
# 1. Run live visual rendering with learned policy and safety guards
.venv/bin/python -m scripts.evaluation.render_request_dqn \
  --experiment 7 \
  --dqn-checkpoint checkpoints/experiment_7/best.pt \
  --mlpx-checkpoint checkpoints/experiment_5/MLPx/intent_stage_best.pt \
  --episodes 1 --seed 1200000 \
  --safety-guard --max-target-speed 30 --lane-change-guard --comfort-ttc 6

# 2. Evaluate request policy across 100 test scenarios
.venv/bin/python -m scripts.evaluation.evaluate_request_dqn \
  --experiment 7 --episodes 100 --seed 1200000 --planning-horizon 3 \
  --dqn-checkpoint checkpoints/experiment_7/best.pt \
  --mlpx-checkpoint checkpoints/experiment_5/MLPx/intent_stage_best.pt \
  --safety-guard --max-target-speed 30 --lane-change-guard \
  --comfort-ttc 6 --batched-planner

# 3. Analyze dataset statistics
.venv/bin/python -m scripts.evaluation.analyze_intent_aware_dataset --experiment 5

# 4. Rebuild static HTML documentation site
.venv/bin/python -m scripts.docs.build_docs_site
```

---

## Documentation Site

- [docs/index.html](docs/index.html): Documentation Home
- [docs/experiments.html](docs/experiments.html): Experiments 1–7 Breakdown & Key Findings
- [docs/experiment_7.html](docs/experiment_7.html): Experiment 7 oracle-taught request policy
- [docs/planner.html](docs/planner.html): Model-predictive planner & safety guards
- [docs/dqn_policy.html](docs/dqn_policy.html): Learned Request DQN communication policy
- [docs/intent_scenarios.html](docs/intent_scenarios.html): Challenge cut-in scenarios & intent validation
- [docs/dataset_dashboard.html](docs/dataset_dashboard.html): Interactive dataset statistics & distribution charts
- [docs/rendering.html](docs/rendering.html): Visualizer controls & video generation
- [docs/training.html](docs/training.html): Step-by-step training & evaluation runbook
- [docs/progress_log.html](docs/progress_log.html): Detailed milestone history & experiment logs
