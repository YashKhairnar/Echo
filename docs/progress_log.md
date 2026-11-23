# Project Progress Log

A historical summary of key milestones and developments in the Echo project, tracing the evolution from the initial baseline proof-of-concept to the oracle-taught cooperative policy.

---

## Current Architecture Snapshot

| Area | Current Implementation |
| :--- | :--- |
| **Highway Environment** | 4 lanes, 1 controlled ego car, aggressive/defensive/IDM background traffic (`CustomEnv`) |
| **Observation Space** | 55-value kinematic state + 2 ego control targets (57 values total, up to 60m visibility) |
| **Communication Space** | 8 spatial slots surrounding ego; 7 message features per slot (`target_lane`, `speed`, `acceleration`) |
| **World Model (MLPx)** | 118 inputs (57 state + 5 action + 56 messages) → 57 next-state + 1 reward prediction (`intent_stage_best.pt`) |
| **Driving Planner** | Monte Carlo tree search rollout (horizon $H=3$; $5^3=125$ trajectories) with dynamic physical safety guard |
| **Request Policy (DQN)** | Experiment 7 Oracle-Taught DQN via behavior cloning (`checkpoints/experiment_7/RequestDQN/best.pt`) |
| **Executive Safety Guard** | 30 m/s speed cap, boundary protection, dynamic stopping clearance, proactive acceleration masking |

---

## Milestone History

### Milestone 7: Oracle-Taught Request Policy (Experiment 7)
- **Objective:** Train the Request DQN directly from a privileged simulation oracle to ask only when intent messages change the outcome.
- **Implementation:** Created `request_oracle.py` which forks candidate request decisions into parallel simulator rollouts and extracts clean demonstration labels. Trained the policy via balanced behavior cloning.
- **Key Result:** Across 2,000 challenge cut-in trials, slashed crashes from 124 down to 75 (40% reduction) compared to DQN6, while asking 23% fewer questions and maintaining full cruising speed (27 m/s).

### Milestone 6: Selective Request Teacher & Smooth Control (Experiment 6)
- **Objective:** Discourage redundant communication on open highways and smooth out lane-change maneuvers.
- **Implementation:** Added a counterfactual teacher penalty (+0.10) for questions whose replies did not alter the planner's decision. Introduced a 5-step lane-change cooldown and comfort TTC smoothing (`--comfort-ttc 6`).
- **Key Result:** Reduced highway query rate by 50% (33.3 → 16.9 requests/episode) and cut challenge cut-in crashes to 9/200.

### Milestone 5: Matched-Intent Challenge Scenarios (Experiment 5)
- **Objective:** Prove the tangible, measurable value of communication in scenes where external kinematics are ambiguous.
- **Implementation:** Created the "parallel worlds" benchmark (identical initial kinematics, diverging hidden intent: stay vs. cut-in). Upgraded MLPx to 118 inputs / 57 outputs to include ego target speed and lane.
- **Key Result:** Intent messages cut world-model crashes in half (12 down to 6), definitively proving that communication resolves ambiguous cut-ins.

### Milestone 4: Crash Forensics & Executive Safety Guards (Experiment 4)
- **Objective:** Uncover why raw neural world-model planning crashed despite low training loss.
- **Implementation:** Discovered runaway speed (ego accelerating to >50 m/s) and continuation optimism. Built deterministic executive safety guards (`safety.py`) enforcing a 30 m/s speed cap, boundary checking, and emergency front braking.
- **Key Result:** Completely eliminated closed-loop crashes across 100 test seeds (**97% → 0% crashes**).

### Milestone 3: Spatial Slot Realignment (Experiment 3)
- **Objective:** Fix neighbor message indexing so models receive consistent spatial representations.
- **Implementation:** Realigned communication to 8 fixed ego-centric spatial slots (Left-Front, Same-Front, Right-Front, etc.).
- **Key Result:** Produced stable baseline models (`WM3` and `DQN3`) on 263k transitions.

### Milestone 2: Scaling & Vehicle Intent Previews (Experiment 2)
- **Objective:** Scale data collection and query non-mutating vehicle intentions.
- **Implementation:** Expanded dataset to 80k+ transitions and allowed ego to query candidate vehicles for private cruise speed and destination lane.
- **Key Result:** Established automated multi-threaded dataset generation and counterfactual transition logging.

### Milestone 1: Baseline Closed-Loop Pipeline (Experiment 1)
- **Objective:** Build initial proof-of-concept integrating HighwayEnv, MLPx neural world model, and Request DQN.
- **Implementation:** 3-layer MLP (`256 → 256 → 128`) trained on 26k transitions. Evaluated 1-step, 3-step, and 5-step planning horizons.
- **Key Result:** Proved feasibility of world-model driving planning; established the 3-step rollout horizon standard (lowest reward error and highest action agreement).
