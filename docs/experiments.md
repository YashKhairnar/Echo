# Experiments Overview

This hub outlines the progression of experiments in **Echo**, exploring communication-efficient cooperative driving across six sequential milestones.

Each experiment below has a dedicated page with detailed setup, world-model loss curves, request-policy graphs, and playable driving videos.

---

## Experiment Progression at a Glance

| Experiment | Focus Area | Dataset | Model Architecture | Key Outcome / Finding |
| :--- | :--- | :---: | :---: | :--- |
| **[Experiment 1](experiment_1.html)** | Baseline Proof-of-Concept | 26k transitions | 116 in / 55 out | Proved basic integration of HighwayEnv, MLPx, and Request DQN. High crash rates. |
| **[Experiment 2](experiment_2.html)** | Scaling & Intent Previews | 80k+ transitions | 116 in / 55 out | Added neighbor next-action previews. Established data collection pipelines. |
| **[Experiment 3](experiment_3.html)** | Spatial Slot Realignment | 263k transitions | 116 in / 55 out | Fixed spatial alignment for neighbor messages. Produced baseline checkpoints (`WM3`/`DQN3`). |
| **[Experiment 4](experiment_4.html)** | Planner Crash Diagnosis & Safety Guards | 566k transitions | 116 in / 55 out | Identified runaway speed (50 m/s) and missing controller state. Safety guards **reduced crashes from 97% to 0%**. |
| **[Experiment 5](experiment_5.html)** | Matched-Intent Scenarios & Mixed Traffic | 145k transitions | **118 in / 57 out** | Proved value of communication in cut-in scenes: intent messages **cut crashes in half (12 → 6)**. |
| **[Experiment 6](experiment_6.html)** | Selective Requests + Smooth Control | Reused Experiment 5 | Frozen WM5 + selective DQN | Best learned configuration: fewer requests, 9/200 intent crashes, and 240 completed lane changes; ask-nobody still leads adjusted return. |
| **[Experiment 7](experiment_7.html)** | Oracle-Taught Requests | ~10k oracle labels | Frozen WM5 + BC request DQN | Balanced oracle-taught DQN: 3/200 intent crashes at ask-nobody intent return; ask-nobody still leads on ordinary roads. |

---

## Detailed Experiment Sub-Sections

<div class="cards">
<p><a href="experiment_1.html"><strong>Experiment 1: Baseline Integration</strong></a><br>Initial proof-of-concept pipeline, 26k dataset, 3-step horizon selection, and baseline video.</p>
<p><a href="experiment_2.html"><strong>Experiment 2: Data Scaling & Intent Previews</strong></a><br>Scaling to 80k+ transitions and querying non-mutating vehicle intent previews.</p>
<p><a href="experiment_3.html"><strong>Experiment 3: Spatial Realignment & Baselines</strong></a><br>Binding messages to 8 fixed spatial slots around the ego and producing baseline checkpoints.</p>
<p><a href="experiment_4.html"><strong>Experiment 4: Crash Forensics & Safety Guards</strong></a><br>Root cause analysis of planner failures and executive safety guards eliminating crashes (97% → 0%).</p>
<p><a href="experiment_5.html"><strong>Experiment 5: Matched-Intent Challenge Scenarios</strong></a><br>Parallel stay vs. cut-in worlds proving that intent messages cut crashes in half (12 → 6).</p>
<p><a href="experiment_6.html"><strong>Experiment 6: Selective Request Learning</strong></a><br>A training-only counterfactual teacher penalizes questions whose replies do not change the planner's move.</p>
<p><a href="experiment_7.html"><strong>Experiment 7: Oracle-Taught Requests</strong></a><br>A privileged oracle tries every request option in a copy of the road; the request DQN learns its choices by weighted behaviour cloning.</p>
</div>

## Current recommendation

- **Best learned/safety-oriented policy:** Experiment 6 DQN with
  `--comfort-ttc 6`.
- **Best communication-cost-adjusted baseline:** ask nobody.
- **Why the learned policy is still useful:** it reduces intent crashes from
  14/200 to 9/200 and produces substantially more smooth, completed lane
  changes, but its remaining requests still cost more reward than they add.

---

## Cross-Experiment Comparison Matrix

| Metric | Experiment 1 | Experiment 3 | Experiment 4 | Experiment 5 | Experiment 6 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Total Dataset Size** | 26k | 263k | 566k | 145k paired trees | Reused Exp. 5 |
| **State Vector Size** | 55 | 55 | 55 | **57 (includes ego controls)** | 57 |
| **Total Model Inputs** | 116 | 116 | 116 | **118** | 118 |
| **Highway Crashes (Guarded)** | n/a | ~88% (raw) | **0% (guarded)** | **0% (guarded)** | **0 / 100** |
| **Challenge Crashes** | n/a | n/a | 16 / 200 (blind) | 18 / 200 (DQN5) | **9 / 200 (DQN6)** |
| **Primary Checkpoint** | `exp_1/best.pt` | `exp_3/best.pt` | `exp_4/best.pt` | `exp_5/intent_stage_best.pt` | `exp_6/RequestDQN/best.pt` |
