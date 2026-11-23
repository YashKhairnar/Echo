# Experiments Overview

This hub outlines the progression of experiments in **Echo**, exploring communication-efficient cooperative driving across seven sequential milestones.

Each experiment below has a dedicated page with detailed setup, world-model loss curves, request-policy graphs, and playable driving videos.

---

## Experiment Progression at a Glance

| Experiment | Focus Area | Dataset | Model Architecture | Key Outcome / Finding |
| :--- | :--- | :---: | :---: | :--- |
| **[Experiment 1](experiment_1.html)** | Baseline Integration | 26k transitions | 116 in / 55 out | Proved closed-loop pipeline connecting HighwayEnv, MLPx, and Request DQN. |
| **[Experiment 2](experiment_2.html)** | Scaling & Intent Previews | 80k+ transitions | 116 in / 55 out | Added vehicle intent previews; established robust data collection pipelines. |
| **[Experiment 3](experiment_3.html)** | Spatial Slot Realignment | 263k transitions | 116 in / 55 out | Fixed spatial slot alignment for neighbor messages around the ego vehicle. |
| **[Experiment 4](experiment_4.html)** | Planner Forensics & Safety Guards | 566k transitions | 116 in / 55 out | Uncovered runaway speed; added physical safety guards that **cut crashes from 97% to 0%**. |
| **[Experiment 5](experiment_5.html)** | Matched-Intent Scenarios | 145k transitions | **118 in / 57 out** | Proved communication value in cut-in scenes: intent messages **cut crashes in half (12 → 6)**. |
| **[Experiment 6](experiment_6.html)** | Selective Requests | Reused Exp. 5 | Frozen WM5 + DQN | Added counterfactual redundancy penalty to discourage unnecessary queries. |
| **[Experiment 7](experiment_7.html)** | Oracle-Taught Request Policy | 10k oracle labels | Frozen WM5 + BC DQN | Taught policy via privileged simulation oracle: **cut cut-in crashes by 40% (124 → 75)**. |

---

## Detailed Experiment Sub-Sections

<div class="cards">
<p><a href="experiment_1.html"><strong>Experiment 1: Baseline Integration</strong></a><br>Initial proof-of-concept pipeline, 26k dataset, 3-step horizon selection, and baseline video.</p>
<p><a href="experiment_2.html"><strong>Experiment 2: Data Scaling & Intent Previews</strong></a><br>Scaling to 80k+ transitions and querying non-mutating vehicle intent previews.</p>
<p><a href="experiment_3.html"><strong>Experiment 3: Spatial Realignment & Baselines</strong></a><br>Binding messages to 8 fixed spatial slots around the ego and producing baseline checkpoints.</p>
<p><a href="experiment_4.html"><strong>Experiment 4: Crash Forensics & Safety Guards</strong></a><br>Root cause analysis of planner failures and executive safety guards eliminating crashes (97% → 0%).</p>
<p><a href="experiment_5.html"><strong>Experiment 5: Matched-Intent Challenge Scenarios</strong></a><br>Parallel stay vs. cut-in worlds proving that intent messages cut crashes in half (12 → 6).</p>
<p><a href="experiment_6.html"><strong>Experiment 6: Selective Request Learning</strong></a><br>A training-only counterfactual teacher penalizes questions whose replies do not change the planner's move.</p>
<p><a href="experiment_7.html"><strong>Experiment 7: Oracle-Taught Requests</strong></a><br>A privileged simulation oracle teaches the Request DQN via behavior cloning, slashing cut-in crashes to 3.75%.</p>
</div>

---

## Cross-Experiment Comparison Matrix

| Metric | Experiment 1 | Experiment 3 | Experiment 4 | Experiment 5 | Experiment 6 | Experiment 7 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Total Dataset Size** | 26k | 263k | 566k | 145k paired trees | Reused Exp. 5 | ~10k oracle labels |
| **State Vector Size** | 55 | 55 | 55 | **57 (ego controls)** | 57 | 57 |
| **Total Model Inputs** | 116 | 116 | 116 | **118** | 118 | 118 |
| **Highway Crashes (Guarded)** | n/a | ~88% (raw) | **0% (guarded)** | **0% (guarded)** | **0% (guarded)** | **0% (guarded)** |
| **Cut-In Challenge Crashes** | n/a | n/a | 16 / 200 (blind) | 18 / 200 (DQN5) | 9 / 200 (DQN6) | **3 / 200 (Exp 7)** |
| **Large-Scale Cut-In Crashes** | n/a | n/a | n/a | n/a | 124 / 2000 | **75 / 2000** |
| **Primary Checkpoint** | `exp_1/best.pt` | `exp_3/best.pt` | `exp_4/best.pt` | `exp_5/intent_stage_best.pt` | `exp_6/RequestDQN/best.pt` | `exp_7/RequestDQN/best.pt` |

---

## Current Recommendations

- **Best Learned Cooperative Policy:** **Experiment 7** (`checkpoints/experiment_7/RequestDQN/best.pt`) with `--safety-guard --max-target-speed 30 --lane-change-guard`.
- **Highway Cruising Baseline:** On open highways with predictable IDM traffic, **Ask Nobody** achieves zero communication overhead and collision-free driving.
- **Why Communication Matters:** In high-speed, ambiguous cut-in scenes, intent messages are the only way to distinguish benign straight-driving cars from aggressive cut-in vehicles, reducing crashes from ~9% down to 3.75%.
