# Experiment 8: DAgger, Ask Margin and Message-Aware Brake Rule

**Experiment 8** improves the Experiment 7 request DQN in two phases:

- **Phase 1 (DAgger and ask margin):** the DQN drives, the oracle labels the states that the DQN visits, and the DQN asks only when it is clearly sure.
- **Phase 2 (message-aware brake rule):** if a car that the ego asked says that it is moving into the ego lane, the ego treats it as a car in front: it brakes when the car is close, and it never speeds up. The request DQN is then taught again by the oracle with this rule.

- **Request DQN Checkpoint:** `checkpoints/experiment_8/RequestDQN/best.pt` (phase 2). Phase 1 policy: `checkpoints/experiment_8/phase1/best.pt`.
- **Flags:** `--merge-brake --ask-margin 0.5` (neither is stored in the checkpoint). Phase 1 uses `--ask-margin 0.5` only.
- **World Model Used:** `checkpoints/experiment_5/MLPx/intent_stage_best.pt` (the Experiment 6 and 7 world model)
- **Configuration:** the same as Experiments 6 and 7: `--safety-guard --max-target-speed 30 --lane-change-guard --batched-planner`, with `--comfort-ttc 6` for cut-in scenes, and 40-step ordinary roads.

**Result (fresh 2,000 cut-in trials, seeds 1900000):**

| Policy | Crashes | Net return | Requests / trial | Ordinary return (1200000) |
| :--- | ---: | ---: | ---: | ---: |
| Ask nobody | 163 | 2.613 | 0.00 | 36.566 |
| Experiment 7 | 74 | 2.622 | 0.76 | 36.390 |
| Experiment 8, phase 1 | 65 | 2.645 | 0.41 | 36.370 |
| **Experiment 8, phase 2 (final)** | **2** | **2.677** | 0.57 | **36.576** |

The final policy removes almost all cut-in crashes, has the highest return, and equals ask nobody on ordinary roads. In merge scenes it drives 1.1 m/s slower than phase 1, because it brakes for the merging car.

---

## Seeds

| Use | Seeds |
| :--- | :--- |
| Labels | Intent 1100000+, 1700000+, 2000000+; ordinary 1000000+ (see the label tables) |
| Validation, small | Intent 1400000-1400099, ordinary 1250000-1250059 |
| Validation, large | Intent 1800000-1800999 (2,000 trials) |
| Test, fresh | Intent 1900000-1900999 (2,000 trials) |
| Test, Experiment 7 sets | Intent 1300000, 1500000, 1600000-1600999; ordinary 1200000-1200099 |

---

## Phase 1: DAgger and Ask Margin

### Why

Experiment 7 learns only from the states that the oracle drives into. When the DQN drives, it reaches other states, where it has no labels. DAgger adds labels for those states. Experiment 7 also asks about 5 times per 40-step ordinary episode; the ask margin removes the requests where the DQN is not sure.

### Method

**DAgger rounds.** Each round trains a new DQN from the start on all labels so far, with the Experiment 7 recipe (behaviour cloning, 30 epochs, balanced ask weight).

| Round | Driver | New label seeds | New labels (intent / ordinary steps) |
| :--- | :--- | :--- | :--- |
| 0 | Oracle (Experiment 7 labels) | Intent 1100000+, ordinary 1000000+ | 5,967 / 4,000 |
| 1 | Experiment 7 DQN | Intent 1700000-1700999, ordinary 1000100-1000199 | 5,925 / 4,000 |
| 2 | Round 1 DQN | Intent 1701000-1701999, ordinary 1000200-1000299 | 5,951 / 3,900 |
| 3 | Round 2 DQN | Intent 1702000-1702999, ordinary 1000300-1000399 | 5,943 / 4,000 |

In round 2, the round 1 DQN (no margin) crashed in 3 of the 100 ordinary label episodes (seeds 1000203, 1000205, 1000271). The oracle labels these states too. Round 3 was not better than round 2 on validation, so phase 1 uses round 2.

**Ask margin.** The DQN scores each request option. With `--ask-margin m`, it asks somebody only if the best ask score is more than `m` above the ask-nobody score.

Code: `--drive-checkpoint` in `scripts/data_collection/collect_oracle_requests.py`; `select_action(..., ask_margin=m)` in `scripts/training/train_request_dqn.py` and `--ask-margin` in `scripts/evaluation/evaluate_request_dqn.py`.

### Model selection

Small validation (200 intent trials, 60 ordinary episodes):

| Policy | Margin | Crashes / 200 | Intent return | Ordinary return | Ordinary requests |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Experiment 7 | 0 | 4 | 2.642 | 36.452 | 5.6 |
| Experiment 7 | 1.0 | 9 | 2.668 | 36.572 | 2.1 |
| DAgger round 1 | 1.0 | 3 | 2.690 | 36.593 | 3.1 |
| DAgger round 2 | 0.5 | 3 | 2.678 | 36.534 | 5.2 |
| DAgger round 2 | 1.0 | 3 | 2.691 | 36.602 | 2.5 |
| DAgger round 3 | 1.0 | 3 | 2.689 | 36.543 | 2.2 |

The first pick was round 2 with margin 1.0. On the Experiment 7 test set it had **more** crashes than Experiment 7 (87 against 75 of 2,000, p = 0.008); 200 validation trials were too few to show this. So the margin was chosen again on a large validation set, and the final test uses a fresh test set.

Large validation (2,000 trials, 1800000):

| Policy | Margin | Crashes | Net return | Requests / trial |
| :--- | ---: | ---: | ---: | ---: |
| Experiment 7 | 0 | 77 | 2.627 | 0.75 |
| DAgger round 2 | 0 | 61 | 2.635 | 0.61 |
| DAgger round 2 | **0.5** | 68 | 2.646 | 0.42 |
| DAgger round 2 | 1.0 | 81 | 2.652 | 0.27 |

Margin 0.5 is the only setting that beats Experiment 7 on crashes, intent return and ordinary return on validation.

### Phase 1 results against Experiment 7

| Test set | Policy | Crashes | Net return | Requests / trial | Speed, stay / merge (m/s) |
| :--- | :--- | ---: | ---: | ---: | :---: |
| Fresh, 1900000 (2,000) | Experiment 7 | 74 | 2.622 | 0.76 | 27.58 / 26.24 |
| | Phase 1 | **65** | **2.645** | **0.41** | 27.54 / 26.20 |
| 1600000 (2,000) | Experiment 7 | 75 | 2.631 | 0.75 | 27.63 / 26.30 |
| | Phase 1 | **72** | **2.648** | **0.41** | 27.60 / 26.24 |
| 1300000 (200) | Experiment 7 | 3 | 2.638 | 0.80 | 27.37 / 26.15 |
| | Phase 1 | 3 | **2.663** | **0.47** | 27.44 / 26.18 |
| 1500000 (200) | Experiment 7 | 8 | 2.620 | 0.82 | 27.41 / 26.33 |
| | Phase 1 | **7** | **2.654** | **0.46** | 27.58 / 26.32 |

Paired difference, phase 1 minus Experiment 7: fresh set +0.022 [+0.018, +0.027] return, crashes 10 / 1 discordant (McNemar p = 0.012); 1600000 +0.017 [+0.011, +0.022], crashes 10 / 7 (p = 0.63).

Three training seeds on the fresh set (each against the Experiment 7 policy of the same seed): crashes 74 → 65 (p = 0.012), 64 → 57 (p = 0.092), 66 → 55 (p = 0.001), and a higher return for all three.

Ordinary roads (1200000): 36.370 against 36.390 for Experiment 7 (original guard), -0.019 [-0.138, +0.087]; 36.049 against 36.118 (`--comfort-ttc 6`), -0.069 [-0.140, +0.004]. Phase 1 does not improve ordinary roads.

The oracle has 42 crashes on 1600000, so phase 1 left room to improve.

---

## Phase 2: Message-Aware Brake Rule

### Why

A test after phase 1 showed that the request policy was not the limit:

| Policy (fresh 2,000 trials, 1900000) | Crashes, stay scenes | Crashes, merge scenes |
| :--- | ---: | ---: |
| Ask nobody | 0 | 163 |
| Ask every visible car | 0 | 154 |
| Always ask the cut-in car (privileged) | 0 | 83 |
| Phase 1 | 0 | 65 |

Even with the correct "I will merge" message at every step, the planner crashed in 83 of 1,000 merge scenes. The world model and the 3-step planner do not act safely enough on the message. Messages from every car made the prediction worse.

With the brake rule and the privileged "always ask the cut-in car" policy, the large validation set had **0 crashes in 2,000 trials**. So the rule fixes the planner problem when the right car is asked.

### Brake rule (`--merge-brake`)

After the guard chooses the action, for each requested message:

1. Skip the car if its message target lane is not the ego lane, if it is already in the ego lane (the guard handles it), or if it is fully behind the ego.
2. Compute its clearance and time to collision, as for a front car.
3. If the clearance is below 8 m or the time to collision is below 3 s, brake (SLOWER).
4. Otherwise, if the action is FASTER, change it to IDLE.

The thresholds are the existing guard thresholds (`--min-clearance 8`, `--min-ttc 3`). A sweep on the large validation set (clearance 0 to 25 m, time to collision 1.5 to 10 s) gave the same number of crashes for every setting, so no new value was tuned. With the phase 1 policy on the ordinary validation episodes, the rule fired in no step.

Code: `merging_neighbours` and `merge_brake_action` in `src/cooperative_driving/planner/safety.py`, and `merge_brake` in `src/cooperative_driving/planner/controller.py`.

### Request policy

The oracle uses the same controller, so with `--merge-brake` it sees the new value of asking the merging car. The labels were collected again with the rule, and the Experiment 7 recipe was used for training.

| Round | Driver | Seeds | Labels (intent / ordinary steps) | Oracle ask rate (intent / ordinary) |
| :--- | :--- | :--- | :--- | :--- |
| 0 | Oracle | Intent 1100000-1100999, ordinary 1000000-1000099 | 6,000 / 4,000 | 8.2% / 3.6% |
| 1 (DAgger) | Round 0 DQN | Intent 2000000-2000999, ordinary 1000500-1000599 | 6,000 / 4,000 | 9.1% / 5.4% |

With the rule, the oracle drove all 2,000 label scenes without a crash. Without the rule (Experiment 7 labels), the oracle asked in 5.8% of intent steps.

### Model selection (large validation 1800000; ordinary 1250000, 60 episodes)

| Policy | Margin | Crashes | Intent return | Intent requests | Ordinary return | Ordinary requests |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Phase 1 | 0.5 | 68 | 2.646 | 0.42 | 36.534 | 5.2 |
| Phase 1 + rule | 0.5 | 50 | 2.651 | 0.44 | 36.534 | 5.2 |
| Round 0 | 0 | **0** | 2.665 | 0.87 | 36.615 | 1.7 |
| Round 0 | **0.5** | 2 | 2.688 | 0.55 | 36.677 | 0.7 |
| Round 0 | 1.0 | 23 | 2.690 | 0.38 | 36.675 | 0.2 |
| Round 1 | 0 | 4 | 2.665 | 0.79 | 36.488 | 4.6 |
| Round 1 | 0.5 | 8 | 2.678 | 0.61 | 36.635 | 2.1 |
| Round 1 | 1.0 | 23 | 2.682 | 0.46 | 36.682 | 0.7 |

A DAgger round did not help in phase 2, so the final policy uses round 0. Before the test, margin 0.5 was chosen: it has the best cost-adjusted return on both road types, and its 2 crashes against 0 are within noise. Margin 0 is reported as the safety-first variant. Ask nobody on the ordinary validation seeds: 36.715.

### Phase 2 results against phase 1

Cut-in scenes:

| Test set | Policy | Crashes | Net return | Requests / trial | Speed, stay / merge (m/s) |
| :--- | :--- | ---: | ---: | ---: | :---: |
| Fresh, 1900000 (2,000) | Ask nobody | 163 | 2.613 | 0.00 | 27.80 / 26.75 |
| | Phase 1 | 65 | 2.645 | 0.41 | 27.54 / 26.20 |
| | **Phase 2** | **2** | **2.677** | 0.57 | 27.87 / 25.07 |
| | Phase 2, margin 0 | **0** | 2.655 | 0.87 | 27.92 / 24.80 |
| 1600000 (2,000) | DQN6 | 124 | 2.575 | 0.97 | 27.56 / 26.33 |
| | Phase 1 | 72 | 2.648 | 0.41 | 27.60 / 26.24 |
| | **Phase 2** | **1** | **2.682** | 0.58 | 27.86 / 25.10 |
| | Phase 2, margin 0 | **0** | 2.660 | 0.87 | n/a |
| 1300000 (200) | Phase 1 | 3 | 2.663 | 0.47 | 27.44 / 26.18 |
| | **Phase 2** | **0** | **2.690** | 0.54 | 27.85 / 25.36 |
| 1500000 (200) | Phase 1 | 7 | 2.654 | 0.46 | 27.58 / 26.32 |
| | **Phase 2** | **0** | **2.689** | 0.52 | 27.80 / 25.28 |

Paired difference, phase 2 minus phase 1 (95% bootstrap interval; exact McNemar test on crashes):

| Test set | Return | Requests | Crashes only under phase 1 / only under phase 2 | McNemar p |
| :--- | :--- | :--- | :---: | ---: |
| Fresh, 1900000 | +0.032 [+0.022, +0.043] | +0.15 | 63 / 0 | 2e-19 |
| 1600000 | +0.034 [+0.023, +0.044] | +0.17 | 72 / 1 | 2e-20 |
| 1300000 | +0.027 [-0.000, +0.057] | +0.07 | 3 / 0 | 0.25 |
| 1500000 | +0.035 [+0.007, +0.067] | +0.06 | 7 / 0 | 0.016 |

Against DQN6 (1600000): 1 against 124 crashes (p = 2e-37), +0.106 return, -0.39 requests. Against ask nobody (fresh): 2 against 163 crashes, +0.064 return.

**Speed.** In merge scenes, phase 2 drives 1.1 m/s slower than phase 1 (25.07 against 26.20 m/s), because it brakes for the car that merges. In stay scenes it is faster than every other policy (27.87 against 27.54 m/s for phase 1). The return includes the speed reward, and it is higher on every test set.

Three training seeds on the fresh set (each against the phase 1 policy of the same seed):

| Seed | Phase 1 crashes | Phase 2 crashes | Return difference | McNemar p |
| :---: | ---: | ---: | :--- | ---: |
| 0 | 65 | **2** | +0.032 [+0.022, +0.043] | 2e-19 |
| 1 | 57 | **8** | +0.010 [-0.002, +0.022] | 3e-13 |
| 2 | 55 | **1** | +0.023 [+0.012, +0.034] | 1e-16 |

On the 200-trial sets 1300000 and 1500000, all three seeds have 0 crashes (returns 2.671 to 2.690).

Ordinary roads (1200000, 100 episodes of 40 steps):

| Controller | Policy | Net return | Requests / episode | Crashes | Speed (m/s) |
| :--- | :--- | ---: | ---: | ---: | ---: |
| Original guard | Ask nobody | 36.566 | 0.00 | 0 | 27.49 |
| | DQN6 | 35.801 | 16.88 | 0 | 27.53 |
| | Experiment 7 | 36.390 | 4.88 | 0 | 27.52 |
| | Phase 1 | 36.370 | 5.54 | 0 | 27.55 |
| | **Phase 2** | **36.576** | 0.58 | 0 | 27.49 |
| `--comfort-ttc 6` | Ask nobody | **36.391** | 0.00 | 0 | 27.41 |
| | DQN6 | 35.748 | 13.17 | 0 | 27.40 |
| | Experiment 7 | 36.118 | 4.93 | 0 | 27.36 |
| | Phase 1 | 36.049 | 7.15 | 0 | 27.40 |
| | **Phase 2** | 36.359 | 0.60 | 0 | 27.40 |

Phase 2 minus phase 1: +0.205 [+0.104, +0.331] (original guard) and +0.310 [+0.230, +0.396] (`--comfort-ttc 6`). Phase 2 minus ask nobody: +0.009 [-0.028, +0.061] and -0.032 [-0.065, +0.006]; both intervals include zero. Seed means (seeds 0-2): 36.527 and 36.322.

The final policy has no crash in its 600 ordinary test episodes (3 seeds, 2 controllers, 100 episodes each).

---

## Video

Two fresh cut-in scenes where phase 1 crashes and phase 2 does not, then the 2,000-trial results. In each pair the scene, world model, planner and guard are the same; phase 2 adds the brake rule and its own request policy.

- **Scene 1900012:** phase 1 asks the merging car, speeds up and crashes. Phase 2 asks, holds its speed, then brakes, and is safe.
- **Scene 1900038:** phase 1 asks nobody, speeds up and crashes. Phase 2 asks the merging car, brakes, and is safe.

![Experiment 8 phase 1 vs phase 2](images/experiment_8_phase1_vs_phase2.mp4)

---

## Conclusion

- Phase 1 (DAgger and ask margin) reduces cut-in crashes against Experiment 7 (65 against 74 on 2,000 fresh trials, p = 0.012) and halves the requests, but it does not improve ordinary roads.
- Phase 2 (message-aware brake rule) removes almost all cut-in crashes: 2 of 2,000 fresh trials, against 65 for phase 1 and 163 for ask nobody. All three training seeds agree.
- The final policy has the highest return on every test set, and on ordinary roads it equals ask nobody with about 0.6 requests per episode.
- The gain needs both parts of phase 2: the rule with the phase 1 DQN still gives 50 crashes on validation, because that DQN often does not ask the merging car. The DQN taught with the rule asks the right car.
- In merge scenes, the final policy drives 1.1 m/s slower than phase 1, because it brakes for the merging car. In stay scenes it is faster.
- The ask margin must be chosen on a large validation set: a margin that is too large brings crashes back, and 200 trials cannot show this.
- The rule is hand-written. It uses only the message and public geometry, and it fires only when a requested car announces a move into the ego lane.

---

## Commands

Run from the project root. `WM` is `checkpoints/experiment_5/MLPx/intent_stage_best.pt`. Run the intent label collection in four parts of 250 scenes and the ordinary part in two parts of 50 episodes. Run each 2,000-trial evaluation in four parts (`--seed 1900000`, `1900250`, `1900500`, `1900750`).

### Phase 1

DAgger round 1 labels (round 2: `--drive-checkpoint` of round 1, seeds +1000 and +100):

```bash
.venv/bin/python -m scripts.data_collection.collect_oracle_requests \
  --experiment 8 --mlpx-checkpoint $WM \
  --drive-checkpoint checkpoints/experiment_7/stage_bc_balanced/best.pt --drive-seed 1 \
  --safety-guard --lane-change-guard --max-target-speed 30 \
  --ordinary-episodes 0 --intent-scenes 250 --challenge-seed 1700000 \
  --output data/experiment_8/dagger_r1_intent_0.npz
```

```bash
.venv/bin/python -m scripts.data_collection.collect_oracle_requests \
  --experiment 8 --mlpx-checkpoint $WM \
  --drive-checkpoint checkpoints/experiment_7/stage_bc_balanced/best.pt --drive-seed 1 \
  --safety-guard --lane-change-guard --max-target-speed 30 \
  --ordinary-episodes 50 --ordinary-seed 1000100 --intent-scenes 0 \
  --output data/experiment_8/dagger_r1_ordinary_0.npz
```

Train (all Experiment 7 labels plus rounds 1 and 2):

```bash
.venv/bin/python -m scripts.training.train_request_dqn \
  --experiment 8 --episodes 0 --seed 0 --mlpx-checkpoint $WM \
  --scenario-mix mixed --request-state-encoding ego_control_v1 \
  --safety-guard --max-target-speed 30 --lane-change-guard --batched-planner \
  --oracle-demos data/experiment_7/stage_oracle_requests_*.npz \
    data/experiment_8/dagger_r1_*.npz data/experiment_8/dagger_r2_*.npz \
  --bc-epochs 30 --bc-ask-weight balanced \
  --output checkpoints/experiment_8/phase1/best.pt
```

Evaluate:

```bash
.venv/bin/python -m scripts.evaluation.evaluate_request_dqn \
  --experiment 8 --episodes 250 --seed 1900000 --scenario paired-intent --policies dqn \
  --dqn-checkpoint checkpoints/experiment_8/phase1/best.pt --mlpx-checkpoint $WM \
  --ask-margin 0.5 --safety-guard --max-target-speed 30 --lane-change-guard \
  --comfort-ttc 6 --batched-planner --output-dir results/experiment_8/evaluation/fresh_exp8_0
```

For Experiment 7, use `--dqn-checkpoint checkpoints/experiment_7/RequestDQN/best.pt`, `--ask-margin 0` and the output `fresh_exp7_0`. For ordinary roads, use `--seed 1200000 --episodes 100`, and remove `--scenario paired-intent` and `--comfort-ttc 6`.

### Phase 2

Labels with the rule:

```bash
.venv/bin/python -m scripts.data_collection.collect_oracle_requests \
  --experiment 8 --mlpx-checkpoint $WM --merge-brake \
  --safety-guard --lane-change-guard --max-target-speed 30 \
  --ordinary-episodes 0 --intent-scenes 250 --challenge-seed 1100000 \
  --output data/experiment_8/phase2_r0_intent_0.npz
```

```bash
.venv/bin/python -m scripts.data_collection.collect_oracle_requests \
  --experiment 8 --mlpx-checkpoint $WM --merge-brake \
  --safety-guard --lane-change-guard --max-target-speed 30 \
  --ordinary-episodes 50 --ordinary-seed 1000000 --intent-scenes 0 \
  --output data/experiment_8/phase2_r0_ordinary_0.npz
```

Train:

```bash
.venv/bin/python -m scripts.training.train_request_dqn \
  --experiment 8 --episodes 0 --seed 0 --mlpx-checkpoint $WM \
  --scenario-mix mixed --request-state-encoding ego_control_v1 \
  --safety-guard --max-target-speed 30 --lane-change-guard --batched-planner \
  --oracle-demos data/experiment_8/phase2_r0_*.npz --bc-epochs 30 --bc-ask-weight balanced \
  --output checkpoints/experiment_8/RequestDQN/best.pt
```

Evaluate:

```bash
.venv/bin/python -m scripts.evaluation.evaluate_request_dqn \
  --experiment 8 --episodes 250 --seed 1900000 --scenario paired-intent --policies dqn \
  --dqn-checkpoint checkpoints/experiment_8/RequestDQN/best.pt --mlpx-checkpoint $WM \
  --ask-margin 0.5 --merge-brake --safety-guard --max-target-speed 30 --lane-change-guard \
  --comfort-ttc 6 --batched-planner --output-dir results/experiment_8/phase2/evaluation/fresh_m0.5_0
```

### Compare

```bash
.venv/bin/python -m scripts.evaluation.compare_request_runs \
  --a results/experiment_8/evaluation/fresh_exp7_{0,1,2,3}/request_dqn_evaluation.csv --a-policy dqn \
  --b results/experiment_8/evaluation/fresh_exp8_{0,1,2,3}/request_dqn_evaluation.csv --b-policy dqn
```

```bash
.venv/bin/python -m scripts.evaluation.compare_request_runs \
  --a results/experiment_8/evaluation/fresh_exp8_{0,1,2,3}/request_dqn_evaluation.csv --a-policy dqn \
  --b results/experiment_8/phase2/evaluation/fresh_m0.5_{0,1,2,3}/request_dqn_evaluation.csv --b-policy dqn
```

## Artifacts

- Labels: `data/experiment_8/dagger_r{1,2,3}_*.npz` (phase 1), `data/experiment_8/phase2_r{0,1}_*.npz` (phase 2)
- Policies: `checkpoints/experiment_8/dagger_r*_s*/best.pt` and `phase1/best.pt` (phase 1); `checkpoints/experiment_8/phase2/r*_s*/best.pt` (phase 2); final: `checkpoints/experiment_8/RequestDQN/best.pt`
- Evaluations: `results/experiment_8/evaluation/` (phase 1), `results/experiment_8/phase2/evaluation/` (phase 2)
