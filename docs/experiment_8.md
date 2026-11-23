# Experiment 8: DAgger and Ask Margin

**Experiment 8** improves the Experiment 7 request DQN with two changes:

1. **DAgger:** the DQN drives, and the oracle labels the road states that the DQN visits. The DQN then trains again on all labels.
2. **Ask margin:** the DQN asks only if its best ask score is more than a margin above the ask-nobody score.

- **Request DQN Checkpoint:** `checkpoints/experiment_8/RequestDQN/best.pt` (DAgger round 2, seed 0)
- **Ask margin:** `--ask-margin 0.5` (pass it at evaluation; it is not stored in the checkpoint)
- **World Model Used:** `checkpoints/experiment_5/MLPx/intent_stage_best.pt` (the Experiment 6 and 7 world model)
- **Configuration:** the same as Experiments 6 and 7: `--safety-guard --max-target-speed 30 --lane-change-guard --batched-planner`, with `--comfort-ttc 6` for cut-in scenes, and 40-step ordinary roads.

**Result:** on 2,000 fresh cut-in trials, Experiment 8 has 65 crashes against 74 for Experiment 7 (McNemar p = 0.012). It also has a higher return (+0.022) and asks about half as often. All three training seeds give the same result. On ordinary roads, Experiment 8 is the same as Experiment 7.

---

## Why

Experiment 7 learns only from the states that the oracle drives into. When the DQN drives, it reaches other states, where it has no labels. DAgger adds labels for those states.

Experiment 7 also asks too often on ordinary roads (about 5 requests per 40-step episode). The ask margin removes the requests where the DQN is not sure.

---

## Method

### DAgger rounds

| Round | Driver | New label seeds | New labels (intent / ordinary steps) |
| :--- | :--- | :--- | :--- |
| 0 | Oracle (Experiment 7 labels) | Intent 1100000+, ordinary 1000000+ | 5,967 / 4,000 |
| 1 | Experiment 7 DQN | Intent 1700000-1700999, ordinary 1000100-1000199 | 5,925 / 4,000 |
| 2 | Round 1 DQN | Intent 1701000-1701999, ordinary 1000200-1000299 | 5,951 / 3,900 |
| 3 | Round 2 DQN | Intent 1702000-1702999, ordinary 1000300-1000399 | 5,943 / 4,000 |

Ordinary episodes have 40 steps. In round 2, the round 1 DQN (no margin) crashed in 3 of the 100 ordinary label episodes (seeds 1000203, 1000205, 1000271), so those episodes are shorter. The oracle labels these states too. The final Experiment 8 policy has no crash in its 600 ordinary test episodes (3 seeds, 2 controllers, 100 episodes each). Each round trains a new DQN from the start on all labels so far, with the Experiment 7 recipe (behaviour cloning, 30 epochs, balanced ask weight). Round 3 was not better than round 2 on validation, so Experiment 8 uses round 2.

Code: `--drive-checkpoint` in `scripts/data_collection/collect_oracle_requests.py`.

### Ask margin

The DQN scores each request option. Without a margin it picks the highest score. With `--ask-margin m`, it asks somebody only if the best ask score is more than `m` above the ask-nobody score.

Code: `select_action(..., ask_margin=m)` in `scripts/training/train_request_dqn.py` and `--ask-margin` in `scripts/evaluation/evaluate_request_dqn.py`.

### Seeds

| Use | Seeds |
| :--- | :--- |
| Labels | See the DAgger table |
| Validation, small | Intent 1400000-1400099, ordinary 1250000-1250059 |
| Validation, large | Intent 1800000-1800999 (2,000 trials) |
| Test, fresh | Intent 1900000-1900999 (2,000 trials) |
| Test, Experiment 7 sets | Intent 1300000, 1500000, 1600000-1600999; ordinary 1200000-1200099 |

---

## Results

### Oracle upper limit (2,000 trials, 1600000)

| Policy | Crashes | Requests / trial | Net return |
| :--- | :---: | :---: | :---: |
| Experiment 7 | 75 | 0.75 | 2.631 |
| Privileged oracle | 42 | 0.16 | 2.709 |

The oracle shows that about 33 more crashes can be prevented with the same world model.

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

A margin alone (Experiment 7 with margin 1.0) brings crashes back. DAgger keeps the crashes low at larger margins.

The first pick was round 2 with margin 1.0. On the Experiment 7 test set it had **more** crashes than Experiment 7 (87 against 75 of 2,000, p = 0.008). The 200-trial validation set was too small to show this. So the margin was chosen again on a large validation set, and the final test uses a fresh test set:

Large validation (2,000 trials, 1800000):

| Policy | Margin | Crashes | Net return | Requests / trial |
| :--- | ---: | ---: | ---: | ---: |
| Experiment 7 | 0 | 77 | 2.627 | 0.75 |
| DAgger round 2 | 0 | 61 | 2.635 | 0.61 |
| DAgger round 2 | **0.5** | 68 | 2.646 | 0.42 |
| DAgger round 2 | 1.0 | 81 | 2.652 | 0.27 |

Margin 0.5 is the only setting that beats Experiment 7 on crashes, intent return and ordinary return on validation. Margin 0 has fewer crashes but a lower ordinary return than Experiment 7 (36.377 against 36.452).

### Final test: cut-in scenes

| Test set | Policy | Crashes | Net return | Requests / trial | Speed, stay / merge (m/s) |
| :--- | :--- | ---: | ---: | ---: | :---: |
| Fresh, 1900000 (2,000) | Ask nobody | 163 | 2.613 | 0.00 | 27.80 / 26.75 |
| | Experiment 7 | 74 | 2.622 | 0.76 | 27.58 / 26.24 |
| | **Experiment 8** | **65** | **2.645** | **0.41** | 27.54 / 26.20 |
| Experiment 7 set, 1600000 (2,000) | DQN6 | 124 | 2.575 | 0.97 | 27.56 / 26.33 |
| | Experiment 7 | 75 | 2.631 | 0.75 | 27.63 / 26.30 |
| | **Experiment 8** | **72** | **2.648** | **0.41** | 27.60 / 26.24 |
| 1300000 (200) | Experiment 7 | 3 | 2.638 | 0.80 | 27.37 / 26.15 |
| | **Experiment 8** | 3 | **2.663** | **0.47** | 27.44 / 26.18 |
| 1500000 (200) | Experiment 7 | 8 | 2.620 | 0.82 | 27.41 / 26.33 |
| | **Experiment 8** | **7** | **2.654** | **0.46** | 27.58 / 26.32 |

Paired difference, Experiment 8 minus Experiment 7 (95% bootstrap interval; exact McNemar test on crashes):

| Test set | Return | Requests | Crashes only under Exp 7 / only under Exp 8 | McNemar p |
| :--- | :--- | :--- | :---: | ---: |
| Fresh, 1900000 | +0.022 [+0.018, +0.027] | -0.34 | 10 / 1 | 0.012 |
| 1600000 | +0.017 [+0.011, +0.022] | -0.34 | 10 / 7 | 0.63 |
| 1300000 | +0.025 [+0.015, +0.038] | -0.33 | 0 / 0 | 1 |
| 1500000 | +0.034 [+0.017, +0.051] | -0.37 | 1 / 0 | 1 |

Against DQN6 on 1600000: 72 against 124 crashes (p = 3e-12), +0.072 return, -0.56 requests.

Speed is the same as Experiment 7. Experiment 8 does not avoid crashes by driving slower.

### Three training seeds (fresh 2,000 trials, 1900000)

Each seed is compared with the Experiment 7 policy of the same seed.

| Seed | Exp 7 crashes | Exp 8 crashes | Return difference | Request difference | Only Exp 7 / only Exp 8 crash | McNemar p |
| :---: | ---: | ---: | :--- | ---: | :---: | ---: |
| 0 | 74 | **65** | +0.022 [+0.018, +0.027] | -0.34 | 10 / 1 | 0.012 |
| 1 | 64 | **57** | +0.017 [+0.011, +0.024] | -0.13 | 10 / 3 | 0.092 |
| 2 | 66 | **55** | +0.019 [+0.012, +0.026] | -0.19 | 11 / 0 | 0.001 |

All three seeds have fewer crashes, a higher return and fewer requests. Over the three seeds, 31 trials crash only under Experiment 7 and 4 crash only under Experiment 8.

### Final test: ordinary roads (1200000, 100 episodes of 40 steps)

| Controller | Policy | Net return | Requests / episode | Crashes | Speed (m/s) |
| :--- | :--- | ---: | ---: | ---: | ---: |
| Original guard | Ask nobody | **36.566** | 0.00 | 0 | 27.49 |
| | DQN6 | 35.801 | 16.88 | 0 | 27.53 |
| | Experiment 7 | 36.390 | 4.88 | 0 | 27.52 |
| | Experiment 8 | 36.370 | 5.54 | 0 | 27.55 |
| `--comfort-ttc 6` | Ask nobody | **36.391** | 0.00 | 0 | 27.41 |
| | DQN6 | 35.748 | 13.17 | 0 | 27.40 |
| | Experiment 7 | 36.118 | 4.93 | 0 | 27.36 |
| | Experiment 8 | 36.049 | 7.15 | 0 | 27.40 |

Experiment 8 minus Experiment 7: -0.019 [-0.138, +0.087] (original guard) and -0.069 [-0.140, +0.004] (`--comfort-ttc 6`). Both intervals include zero.

Seed means (seeds 0-2): Experiment 8 36.434 and 36.165; Experiment 7 36.367 and 36.104. On ordinary roads the two policies are the same within noise. Ask nobody is still the best there.

---

## Conclusion

- DAgger reduces cut-in crashes: 65 against 74 on 2,000 fresh trials (p = 0.012), and fewer crashes for all three training seeds.
- Experiment 8 has a higher cut-in return on every test set and asks about half as often as Experiment 7, at the same speed.
- The ask margin must be chosen on a large validation set. A margin that is too large (1.0) brings crashes back, and 200 trials cannot show this.
- On ordinary roads, Experiment 8 is the same as Experiment 7. Ask nobody is still the best there. The ask margin did not fix this.
- The oracle has 42 crashes on 2,000 trials, so there is still room to improve.

---

## Commands

Run from the project root. `WM` is `checkpoints/experiment_5/MLPx/intent_stage_best.pt`.

DAgger round 1 labels (round 2: `--drive-checkpoint` of round 1, seeds +1000 and +100). Run the intent part in four parts of 250 scenes and the ordinary part in two parts of 50 episodes:

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

Train round 2 (all Experiment 7 labels plus rounds 1 and 2):

```bash
.venv/bin/python -m scripts.training.train_request_dqn \
  --experiment 8 --episodes 0 --seed 0 --mlpx-checkpoint $WM \
  --scenario-mix mixed --request-state-encoding ego_control_v1 \
  --safety-guard --max-target-speed 30 --lane-change-guard --batched-planner \
  --oracle-demos data/experiment_7/stage_oracle_requests_*.npz \
    data/experiment_8/dagger_r1_*.npz data/experiment_8/dagger_r2_*.npz \
  --bc-epochs 30 --bc-ask-weight balanced \
  --output checkpoints/experiment_8/RequestDQN/best.pt
```

Evaluate (fresh test, first quarter; run `--seed 1900250`, `1900500` and `1900750` for the rest):

```bash
.venv/bin/python -m scripts.evaluation.evaluate_request_dqn \
  --experiment 8 --episodes 250 --seed 1900000 --scenario paired-intent --policies dqn \
  --dqn-checkpoint checkpoints/experiment_8/RequestDQN/best.pt --mlpx-checkpoint $WM \
  --ask-margin 0.5 --safety-guard --max-target-speed 30 --lane-change-guard \
  --comfort-ttc 6 --batched-planner --output-dir results/experiment_8/evaluation/fresh_exp8_0
```

For Experiment 7, use `--dqn-checkpoint checkpoints/experiment_7/RequestDQN/best.pt` and `--ask-margin 0`. For ordinary roads, use `--seed 1200000 --episodes 100`, and remove `--scenario paired-intent` and `--comfort-ttc 6`.

Compare:

```bash
.venv/bin/python -m scripts.evaluation.compare_request_runs \
  --a results/experiment_8/evaluation/fresh_exp7_{0,1,2,3}/request_dqn_evaluation.csv --a-policy dqn \
  --b results/experiment_8/evaluation/fresh_exp8_{0,1,2,3}/request_dqn_evaluation.csv --b-policy dqn
```

## Artifacts

- Labels: `data/experiment_8/dagger_r{1,2,3}_*.npz`
- Policies: `checkpoints/experiment_8/dagger_r{1,2,3}_s0/best.pt`, `dagger_r2_s{1,2}/best.pt`; final: `checkpoints/experiment_8/RequestDQN/best.pt`
- Evaluations: `results/experiment_8/evaluation/`
