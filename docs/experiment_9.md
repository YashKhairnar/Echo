# Experiment 9: Message-Aware Brake Rule

**Experiment 9** adds a simple rule to the controller. If a car that the ego asked says that it is moving into the ego lane, the ego treats it as a car in front: it brakes when the car is close, and it never speeds up. The request DQN is then taught again by the oracle with this rule in the controller.

- **Request DQN Checkpoint:** `checkpoints/experiment_9/RequestDQN/best.pt` (round 0, seed 0)
- **Flags:** `--merge-brake --ask-margin 0.5` (neither is stored in the checkpoint)
- **World Model Used:** `checkpoints/experiment_5/MLPx/intent_stage_best.pt` (the Experiment 6, 7 and 8 world model)
- **Configuration:** the same as Experiments 6 to 8: `--safety-guard --max-target-speed 30 --lane-change-guard --batched-planner`, with `--comfort-ttc 6` for cut-in scenes, and 40-step ordinary roads.

**Result:** on 2,000 fresh cut-in trials, Experiment 9 has **2 crashes**, against 65 for Experiment 8 and 163 for ask nobody. It also has a higher return than Experiment 8 (+0.032). On ordinary roads, it is now equal to ask nobody (the best policy there) and better than Experiment 8 (+0.21). In merge scenes it drives 1.1 m/s slower than Experiment 8, because it brakes for the merging car.

---

## Why

A test after Experiment 8 showed that the request policy was not the limit:

| Policy (fresh 2,000 trials, 1900000) | Crashes, stay scenes | Crashes, merge scenes |
| :--- | ---: | ---: |
| Ask nobody | 0 | 163 |
| Ask every visible car | 0 | 154 |
| Always ask the cut-in car (privileged) | 0 | 83 |
| Experiment 8 | 0 | 65 |

Even with the correct "I will merge" message at every step, the planner crashed in 83 of 1,000 merge scenes. The world model and the 3-step planner do not act safely enough on the message. Messages from every car made the prediction worse.

With the brake rule and the privileged "always ask the cut-in car" policy, the validation set (1800000) had **0 crashes in 2,000 trials**. So the rule fixes the planner problem when the right car is asked, and the request policy matters again.

---

## Method

### Brake rule (`--merge-brake`)

After the guard chooses the action, for each requested message:

1. Skip the car if its message target lane is not the ego lane, if it is already in the ego lane (the guard handles it), or if it is fully behind the ego.
2. Compute its clearance and time to collision, as for a front car.
3. If the clearance is below 8 m or the time to collision is below 3 s, brake (SLOWER).
4. Otherwise, if the action is FASTER, change it to IDLE.

The thresholds are the existing guard thresholds (`--min-clearance 8`, `--min-ttc 3`). A sweep on the validation set (clearance 0 to 25 m, time to collision 1.5 to 10 s) gave the same number of crashes for every setting, so no new value was tuned.

With the Experiment 8 policy on the ordinary validation episodes (1250000), the rule fired in no step.

Code: `merging_neighbours` and `merge_brake_action` in `src/cooperative_driving/planner/safety.py`, and `merge_brake` in `src/cooperative_driving/planner/controller.py`.

### Request policy

The oracle uses the same controller, so with `--merge-brake` it sees the new value of asking the merging car. The labels were collected again with the rule, on the Experiment 7 label seeds, and the Experiment 7 recipe was used for training (behaviour cloning, 30 epochs, balanced ask weight).

| Round | Driver | Seeds | Labels (intent / ordinary steps) | Oracle ask rate (intent / ordinary) |
| :--- | :--- | :--- | :--- | :--- |
| 0 | Oracle | Intent 1100000-1100999, ordinary 1000000-1000099 | 6,000 / 4,000 | 8.2% / 3.6% |
| 1 (DAgger) | Round 0 DQN | Intent 2000000-2000999, ordinary 1000500-1000599 | 6,000 / 4,000 | 9.1% / 5.4% |

With the rule, the oracle drove all 2,000 label scenes without a crash. Without the rule (Experiment 7 labels), the oracle asked in 5.8% of intent steps.

---

## Model selection (validation 1800000, 2,000 trials; ordinary 1250000, 60 episodes)

| Policy | Margin | Crashes | Intent return | Intent requests | Ordinary return | Ordinary requests |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Experiment 8 | 0.5 | 68 | 2.646 | 0.42 | 36.534 | 5.2 |
| Experiment 8 + rule | 0.5 | 50 | 2.651 | 0.44 | 36.534 | 5.2 |
| Round 0 | 0 | **0** | 2.665 | 0.87 | 36.615 | 1.7 |
| Round 0 | **0.5** | 2 | 2.688 | 0.55 | 36.677 | 0.7 |
| Round 0 | 1.0 | 23 | 2.690 | 0.38 | 36.675 | 0.2 |
| Round 1 | 0 | 4 | 2.665 | 0.79 | 36.488 | 4.6 |
| Round 1 | 0.5 | 8 | 2.678 | 0.61 | 36.635 | 2.1 |
| Round 1 | 1.0 | 23 | 2.682 | 0.46 | 36.682 | 0.7 |

Round 1 (DAgger) was not better than round 0, so Experiment 9 uses round 0. Before the test, margin 0.5 was chosen as the main policy: it has the best cost-adjusted return on both road types, and its 2 crashes against 0 are within noise. Margin 0 is reported as the safety-first variant. Ask nobody on the ordinary validation seeds: 36.715.

---

## Results

### Cut-in scenes

| Test set | Policy | Crashes | Net return | Requests / trial | Speed, stay / merge (m/s) |
| :--- | :--- | ---: | ---: | ---: | :---: |
| Fresh, 1900000 (2,000) | Ask nobody | 163 | 2.613 | 0.00 | 27.80 / 26.75 |
| | Experiment 8 | 65 | 2.645 | 0.41 | 27.54 / 26.20 |
| | **Experiment 9** | **2** | **2.677** | 0.57 | 27.87 / 25.07 |
| | Experiment 9, margin 0 | **0** | 2.655 | 0.87 | 27.92 / 24.80 |
| Experiment 7 set, 1600000 (2,000) | DQN6 | 124 | 2.575 | 0.97 | 27.56 / 26.33 |
| | Experiment 8 | 72 | 2.648 | 0.41 | 27.60 / 26.24 |
| | **Experiment 9** | **1** | **2.682** | 0.58 | 27.86 / 25.10 |
| | Experiment 9, margin 0 | **0** | 2.660 | 0.87 | n/a |
| 1300000 (200) | Experiment 8 | 3 | 2.663 | 0.47 | 27.44 / 26.18 |
| | **Experiment 9** | **0** | **2.690** | 0.54 | 27.85 / 25.36 |
| 1500000 (200) | Experiment 8 | 7 | 2.654 | 0.46 | 27.58 / 26.32 |
| | **Experiment 9** | **0** | **2.689** | 0.52 | 27.80 / 25.28 |

Paired difference, Experiment 9 minus Experiment 8 (95% bootstrap interval; exact McNemar test on crashes):

| Test set | Return | Requests | Crashes only under Exp 8 / only under Exp 9 | McNemar p |
| :--- | :--- | :--- | :---: | ---: |
| Fresh, 1900000 | +0.032 [+0.022, +0.043] | +0.15 | 63 / 0 | 2e-19 |
| 1600000 | +0.034 [+0.023, +0.044] | +0.17 | 72 / 1 | 2e-20 |
| 1300000 | +0.027 [-0.000, +0.057] | +0.07 | 3 / 0 | 0.25 |
| 1500000 | +0.035 [+0.007, +0.067] | +0.06 | 7 / 0 | 0.016 |

Against DQN6 (1600000): 1 against 124 crashes (p = 2e-37), +0.106 return, -0.39 requests. Against ask nobody (fresh): 2 against 163 crashes, +0.064 return.

**Speed.** In merge scenes, Experiment 9 drives 1.1 m/s slower than Experiment 8 (25.07 against 26.20 m/s), because it brakes for the car that merges. In stay scenes it is faster than every other policy (27.87 against 27.54 m/s for Experiment 8), because it no longer slows down for a car that stays in its lane. The return includes the speed reward, and it is higher than Experiment 8 on every test set.

### Three training seeds (fresh 2,000 trials, 1900000)

Each Experiment 9 seed is compared with the Experiment 8 policy of the same seed.

| Seed | Exp 8 crashes | Exp 9 crashes | Return difference | McNemar p |
| :---: | ---: | ---: | :--- | ---: |
| 0 | 65 | **2** | +0.032 [+0.022, +0.043] | 2e-19 |
| 1 | 57 | **8** | +0.010 [-0.002, +0.022] | 3e-13 |
| 2 | 55 | **1** | +0.023 [+0.012, +0.034] | 1e-16 |

On the 200-trial sets 1300000 and 1500000, all three seeds have 0 crashes (returns 2.671 to 2.690).

### Ordinary roads (1200000, 100 episodes of 40 steps)

| Controller | Policy | Net return | Requests / episode | Crashes | Speed (m/s) |
| :--- | :--- | ---: | ---: | ---: | ---: |
| Original guard | Ask nobody | 36.566 | 0.00 | 0 | 27.49 |
| | DQN6 | 35.801 | 16.88 | 0 | 27.53 |
| | Experiment 8 | 36.370 | 5.54 | 0 | 27.55 |
| | **Experiment 9** | **36.576** | 0.58 | 0 | 27.49 |
| `--comfort-ttc 6` | Ask nobody | **36.391** | 0.00 | 0 | 27.41 |
| | DQN6 | 35.748 | 13.17 | 0 | 27.40 |
| | Experiment 8 | 36.049 | 7.15 | 0 | 27.40 |
| | **Experiment 9** | 36.359 | 0.60 | 0 | 27.40 |

Experiment 9 minus Experiment 8: +0.205 [+0.104, +0.331] (original guard) and +0.310 [+0.230, +0.396] (`--comfort-ttc 6`). Experiment 9 minus ask nobody: +0.009 [-0.028, +0.061] and -0.032 [-0.065, +0.006]; both intervals include zero.

Seed means (seeds 0-2): 36.527 (original guard) and 36.322 (`--comfort-ttc 6`). Margin 0: 36.528 and 36.283.

---

## Video

Two fresh cut-in scenes where Experiment 8 crashes and Experiment 9 does not, then the 2,000-trial results. In each pair the scene, world model, planner and guard are the same; Experiment 9 adds the brake rule and its own request policy.

- **Scene 1900012:** Experiment 8 asks the merging car, speeds up and crashes. Experiment 9 asks, holds its speed, then brakes, and is safe.
- **Scene 1900038:** Experiment 8 asks nobody, speeds up and crashes. Experiment 9 asks the merging car, brakes, and is safe.

![Experiment 8 vs Experiment 9](images/exp8_vs_exp9_comparison.mp4)

---

## Conclusion

- The message-aware brake rule removes almost all cut-in crashes: 2 of 2,000 fresh trials, against 65 for Experiment 8 and 163 for ask nobody. All three training seeds agree.
- The return is higher than Experiment 8 on every test set, and on ordinary roads Experiment 9 is now equal to ask nobody, with about 0.6 requests per episode.
- The gain needs both parts: the rule alone (with the Experiment 8 DQN) gives 50 crashes on validation, because that DQN often does not ask the merging car. The DQN taught with the rule asks the right car.
- In merge scenes, the ego drives 1.1 m/s slower than Experiment 8, because it brakes for the merging car. This is the intended behaviour; in stay scenes it is faster.
- DAgger did not help here (round 1 was worse on validation).
- The rule is hand-written. It uses only the message and public geometry, and it fires only when a requested car announces a move into the ego lane.

---

## Commands

Run from the project root. `WM` is `checkpoints/experiment_5/MLPx/intent_stage_best.pt`.

Labels (run the intent part in four parts of 250 scenes and the ordinary part in two parts of 50 episodes):

```bash
.venv/bin/python -m scripts.data_collection.collect_oracle_requests \
  --experiment 9 --mlpx-checkpoint $WM --merge-brake \
  --safety-guard --lane-change-guard --max-target-speed 30 \
  --ordinary-episodes 0 --intent-scenes 250 --challenge-seed 1100000 \
  --output data/experiment_9/r0_intent_0.npz
```

```bash
.venv/bin/python -m scripts.data_collection.collect_oracle_requests \
  --experiment 9 --mlpx-checkpoint $WM --merge-brake \
  --safety-guard --lane-change-guard --max-target-speed 30 \
  --ordinary-episodes 50 --ordinary-seed 1000000 --intent-scenes 0 \
  --output data/experiment_9/r0_ordinary_0.npz
```

Train:

```bash
.venv/bin/python -m scripts.training.train_request_dqn \
  --experiment 9 --episodes 0 --seed 0 --mlpx-checkpoint $WM \
  --scenario-mix mixed --request-state-encoding ego_control_v1 \
  --safety-guard --max-target-speed 30 --lane-change-guard --batched-planner \
  --oracle-demos data/experiment_9/r0_*.npz --bc-epochs 30 --bc-ask-weight balanced \
  --output checkpoints/experiment_9/RequestDQN/best.pt
```

Evaluate (fresh test, first quarter; run `--seed 1900250`, `1900500` and `1900750` for the rest):

```bash
.venv/bin/python -m scripts.evaluation.evaluate_request_dqn \
  --experiment 9 --episodes 250 --seed 1900000 --scenario paired-intent --policies dqn \
  --dqn-checkpoint checkpoints/experiment_9/RequestDQN/best.pt --mlpx-checkpoint $WM \
  --ask-margin 0.5 --merge-brake --safety-guard --max-target-speed 30 --lane-change-guard \
  --comfort-ttc 6 --batched-planner --output-dir results/experiment_9/evaluation/fresh_m0.5_0
```

For ordinary roads, use `--seed 1200000 --episodes 100`, and remove `--scenario paired-intent` and `--comfort-ttc 6`.

Compare with Experiment 8:

```bash
.venv/bin/python -m scripts.evaluation.compare_request_runs \
  --a results/experiment_8/evaluation/fresh_exp8_{0,1,2,3}/request_dqn_evaluation.csv --a-policy dqn \
  --b results/experiment_9/evaluation/fresh_m0.5_{0,1,2,3}/request_dqn_evaluation.csv --b-policy dqn
```

## Artifacts

- Labels: `data/experiment_9/r{0,1}_*.npz`
- Policies: `checkpoints/experiment_9/r0_s{0,1,2}/best.pt`, `r1_s0/best.pt`; final: `checkpoints/experiment_9/RequestDQN/best.pt`
- Evaluations: `results/experiment_9/evaluation/`
