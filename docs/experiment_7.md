# Experiment 7: Oracle-Taught Request Policy

Experiment 7 teaches the request DQN from a privileged oracle. The oracle
knows the real future, so it knows when a message changes the outcome.

**Result:** on 2000 new cut-in trials, the oracle-taught DQN with a balanced
ask weight has 75 crashes, against 124 for DQN6 and 179 for ask nobody. It
also has the highest cost-adjusted return and asks less than DQN6. It is not
slower than DQN6. On ordinary roads (40-step episodes, the Experiment 6
configuration), it asks about a third as often as DQN6 and has a higher
cost-adjusted return than DQN6, but ask nobody still has the best
cost-adjusted return there.

## Method

### The oracle

For each scene, the oracle tries each request option: ask nobody, or ask one
visible vehicle. For each option it does these steps:

1. Get the message that this option gives.
2. Let the 3-step planner choose the driving action with that message.
3. Drive a copy of the real road with that action. Then drive blind for the
   rest of a 5-step look-ahead.
4. Add the real rewards and subtract the request cost (0.05).

The oracle picks the best option. A tie with ask nobody picks ask nobody. The
oracle uses the real simulator future, so it is a teacher and an upper
reference, not a policy that a car can use.

Code: `scripts/policy/request_oracle.py`.

### Labels

The ego drives with the oracle choice. Each step stores the public request
state, the oracle value of each option and the oracle choice.

| Label set | Steps | Oracle ask rate |
| :--- | ---: | ---: |
| Intent scenes, seeds 1100000-1100999 (stay and merge) | 5,967 | 5.8% |
| Ordinary roads, seeds 1000000-1000039 | 4,000 | 3.9% |

The labels depend on the world model, because the planner chooses the
driving action. The table shows the labels for `intent_stage_best.pt`.

### Training

- **Behaviour cloning (BC):** cross-entropy between the masked request scores
  and the oracle choice. Ask labels are rare, so `--bc-ask-weight` scales
  their loss. `balanced` uses inverse class frequency (19.85 here).
- **DQfD:** oracle records in every DQN update, with a TD loss and a
  large-margin loss. This did not help (see below), so the final policies use
  BC only.

## Setup

- World model: `checkpoints/experiment_5/MLPx/intent_stage_best.pt` (the
  Experiment 6 world model).
- Controller: safety guard, lane-change guard, speed cap 30 m/s, smooth
  controller `--comfort-ttc 6`, batched planner.
- Request cost: 0.05 per request.
- Seeds:

| Use | Seeds |
| :--- | :--- |
| Labels | Ordinary 1000000+, intent 1100000+ |
| Model selection (validation) | Intent 1400000-1400099 |
| Test (Experiment 6 seeds) | Ordinary 1200000-1200099, intent 1300000-1300099 |
| Fresh confirmation | Intent 1500000-1500099 |
| Large comparison | Intent 1600000-1600999 |

## Results

Each intent scene runs with the neighbour staying and with the neighbour
merging, so 100 scenes give 200 trials.

### DQN6 reproduction

The Experiment 6 checkpoint (`checkpoints/experiment_6/RequestDQN/best.pt`)
gives 2.594 return, 1.04 requests per trial and 9/200 crashes on the
Experiment 6 test seeds. This is the published result, so the comparison
below uses the right DQN6 and settings.

### Large comparison: 2000 new trials (1600000-1600999)

| Policy | Return | Requests / trial | Crashes | Speed, stay (m/s) | Speed, merge (m/s) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Ask nobody | 2.616 | 0.00 | 179 (8.95%) | 27.79 | 26.84 |
| DQN6 | 2.575 | 0.97 | 124 (6.20%) | 27.56 | 26.33 |
| Experiment 7 (balanced) | **2.631** | 0.75 | **75 (3.75%)** | 27.63 | 26.30 |

Paired results (95% bootstrap interval over scenes, exact McNemar test on
crashes):

| Comparison | Return | Crash rate | Requests | Crashes only under A / only under B | McNemar p |
| :--- | :--- | :--- | :--- | :--- | ---: |
| Exp 7 (B) vs DQN6 (A) | +0.056 [+0.047, +0.065] | -2.45 pp [-3.15, -1.80] | -0.23 [-0.28, -0.17] | 51 / 2 | 3.2e-13 |
| Exp 7 (B) vs ask nobody (A) | +0.015 [+0.003, +0.028] | -5.2 pp [-6.2, -4.3] | +0.75 | 104 / 0 | 9.9e-32 |
| DQN6 (B) vs ask nobody (A) | -0.040 [-0.051, -0.030] | -2.75 pp [-3.45, -2.05] | +0.97 | 55 / 0 | 5.6e-17 |

Speed: Experiment 7 and DQN6 have the same speed when the neighbour merges.
When the neighbour stays, Experiment 7 is a little faster than DQN6. Both are
slower than ask nobody, mostly in the merge scenes, where ask nobody crashes.
The request DQN only chooses whom to ask; the planner and the guard choose the
driving action.

### Three training seeds against DQN6

The balanced policy was trained three times (`--seed 0, 1, 2`, same labels).

| Seeds | Policy | Return | Requests / trial | Crashes / 200 | McNemar p vs DQN6 |
| :--- | :--- | ---: | ---: | ---: | ---: |
| 1300000 | DQN6 | 2.594 | 1.04 | 9 | - |
| 1300000 | Exp 7, seed 0 / 1 / 2 | 2.638 / 2.660 / 2.650 | 0.80 / 0.51 / 0.67 | 3 / 3 / 3 | 0.031 each |
| 1500000 | DQN6 | 2.580 | 1.01 | 12 | - |
| 1500000 | Exp 7, seed 0 / 1 / 2 | 2.620 / 2.636 / 2.620 | 0.82 / 0.62 / 0.78 | 8 / 5 / 7 | 0.22 / 0.039 / 0.125 |

Every seed has fewer crashes than DQN6 and a higher return (+0.040 to +0.066;
every interval is above zero). On 200 trials some crash differences are not
significant; the 2000-trial comparison settles them.

### Intent scenes, Experiment 6 test seeds (1300000)

| Policy | Return | Requests / trial | Crashes | Oracle agreement |
| :--- | ---: | ---: | ---: | ---: |
| DQN6 (published and reproduced) | 2.594 | 1.04 | 9 / 200 | n/a |
| Ask nobody (published) | 2.643 | 0.00 | 14 / 200 | n/a |
| Ask nobody (measured) | 2.640 | 0.00 | 14 / 200 | 93.5% |
| BC, ask weight 4 | **2.655** | 0.07 | 12 / 200 | 93.6% |
| BC, balanced ask weight | 2.638 | 0.80 | **3 / 200** | 75.0% |
| Oracle | 2.718 | 0.17 | 3 / 200 | 100% |

The measured ask-nobody row matches the published row (2.640 against 2.643,
14/200 in both). This verifies that the setup matches Experiment 6.

Agreement on the steps where the oracle asks: 18.4% (weight 4) and 52.8%
(balanced). Agreement on the steps where the oracle does not ask: 98.7% and
76.5%.

### Intent scenes, fresh seeds (1500000)

| Policy | Return | Requests / trial | Crashes |
| :--- | ---: | ---: | ---: |
| DQN6 | 2.580 | 1.01 | 12 / 200 |
| Ask nobody | 2.611 | 0.00 | 18 / 200 |
| BC, ask weight 4 | **2.644** | 0.10 | 13 / 200 |
| BC, balanced ask weight | 2.620 | 0.82 | **8 / 200** |

### Paired difference against ask nobody (95% bootstrap interval over scenes)

| Seeds | Policy | Return | Crash rate |
| :--- | :--- | :--- | :--- |
| 1300000 | BC, weight 4 | +0.015 [0.000, +0.040] | -1.0 pp [-2.5, 0.0] |
| 1300000 | BC, balanced | -0.002 [-0.032, +0.032] | **-5.5 pp [-8.5, -2.5]** |
| 1500000 | BC, weight 4 | **+0.033 [+0.007, +0.065]** | **-2.5 pp [-5.0, -0.5]** |
| 1500000 | BC, balanced | +0.009 [-0.028, +0.050] | **-5.0 pp [-8.0, -2.5]** |
| 1300000 | DQN6 (published) | -0.046 [-0.078, -0.012] | -2.5 pp |

### Ordinary roads (Experiment 6 configuration)

Seed 1200000, 100 episodes of 40 steps, `--safety-guard --max-target-speed 30
--lane-change-guard --batched-planner`, world model `intent_stage_best.pt`.
Experiment 6 reports two controller settings, so both are shown. The DQN6 and
ask-nobody rows reproduce the published Experiment 6 numbers exactly.

| Controller | Policy | Adjusted return | Driving reward | Requests / episode | Crashes | Speed (m/s) | Lane changes |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original guard | Ask nobody | **36.566** | 36.566 | 0.00 | 0 / 100 | 27.491 | 180 |
| Original guard | DQN6 | 35.801 | 36.645 | 16.88 | 0 / 100 | 27.527 | 171 |
| Original guard | Experiment 7, seed 0 | 36.390 | 36.634 | 4.88 | 0 / 100 | 27.521 | 172 |
| Original guard | Experiment 7, seeds 0-2 mean (range) | 36.367 (36.341-36.390) | 36.636 | 5.38 (4.88-5.77) | 0 / 100 each | 27.527 | 172.7 |
| `--comfort-ttc 6` | Ask nobody | **36.391** | 36.391 | 0.00 | 0 / 100 | 27.414 | 252 |
| `--comfort-ttc 6` | DQN6 | 35.748 | 36.407 | 13.17 | 0 / 100 | 27.401 | 240 |
| `--comfort-ttc 6` | Experiment 7, seed 0 | 36.118 | 36.364 | 4.93 | 0 / 100 | 27.360 | 241 |
| `--comfort-ttc 6` | Experiment 7, seeds 0-2 mean (range) | 36.104 (36.039-36.154) | 36.378 | 5.50 (4.39-7.17) | 0 / 100 each | 27.377 | 239.3 |

Paired adjusted-return difference for Experiment 7 seed 0 (95% bootstrap
interval over episodes):

| Controller | vs DQN6 | vs ask nobody |
| :--- | :--- | :--- |
| Original guard | **+0.589 [+0.420, +0.750]**, 12.0 fewer requests | -0.177 [-0.270, -0.093] |
| `--comfort-ttc 6` | **+0.370 [+0.274, +0.470]**, 8.2 fewer requests | -0.273 [-0.357, -0.196] |

On ordinary roads Experiment 7 drives like DQN6: the same driving reward,
speed and lane changes, and no crashes. It asks about a third as often, so
its cost-adjusted return is higher than DQN6. Ask nobody still has the best
cost-adjusted return, because no request on these roads prevents a crash.

### Model selection (validation seeds 1400000)

| Policy | Return | Requests / trial | Crashes |
| :--- | ---: | ---: | ---: |
| Ask nobody | 2.626 | 0.00 | 15 / 200 |
| BC, ask weight 1 | 2.626 | 0.00 | 15 / 200 |
| BC, ask weight 4 | **2.673** | 0.07 | 9 / 200 |
| BC, balanced ask weight | 2.642 | 0.82 | **4 / 200** |

Weight 1 learns to never ask. Weight 4 has the best validation return, so it
was the first pick. The balanced weight was tested because it has the fewest
crashes, and the fresh seeds confirm its crash result.

### Earlier runs with `best.pt` (not the Experiment 6 world model)

| Policy (intent seeds 1300000) | Return | Requests / trial | Crashes | Oracle agreement |
| :--- | ---: | ---: | ---: | ---: |
| Ask nobody | 2.616 | 0.00 | 11 / 200 | 89.7% |
| Old method (Experiment 6 recipe, retrained) | 2.551 | 1.67 | 8 / 200 | 44.2% |
| BC, ask weight 1 | 2.632 | 0.03 | 11 / 200 | 90.7% |
| BC + DQfD (1000 episodes) | 2.629 | 0.20 | 13 / 200 | 88.6% |
| Oracle | 2.695 | 0.28 | 6 / 200 | 100% |

BC + DQfD was not better than BC alone, so it was not repeated with
`intent_stage_best.pt`.

## Videos

The ordinary-road video uses the Experiment 6 command (seed 1200000). It has
100 steps, because the `exp 6` commit changed `duration` from 40 to 100. The
balanced DQN drives to the end without a crash (reward 90.261, 22 requests,
mean speed 27.14 m/s). Ask nobody drives the same actions on this road. The
25-30 m/s speed cycle comes from the guard, not from the request policy.

![Experiment 7 ordinary road](images/dqn_experiment_7_policy.mp4)

Cut-in scenes (`--scenario paired-intent --hidden-mode merge`, two frames per
second). In each pair the scene, the planner and the guard are the same; only
the request policy differs.

| Scene | Ask nobody | DQN6 | Experiment 7 |
| :--- | :--- | :--- | :--- |
| 1300051 | n/a | [FASTER, crash at step 2](images/dqn_experiment_6_intent_1300051_merge.mp4) | [asks, SLOWER, safe](images/dqn_experiment_7_intent_1300051_merge_dqn.mp4) |
| 1300005 | [FASTER, crash at step 2](images/dqn_experiment_7_intent_1300005_merge_none.mp4) | [asks, SLOWER, safe](images/dqn_experiment_6_intent_1300005_merge.mp4) | [asks, SLOWER, safe](images/dqn_experiment_7_intent_1300005_merge_dqn.mp4) |
| 1300061 | [FASTER, crash at step 3](images/dqn_experiment_7_intent_1300061_merge_none.mp4) | n/a | [asks, IDLE, safe](images/dqn_experiment_7_intent_1300061_merge_dqn.mp4) |
| 1300074 | [FASTER, crash at step 1](images/dqn_experiment_7_intent_1300074_merge_none.mp4) | n/a | [asks, IDLE, safe](images/dqn_experiment_7_intent_1300074_merge_dqn.mp4) |
| 1300000 (both safe) | [FASTER, safe](images/dqn_experiment_7_intent_1300000_merge_none.mp4) | n/a | [asks, FASTER, safe](images/dqn_experiment_7_intent_1300000_merge_dqn.mp4) |

In 5 of the 6 test scenes where DQN6 crashes and Experiment 7 does not, DQN6
asks nobody at the first step and speeds up into the merging car.

## Conclusion

- On 2000 new cut-in trials, the balanced oracle-taught DQN has 40% fewer
  crashes than DQN6 (75 against 124, McNemar p = 3e-13), a higher return
  (+0.056) and fewer requests (-0.23 per trial), at the same speed.
- Three training seeds give the same result on two more seed sets.
- Against ask nobody, it cuts cut-in crashes by about 5 percentage points,
  and its cut-in return is a little higher.
- On ordinary roads (40 steps) it beats DQN6 by +0.59 (original guard) and
  +0.37 (`--comfort-ttc 6`) cost-adjusted return, with about a third of the
  requests. It is still 0.18 and 0.27 below ask nobody there, and no policy
  crashes.
- The cut-in results use their own 3-step episodes, so the 40-step change
  does not affect them (spot check on 1300000: Experiment 7 3/200, DQN6
  9/200, unchanged).

## Commands

Run from the project root. `WM` is
`checkpoints/experiment_5/MLPx/intent_stage_best.pt`. `GUARD` is
`--safety-guard --max-target-speed 30 --lane-change-guard`.

```bash
.venv/bin/python -m scripts.data_collection.collect_oracle_requests \
  --experiment 7 --ordinary-episodes 0 --intent-scenes 1000 \
  --mlpx-checkpoint checkpoints/experiment_5/MLPx/intent_stage_best.pt \
  --safety-guard --lane-change-guard --max-target-speed 30 \
  --output data/experiment_7/stage_oracle_requests_intent.npz
```

```bash
.venv/bin/python -m scripts.data_collection.collect_oracle_requests \
  --experiment 7 --ordinary-episodes 20 --ordinary-seed 1000000 --intent-scenes 0 \
  --mlpx-checkpoint checkpoints/experiment_5/MLPx/intent_stage_best.pt \
  --safety-guard --lane-change-guard --max-target-speed 30 \
  --output data/experiment_7/stage_oracle_requests_ordinary_a.npz
```

Run the same command with `--ordinary-seed 1000020` and the output
`stage_oracle_requests_ordinary_b.npz`.

```bash
.venv/bin/python -m scripts.training.train_request_dqn \
  --experiment 7 --episodes 0 --seed 0 \
  --mlpx-checkpoint checkpoints/experiment_5/MLPx/intent_stage_best.pt \
  --scenario-mix mixed --request-state-encoding ego_control_v1 \
  --safety-guard --max-target-speed 30 --lane-change-guard --batched-planner \
  --oracle-demos data/experiment_7/stage_oracle_requests_intent.npz \
    data/experiment_7/stage_oracle_requests_ordinary_a.npz \
    data/experiment_7/stage_oracle_requests_ordinary_b.npz \
  --bc-epochs 30 --bc-ask-weight balanced \
  --output checkpoints/experiment_7/stage_bc_balanced/best.pt
```

Use `--bc-ask-weight 4` and `stage_bc_w4` for the weight-4 policy.

```bash
.venv/bin/python -m scripts.evaluation.evaluate_request_dqn \
  --experiment 7 --episodes 100 --seed 1300000 --scenario paired-intent \
  --policies dqn none oracle --oracle-agreement \
  --dqn-checkpoint checkpoints/experiment_7/stage_bc_balanced/best.pt \
  --mlpx-checkpoint checkpoints/experiment_5/MLPx/intent_stage_best.pt \
  --safety-guard --max-target-speed 30 --lane-change-guard --comfort-ttc 6 \
  --batched-planner --output-dir results/experiment_7/evaluation/stage_intent_bc_balanced
```

Use `--seed 1500000` for the fresh seeds and `--seed 1400000` for validation.
For ordinary roads (40-step episodes, the default since commit 96d38e7):

```bash
.venv/bin/python -m scripts.evaluation.evaluate_request_dqn \
  --experiment 7 --episodes 100 --seed 1200000 --policies dqn \
  --dqn-checkpoint checkpoints/experiment_7/stage_bc_balanced/best.pt \
  --mlpx-checkpoint checkpoints/experiment_5/MLPx/intent_stage_best.pt \
  --safety-guard --max-target-speed 30 --lane-change-guard --batched-planner \
  --output-dir results/experiment_7/evaluation/ord40_exp7_s0_guard
```

Add `--comfort-ttc 6` for the smooth controller. For DQN6 and ask nobody, use
`--policies dqn none` and
`--dqn-checkpoint checkpoints/experiment_6/RequestDQN/best.pt`.

```bash
.venv/bin/python -m scripts.evaluation.compare_request_runs \
  --a results/experiment_7/evaluation/stage_intent_reference/request_dqn_evaluation.csv --a-policy none \
  --b results/experiment_7/evaluation/stage_intent_bc_balanced/request_dqn_evaluation.csv --b-policy dqn
```

Large comparison: run each policy in two parts of 500 scenes (`--seed 1600000`
and `--seed 1600500`). For DQN6, use
`--dqn-checkpoint checkpoints/experiment_6/RequestDQN/best.pt` and the output
directories `dqn6_big_a` and `dqn6_big_b`.

```bash
.venv/bin/python -m scripts.evaluation.evaluate_request_dqn \
  --experiment 7 --episodes 500 --seed 1600000 --scenario paired-intent \
  --policies dqn none \
  --dqn-checkpoint checkpoints/experiment_7/stage_bc_balanced/best.pt \
  --mlpx-checkpoint checkpoints/experiment_5/MLPx/intent_stage_best.pt \
  --safety-guard --max-target-speed 30 --lane-change-guard --comfort-ttc 6 \
  --batched-planner --output-dir results/experiment_7/evaluation/big_intent_a
```

```bash
.venv/bin/python -m scripts.evaluation.compare_request_runs \
  --a results/experiment_7/evaluation/dqn6_big_a/request_dqn_evaluation.csv \
      results/experiment_7/evaluation/dqn6_big_b/request_dqn_evaluation.csv --a-policy dqn \
  --b results/experiment_7/evaluation/big_intent_a/request_dqn_evaluation.csv \
      results/experiment_7/evaluation/big_intent_b/request_dqn_evaluation.csv --b-policy dqn
```

Other training seeds: run the training command with `--seed 1` or `--seed 2`
and the output `stage_bc_balanced_seed1` or `stage_bc_balanced_seed2`.

## Artifacts

- Labels: `data/experiment_7/stage_oracle_requests_*.npz`
- Policies: `checkpoints/experiment_7/stage_bc_balanced/best.pt` (and
  `_seed1`, `_seed2`), `checkpoints/experiment_7/stage_bc_w4/best.pt`
- Evaluations: `results/experiment_7/evaluation/`
