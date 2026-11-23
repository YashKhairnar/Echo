# Experiment 7: Oracle-Taught Request Policy

Experiment 7 teaches the request DQN from a privileged oracle. The oracle
knows the real future, so it knows when a message changes the outcome.

**Result:** the oracle-taught DQN with a balanced ask weight has 3/200 intent
crashes, against 9/200 for DQN6 and 14/200 for ask nobody. Its intent return
is equal to ask nobody and higher than DQN6. On ordinary roads, ask nobody
still has the best cost-adjusted return.

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

## Results

Each intent row has 100 scenes. Each scene runs with the neighbour staying
and with the neighbour merging, so there are 200 trials.

### Intent scenes, Experiment 6 test seeds (1300000)

| Policy | Return | Requests / trial | Crashes | Oracle agreement |
| :--- | ---: | ---: | ---: | ---: |
| DQN6 (published) | 2.594 | 1.04 | 9 / 200 | n/a |
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

### Ordinary roads (1200000, 100 episodes)

| Policy | Return | Requests / episode | Crashes |
| :--- | ---: | ---: | ---: |
| Ask nobody | **90.333** | 0.00 | 0 / 100 |
| BC, ask weight 4 | 90.210 | 1.02 | 0 / 100 |
| BC, balanced ask weight | 89.741 | 12.67 | 0 / 100 |

Paired difference against ask nobody: -0.12 [-0.24, -0.03] (weight 4) and
-0.59 [-0.84, -0.36] (balanced).

The `exp 6` commit changed the ordinary-road `duration` from 40 to 100
steps. The published Experiment 6 ordinary returns (about 36) use 40 steps,
so they cannot be compared with these rows. DQN6 lost 0.765 against ask
nobody on 40-step roads with 16.88 requests per episode.

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

## Conclusion

- The balanced oracle-taught DQN beats DQN6 on intent crashes (3/200 against
  9/200) and on intent return (2.638 against 2.594).
- Against ask nobody, it cuts intent crashes by about 5 percentage points on
  two seed sets, and the intent return stays the same.
- On ordinary roads it still asks too often (12.7 requests per 100 steps).
  Ask nobody keeps the best cost-adjusted return there, and no policy crashes.
- The weight-4 DQN is the cheaper choice: a small return gain in the intent
  scenes and a small loss on ordinary roads.

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
For ordinary roads, use `--seed 1200000`, remove `--scenario paired-intent`,
`oracle` and `--oracle-agreement`.

```bash
.venv/bin/python -m scripts.evaluation.compare_request_runs \
  --a results/experiment_7/evaluation/stage_intent_reference/request_dqn_evaluation.csv --a-policy none \
  --b results/experiment_7/evaluation/stage_intent_bc_balanced/request_dqn_evaluation.csv --b-policy dqn
```

## Artifacts

- Labels: `data/experiment_7/stage_oracle_requests_*.npz`
- Policies: `checkpoints/experiment_7/stage_bc_balanced/best.pt`,
  `checkpoints/experiment_7/stage_bc_w4/best.pt`
- Evaluations: `results/experiment_7/evaluation/`
