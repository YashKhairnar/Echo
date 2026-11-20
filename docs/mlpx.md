# MLPx world model

*Experiment 1 · Baseline world-model training*

MLPx is the learned one-step world model. It predicts what happens after the ego takes an action.

## Experiment status

The first MLPx training experiment used the initial merged dataset of 26,597
transitions. Its checkpoint remains the baseline for planner comparisons.

Experiment 2 is now the larger training run: it uses 223,375 merged
transitions from 2,000 source episodes and is evaluated with the same
three-/five-step rollout procedure.

## Input

The model receives 116 values:

| Input | Size | Meaning |
| --- | ---: | --- |
| Current state | 55 | 11 observation rows × 5 kinematic features |
| Ego action | 5 | One-hot action: `LANE_LEFT`, `IDLE`, `LANE_RIGHT`, `FASTER`, `SLOWER` |
| Messages | 56 | 8 spatial slots × 7 message features |

Message slots use this fixed order:

```text
left-front, left-beside, left-rear,
same-lane-front, same-lane-rear,
right-front, right-beside, right-rear
```

Each message contains presence, behavior type, target lane, target speed, acceleration, steering, and the vehicle crash flag. Unrequested or missing slots are zero-padded.

## Architecture

```text
116 inputs
  -> Linear(256) -> ReLU
  -> Linear(256) -> ReLU
  -> Linear(128) -> ReLU
  -> next-state head: 55 values
  -> reward head: 1 value
```

The model intentionally has no crash head. Crashes remain available as dataset metadata and can be analyzed separately.

## Evaluation against real rollouts

MLPx was compared with real HighwayEnv rollouts from the same sampled states.
For each state, the evaluator tested all five ego actions for a fixed horizon.

- **State MAE** measures the difference between predicted and real states.
- **Reward MAE** measures the difference between predicted and real rollout
  reward; lower is better.
- **Action-ranking agreement** measures how often MLPx selected the same best
  action as the real environment.

| Horizon | Message mode | State MAE | Reward MAE | Match with real best action |
| ---: | --- | ---: | ---: | ---: |
| 3 steps | No messages | `0.229` | `0.336` | `66.7%` (`8/12`) |
| 3 steps | All visible messages | `0.245` | `0.383` | `42.9%` (`6/14`) |
| 5 steps | No messages | `0.238` | `0.723` | `23.1%` (`3/13`) |
| 5 steps | All visible messages | `0.377` | `0.635` | `23.1%` (`3/13`) |

The three-step rollout is substantially better at matching real reward:
reward error is `0.336–0.383`, compared with `0.635–0.723` at five steps.
It also matches the real environment's preferred action more often. This is
why the active planner uses a three-step horizon.

## Experiment 2 · Larger dataset training and evaluation

**Experiment 2** trained MLPx for 50 epochs on the merged dataset from
`data/experiment_2/combined_world_model_dataset.jsonl`:

- 223,375 transitions from 2,000 source episodes.
- The dataset combines 197,553 intent-aware records with 25,822 random
  no-message records.
- The best checkpoint is saved at
  `checkpoints/experiment_2/MLPx/best.pt`.
- Evaluation used 10 episodes and up to three sampled states per episode. Some
  five-step cases ended early, so the number of valid five-step samples is
  smaller.

| Horizon | Message mode | Samples | State MAE | Reward MAE | Match with real best action |
| ---: | --- | ---: | ---: | ---: | ---: |
| 3 steps | No messages | 29 | `0.0213` | `0.0422` | `96.55%` |
| 3 steps | All visible messages | 29 | `0.0574` | `0.1777` | `68.97%` |
| 5 steps | No messages | 11 | `0.0385` | `0.1603` | `54.55%` |
| 5 steps | All visible messages | 11 | `0.0749` | `0.2336` | `63.64%` |

### What Experiment 2 tells us

The model learns short-term public vehicle dynamics well: the three-step
no-message rollout has low state and reward error and agrees with the real
environment's preferred action on 28 of 29 valid samples. Prediction quality
degrades over five steps as errors accumulate. Message-conditioned rollouts
are also harder to predict than no-message rollouts, especially at three steps,
which suggests that the message representation or its training coverage still
needs improvement.

These results support using a three-step planning horizon for now. They do not
yet show that requesting messages improves planning; they show that the
current checkpoint models the no-message behavior more accurately. The
five-step comparison is preliminary because it contains only 11 valid samples
per message mode.

The Experiment 2 evaluation artifacts are saved under
`results/experiment_2/evaluation/mlpx_rollout/`.

Run the evaluation for a selected experiment with:

```bash
.venv/bin/python -m scripts.evaluation.evaluate_mlpx_rollout \
  --experiment 1 \
  --episodes 10 \
  --states-per-episode 3
```

The CSV and JSON summary are saved under
`results/experiment_1/evaluation/mlpx_rollout/`.

## Code

The implementation is in `src/cooperative_driving/models/MLPx/mlpx.py`. Its main methods are `forward`, `predict_step`, `predict_batch`, and `load`.
