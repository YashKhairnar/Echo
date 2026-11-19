# MLPx world model

*Experiment 1 · Baseline world-model training*

MLPx is the learned one-step world model. It predicts what happens after the ego takes an action.

## Experiment status

The metrics on this page are from the **first MLPx training experiment** on the
initial merged dataset of 26,597 transitions. The checkpoint is a working
baseline for the planner, not the final world model.

The next MLPx training phase will use more episodes and more varied traffic,
then compare the new checkpoint against this baseline using the same held-out
episodes and three-/five-step rollout tests.

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

## Code

The implementation is in `src/cooperative_driving/models/MLPx/mlpx.py`. Its main methods are `forward`, `predict_step`, `predict_batch`, and `load`.
