# MLPx world model

MLPx is the learned one-step world model. It predicts what happens after the ego takes an action.

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

## Code

The implementation is in `src/cooperative_driving/models/MLPx/mlpx.py`. Its main methods are `forward`, `predict_step`, `predict_batch`, and `load`.
