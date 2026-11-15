# Cooperative merge scenario

The earlier highway setup shared information the ego could already infer
(current acceleration, time to collision computed from visible gaps), and the
default highway task rarely forced a decision that depended on it. That is why
every request policy learned to ask nobody. This scenario is built so that a
message is **hidden** (not inferable from the ego observation) and **matters**
(knowing it changes the best driving action).

## Layout

`CooperativeMergeEnv` (`src/cooperative_driving/environments/customEnv.py`):

- Two lanes. The ego starts in the lower lane (lane 1) at 25 m/s.
- A **stalled car** blocks the ego lane 250–350 m ahead, so the ego must merge
  into the upper lane (lane 0, the target lane).
- The target lane carries a platoon of nine `DriverTypeIDMVehicle` cars
  (25–45 m apart, ~23–26 m/s, no lane changes).
- Episodes last 20 decisions (1 per second).

## Hidden information: driver type

Each platoon car is secretly `yielding` or `aggressive` (50/50).
Both types drive identically until a car **ahead of them within 40 m** starts
changing into their lane (`src/cooperative_driving/vehicles/customVehicles.py`):

| Driver type | Reaction to a merging car |
| --- | --- |
| yielding | Treats the merging car as its leader and brakes to open a gap |
| aggressive | Ignores it and accelerates (6 m/s²) until it is centred in the lane |

So merging in front of a yielding car is safe, while doing the same in front
of an aggressive car usually ends in a crash. The type is only visible
*after* the ego has committed to the lane change.

## Ego vehicle

`CruiseControlledVehicle`: the discrete meta-actions (`LANE_LEFT`, `IDLE`,
`LANE_RIGHT`, `FASTER`, `SLOWER`) set the lane and desired speed, while a
low-level adaptive cruise control keeps a safe distance to the car ahead. This
removes trivial rear-end crashes, so the remaining risk comes from other
drivers' reactions.

## Reward

Per step (not normalised):

```text
reward = 0.4 * speed_score + 0.6 * in_target_lane     (speed_score: 20→30 m/s mapped to 0→1)
reward = -2                                            on a crash (episode ends)
```

## Message format

`encode_message(car)` in `src/cooperative_driving/communication/getMessage.py`
is the single place that turns a car into the 4-value MLPx message slot:

| Index | Feature | Values |
| ---: | --- | --- |
| 0 | driver type | +1 yielding, −1 aggressive, 0 other (e.g. stalled car) |
| 1 | intent acceleration | current acceleration command [m/s²] |
| 2 | target lane | lane index |
| 3 | time to collision | seconds, capped at 10 |

This replaces intent steering in slot 1 of the previous format. MLPx input
size is unchanged (115).

## Information-gap check

`scripts/check_information_gap.py` runs before any learning:

```bash
.venv/bin/python -m scripts.check_information_gap --episodes 300
```

**Part A: is it hidden?** A logistic-regression classifier predicts each
target-lane car's driver type from the ego's observation of it (relative
position, velocity, velocity change).

**Part B: does it change the outcome?** Hand-written driving rules on the same
seeds:

- `merge_now`: merge immediately, never ask.
- `no_info`: merge only when the rear car stays out of reaction range for the
  whole merge (safe for either type), otherwise slow down and wait.
- `ask_rear`: same, but when the gap is only safe for a yielding driver, ask
  the rear car and merge if it yields.

### Results (300 seeds, 2026-11-15)

Part A:

| Ego observation of the car | Classifier accuracy | Chance |
| --- | ---: | ---: |
| Before the ego merges | 51.4% | 51.4% |
| One step after the merge starts | 90.3% | 53.0% |

The driver type is hidden until the ego has already committed to merging.

Part B:

| Rule | Return | Crash rate | Merged | Mean merge time | Requests / episode |
| --- | ---: | ---: | ---: | ---: | ---: |
| `merge_now` | 5.17 ± 0.41 | 52.0% | 48.0% | 1.0 s | 0 |
| `no_info` | 4.64 ± 0.07 | 0.0% | 100% | 13.4 s | 0 |
| `ask_rear` | **7.35 ± 0.19** | **0.0%** | **100%** | **9.2 s** | 1.17 |

(± is the standard error.) Asking the rear car about once per episode raises
return by about 58% over the best rule without communication. It merges as
safely as the cautious rule and about 4 s sooner. The information gap is real,
so a learned world model and request policy have something to learn here.
Without communication, the ego must either gamble (about 1 in 2 crashes) or
wait for a gap that is safe against any driver.

## Collecting world-model data

Both collectors now take `--env {merge,highway}` (default `merge`) and write
merge data to separate files so it never mixes with the legacy highway data:

```bash
.venv/bin/python -m scripts.collect_multi_request_data --episodes 1000      # data/mergeMultiRequestData.csv
.venv/bin/python -m scripts.collect_matched_message_wm_data --episodes 1000 # data/matchedMergeWMData.csv
```

Driving actions come from `MergeExploration` (`scripts/exploration.py`): each
episode draws a merge step between 0 and 12; before it the ego only changes
speed, at it the ego merges, and afterwards actions are uniform random. This
gives merges from many different gaps and speeds. The matched collector always
includes the merge step among its sampled states and adds a `rear` branch that
requests only the rear target-lane car.

## Not yet switched over

`train_DQNx.py`, `render_DQNx.py` and `evaluate_mlpx_rollout.py` use the new
message encoding but still create the legacy `CustomHighwayEnv`. Switch them to
`make_environment("merge")` once MLPx has been retrained on merge data.
