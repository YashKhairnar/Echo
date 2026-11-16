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

The collector writes merge data to `data/mergeMultiRequestData.csv`:

```bash
.venv/bin/python -m scripts.collect_multi_request_data --episodes 2000   # data/mergeMultiRequestData.csv
```

Each step records every transition, and each visible car is asked with 50%
probability, so the data contains the same kinds of situations both with and
without each message.

Driving actions come from `MergeExploration` (`scripts/exploration.py`): each
episode draws a merge step between 0 and 12; before it the ego only changes
speed, at it the ego merges, and afterwards actions are uniform random. This
gives merges from many different gaps and speeds.

## MLPx planner check (first merge-trained MLPx)

MLPx trained on `data/mergeMultiRequestData.csv` (2,000 episodes, 24,826
rows; best test loss 0.61 at epoch 23). `scripts/evaluate_merge_planner.py` (now `evaluate_merge_policies.py`)
drives 100 seeds with the 3-step MCP planner:

| Requests | Return | Crash rate | Merged | Asks / episode |
| --- | ---: | ---: | ---: | ---: |
| none | 4.45 ± 0.64 | 56% | 44% | 0 |
| rear car | 5.23 ± 0.62 | 50% | 50% | 5.0 |
| all visible | 5.85 ± 0.63 | 44% | 56% | 109.8 |

Messages help slightly, but the differences are within noise, and the planner
crashes about half the time in every mode. The hand-written `ask_rear` rule
reaches 7.35 with 0% crashes on the same scenario. MLPx does react to the
message in the right direction: an "aggressive" message lowers its predicted
3-step return for merging by about 0.4 compared with "polite". But it greatly
underestimates the danger: in those states, merging in front of an aggressive
car really crashes 76% of the time.

Likely cause: the planner keeps adding predicted rewards after a predicted
crash, so the one-step −2 penalty is washed out by later positive rewards, and
MSE regression on rare crash rewards pulls predictions toward the average.

## MLPx v2: crash head

Changes:

- `MLPxWorldModel` has a third output, a **crash head** (logit that the step
  ends in a crash), trained with binary cross-entropy on `terminated`.
- The next-state and reward losses use only non-crash steps, and the reward
  target is the *driving* reward (communication cost is the request
  policy's concern, not the world model's).
- `MCPPlanner` values each imagined step as
  `p_crash * crash_value + (1 - p_crash) * (reward + discount * future)`,
  so imagined driving after a likely crash no longer counts.
- `MLPxWorldModel.load()` loads old checkpoints too (crash probability 0).

Training on the same 2,000 episodes: best test loss around epoch 10, overfits
afterwards (`best.pt` keeps the best). Crash AUC on the test split is 0.96:
the head ranks real crash steps above safe steps 96% of the time.

Planner results (100 seeds). `--crash-value` is the value the planner gives an
imagined crash. The 3-step look-ahead cannot see that a crash also loses the
rest of the episode, so the environment's −2 makes crashes look too cheap:

| Crash value | Requests | Return | Crash rate | Merged | Merge time |
| ---: | --- | ---: | ---: | ---: | ---: |
| −2 | none | 5.24 ± 0.62 | 49% | 50% | 5.2 s |
| −2 | rear car | 4.75 ± 0.55 | 43% | 52% | 8.2 s |
| −2 | all visible | 7.03 ± 0.59 | 32% | 67% | 4.1 s |
| −6 | none | 4.23 ± 0.38 | 21% | 63% | 13.5 s |
| −6 | rear car | 4.86 ± 0.40 | 8% | 77% | 13.9 s |
| −6 | all visible | 4.99 ± 0.44 | 11% | 71% | 10.4 s |
| −12 | none | 2.80 ± 0.20 | 5% | 64% | 17.8 s |
| −12 | rear car | 3.12 ± 0.24 | 5% | 65% | 17.1 s |

Why messages still help little: in risky merge states (rear car close enough
to react, but a yielding car could brake in time), the predicted chance of
crashing within 3 steps of merging is

| Rear car | Real crash rate | No message | True message | Wrong message |
| --- | ---: | ---: | ---: | ---: |
| yielding | 0% | 44% | 44% | 61% |
| aggressive | 76% | 48% | 65% | 47% |

Without a message the model predicts the correct average risk, and it has
learned that "aggressive" raises the risk. It has not learned that "yielding"
makes the merge safe, so the planner cannot merge confidently after a polite
answer. The data has only about 2,000 merges, of which roughly 500 have a
"yielding" answer from a close rear car, which is likely too few.

## MLPx v3: crash head + 10,000 episodes

Collected 8,000 more episodes (`--seed 1`, episode IDs 2000–9999): 10,000
episodes, 125,792 rows, 6,859 crash steps. Retraining MLPx: best test loss 0.32
at epoch 34 (v2: about 0.53), crash AUC 0.99 (v2: 0.96).

Planner results (100 seeds):

| Crash value | Requests | Return | Crash rate | Merged | Merge time |
| ---: | --- | ---: | ---: | ---: | ---: |
| −2 | none | 7.43 ± 0.61 | 35% | 65% | 2.9 s |
| −2 | rear car | 7.50 ± 0.60 | 34% | 66% | 3.2 s |
| −2 | all visible | 7.61 ± 0.59 | 30% | 70% | 2.7 s |
| −6 | none | 7.60 ± 0.57 | 29% | 71% | 3.7 s |
| −6 | rear car | 7.98 ± 0.56 | 28% | 72% | 3.5 s |
| −6 | all visible | 7.45 ± 0.52 | 21% | 79% | 4.3 s |

Driving got much better overall, but messages now make almost no difference.
Predicted chance of crashing within 3 steps of merging in risky states:

| Rear car | Real crash rate | No message | True message | Wrong message |
| --- | ---: | ---: | ---: | ---: |
| yielding | 0% | 41% | 46% | 49% |
| aggressive | 76% | 31% | 41% | 38% |

The data does contain the signal. Over all `LANE_LEFT` steps in the dataset
with a rear target-lane car within about 40 m, the crash rate within 3 steps
was:

| Rear car's answer | Merges | Crash within 3 steps |
| --- | ---: | ---: |
| aggressive | 2,802 | 76% |
| not asked | 6,632 | 51% |
| yielding | 3,689 | 31% |

So the model, not the data, is the bottleneck. MLPx sees one flat vector:
the rear car's kinematics are in some observation row *k*, and its message is
in message slot *k* elsewhere in the vector. Using the message requires
learning "find the rear car's row, then read the matching message slot",
which a plain MLP learns poorly. With more data it preferred predicting the
average risk and ignoring messages.

Proposed fix: encode each vehicle together with its own message (a shared
per-vehicle encoder over `[kinematics row, requested flag, message]`, then pool
the vehicles and add the ego state and action), so each message is tied to the
car it describes by construction.

## MLPx v4: per-vehicle encoder

`MLPxWorldModel` is now the per-vehicle model. (The flat model was removed
from the code when it was simplified; a copy of that code is in
`archive/code_snapshot_before_simplify_2026-11-15/`.) The
input and outputs are unchanged (115 values in; next observation, no-crash
reward, crash logit out), so the dataset, planner and scripts did not change.

Inside, neighbour row *k* is paired with message slot *k−1* and its request
flag. A shared vehicle encoder sees
`[kinematics, requested, message, ego kinematics, action]` for each car.
Embeddings are pooled (sum + max) into a scene vector. Each car's next row is
predicted from its own embedding plus the scene; reward and crash come from
the scene. 93k parameters (flat: 136k).

Training on the same 10,000 episodes: best test loss 0.286 at epoch 47 (flat:
0.320), crash loss 0.023 (flat: 0.053), crash AUC 0.998 (flat: 0.992), with no
overfitting by epoch 50.

Planner results (100 seeds):

| Crash value | Requests | Return | Crash rate | Merged | Merge time |
| ---: | --- | ---: | ---: | ---: | ---: |
| −2 | none | 5.21 ± 0.53 | 38% | 61% | 5.3 s |
| −2 | rear car | 6.53 ± 0.50 | 26% | 74% | 4.2 s |
| −2 | all visible | 6.93 ± 0.57 | 31% | 69% | 3.9 s |
| −6 | none | 5.40 ± 0.47 | 27% | 73% | 6.4 s |
| −6 | rear car | 6.27 ± 0.47 | 19% | 80% | 4.6 s |
| −6 | all visible | 6.81 ± 0.52 | 24% | 75% | 4.7 s |

For the first time, the MLPx planner drives clearly better with messages:
asking the rear car raises return by about 1.3 (about 2.5 standard errors)
and cuts crashes from 38% to 26%. However:

- Without messages it drives worse than the flat model (5.2 vs 7.4), and
  all modes remain below the hand-written `ask_rear` rule (7.35, 0% crashes).
- The per-state check still separates the two driver types poorly. Predicted
  chance of crashing within 3 imagined steps of merging:

  | Rear car | Real | No message | True message | Wrong message |
  | --- | ---: | ---: | ---: | ---: |
  | yielding | 0% | 63% | 38% | 45% |
  | aggressive | 76% | 54% | 42% | 37% |

  Any message lowers the predicted risk, and "aggressive" is barely above
  "yielding". The crash usually happens 1–2 steps *after* `LANE_LEFT`, so the
  planner must first imagine the half-merged state and then predict the crash
  from it; errors in that imagined state blur the effect of the message.

Possible next step: train the crash head on "crash within the next k steps"
for the current state and action, so the merge risk is predicted directly at
the decision instead of through imagined intermediate states.

## MLPx v5: crash-within-3-steps head

The DQN's first run (below) showed the planner gained little from a message.
The cause: a bad merge crashes one or two steps *after* the merge action, so
the planner only sees it through imagined states, and those blur the message
away.

MLPx v5 adds a **crash-soon head**: the probability that the episode crashes
at this step or within the next two (`--crash-horizon 3`). The label comes
from the recorded episodes (`MultiRequestWorldModelDataset`), and the head
reads the real current observation and the real messages. The planner uses it as the risk of the *first*
imagined step; deeper steps keep the one-step crash head.

Training: same 10k-episode data and per-vehicle architecture, 50 epochs. Run:
`results/MLPx/experiments/20261115_185956_mlpx/`. One-step crash AUC 0.998,
crash-soon AUC 0.91. v4 is kept in `checkpoints/MLPx_merge_history/v4_per_vehicle_10k/`.

Request rules with the v5 planner, 200 unseen seeds, cost 0.05
(`results/merge_policies/cost0.05_baselines_crash_soon.json`):

| Policy | Total reward | Driving reward | Crash | Merged | Requests / ep |
| --- | ---: | ---: | ---: | ---: | ---: |
| none | 5.79 ± 0.30 | 5.79 | 18% | 78% | 0 |
| all | 2.03 ± 0.26 | 9.63 | 0% | 97% | 152.0 |
| random | 5.58 ± 0.31 | 6.51 | 15% | 82% | 18.7 |
| rear (hand-written) | **8.47 ± 0.28** | 8.80 | 0% | 97% | 6.7 |

Asking the rear car is now worth **+2.68** over asking nobody (v4: +0.66),
about the +2.7 the same information is worth in the true environment, with no
crashes. Asking everyone shows the planner now drives near-perfectly with full
information (driving 9.63), but the communication cost wipes that out.

## Request DQN

The request policy (`src/cooperative_driving/policies/DQN.py`) asks **at most
one car per step**: 11 actions, one per observation slot plus "ask nobody";
slots without a car are masked. Like MLPx it is per-vehicle: a shared network
scores each visible car from its kinematics, the ego's kinematics, and a
pooled scene summary, plus a separate score for asking nobody.

Training (`scripts/train_DQN.py`): the DQN picks whom to ask, the asked car's
message goes to the frozen MLPx + MCP planner (3 steps, crash value −2), and
the DQN reward is driving reward minus `communication_cost` per request.
Double DQN, Huber loss, Adam 1e-4, batch 64, replay 50k, target sync every 500
steps, epsilon 1.0 → 0.05 linearly over the first 60% of episodes (exploration
asks nobody half the time and a random visible car otherwise). Only crashes
stop bootstrapping; the 20-step time limit does not.

Evaluation (`scripts/evaluate_merge_policies.py`) runs every request rule on
the same unseen seeds (100000+): none, all, random (one car every step),
random_matched (random at the DQN's request rate), rear (hand-written), and
dqn.

### First run: cost 0.05, 2,000 episodes (MLPx v4)

Run: `results/DQN/experiments/20261115_171402_dqn_cost0.05/` (~17 min). The training reward rose only slightly (250-episode
means 5.7 → about 6.2) and the Huber loss kept rising after exploration
ended.

Evaluation on 200 unseen seeds, cost 0.05 per request
(`results/merge_policies/cost0.05_best.json`):

| Policy | Total reward | Driving reward | Crash | Merged | Requests / ep | Share asked to rear car |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| none | 6.23 ± 0.37 | 6.23 | 28% | 72% | 0 | – |
| all | 0.69 ± 0.30 | 7.53 | 22% | 77% | 136.8 | 12% |
| random | 5.89 ± 0.35 | 6.73 | 24% | 74% | 16.9 | 12% |
| random_matched | 6.29 ± 0.37 | 6.40 | 26% | 73% | 2.1 | 14% |
| rear (hand-written) | **6.89 ± 0.36** | 7.23 | 20% | 80% | 6.8 | 100% |
| dqn | 6.27 ± 0.38 | 6.38 | 27% | 72% | 2.2 | 29% |

The DQN learned to ask rarely, and twice as often as random it asks the
right car (29% vs 14%), but it drives no better than asking nobody. The
largest possible gain is also small: even the hand-written rear rule beats
"none" by only 0.66 (about 2 standard errors), because the planner gets
little out of a message (in the true environment the same information is
worth +2.7; see the information-gap check). With such a small and noisy
benefit, 2,000 episodes are not enough for the DQN to find it.

The bottleneck is therefore still how much the MLPx planner gains from a
message, not the request policy itself.


### Second run: cost 0.05, 2,000 episodes (MLPx v5)

Run: `results/DQN/experiments/20261115_191042_dqn_cost0.05/` (best checkpoint
at episode 1290). Training reward stayed around 6.2–6.8 (100-episode means).
The first-run DQN is kept in `checkpoints/DQN_history/v1_step_crash_head/`.

Evaluation on 200 unseen seeds, cost 0.05
(`results/merge_policies/cost0.05_dqn_v2_crash_soon.json`):

| Policy | Total reward | Driving reward | Crash | Merged | Requests / ep | Share asked to rear car |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| none | 5.79 ± 0.30 | 5.79 | 18% | 78% | 0 | – |
| all | 2.03 ± 0.26 | 9.63 | 0% | 97% | 152.0 | 13% |
| random | 5.58 ± 0.31 | 6.51 | 15% | 82% | 18.7 | 12% |
| random_matched | 5.84 ± 0.30 | 5.86 | 18% | 79% | 0.3 | 14% |
| rear (hand-written) | **8.47 ± 0.28** | 8.80 | 0% | 97% | 6.7 | 100% |
| dqn | 5.78 ± 0.30 | 5.79 | 18% | 78% | 0.3 | 43% |

The planner now gains a lot from the right message (+2.68 for the rear
rule), but the DQN collapsed to almost never asking (0.3 requests per
episode), so it scores like "none". Likely causes: exploration asks a random
one of ~9 cars, so the rear car is rarely asked right before a merge, and the
benefit of a request only shows up a few steps later as lane reward. Next
things to try: exploration biased towards target-lane cars or longer
training, and a larger replay warm-up.

### Third run: target-lane exploration + 3-step returns, 3,000 episodes

Two training changes in `scripts/train_DQN.py` (now the defaults):

- **Exploration towards the target lane.** Random requests now ask nobody
  50% of the time; otherwise a car in the target lane 80% of the time (while
  the ego has not merged), else any visible car. Before, a random one of ~9
  cars was asked, so the car that matters was almost never asked right
  before a merge. Which target-lane car to ask is still left to the DQN.
- **3-step returns** (`N_STEP = 3`, `NStepCollector`). A request's benefit
  shows up a few steps later as lane reward; multi-step targets carry it
  back to the request directly.

Run: `results/DQN/experiments/20261115_195934_dqn_cost0.05/`, best checkpoint
at episode 2,968 (100-episode training mean 7.73 at the end, still rising).
The second-run DQN is kept in `checkpoints/DQN_history/v2_crash_soon/`.

Evaluation on 200 unseen seeds, cost 0.05
(`results/merge_policies/cost0.05_dqn_v3_nstep.json`):

| Policy | Total reward | Driving reward | Crash | Merged | Merge time | Requests / ep | Share asked to rear car |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| none | 5.79 ± 0.30 | 5.79 | 18% | 78% | 8.3 s | 0 | – |
| all | 2.03 ± 0.26 | 9.63 | 0% | 97% | 4.4 s | 152.0 | 13% |
| random | 5.58 ± 0.31 | 6.51 | 15% | 82% | 7.4 s | 18.7 | 12% |
| random_matched | 5.53 ± 0.31 | 6.17 | 17% | 80% | 7.9 s | 12.7 | 12% |
| rear (hand-written) | **8.47 ± 0.28** | 8.80 | 0% | 97% | 4.9 s | 6.7 | 100% |
| **dqn** | **6.77 ± 0.28** | 7.44 | 2% | 94% | 7.8 s | 13.5 | 57% |

The first learned request policy that clearly beats asking nobody: +0.98
(about 3 standard errors), crashes down from 18% to 2%, and 57% of its
requests go to the rear car (random: 12%). It is still 1.7 below the
hand-written rule: it asks twice as often (13.5 vs 6.7 per episode), only
about half of its requests go to the rear car, and it merges late (7.8 s vs
4.9 s). Training reward was still rising at the end, so longer training is
the obvious next step, followed by the cost sweep.
