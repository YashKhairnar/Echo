# Cooperative merge scenario

How the current experiment works, where it stands, and how it got here.
The full step-by-step record is in
[archive/merge_scenario_full_history.md](archive/merge_scenario_full_history.md).

## 1. The idea

The ego car must merge into a line of traffic. Some drivers let you in, some
don't, and **you can't tell which by watching**. Asking a car tells you, but
every question costs something. The goal: learn **when to ask, and whom**.

## 2. The scenario

`CooperativeMergeEnv` in `src/cooperative_driving/environments/customEnv.py`.

- **Road:** two lanes. The ego starts in the lower lane at 25 m/s.
- **Forced merge:** a stalled car blocks the ego's lane 250–350 m ahead, so
  the ego must move into the upper (target) lane.
- **Platoon:** the target lane has 9 cars, 25–45 m apart, at 23–26 m/s.
- **Hidden driver type** (50/50 per car, `DriverTypeIDMVehicle`):
  - *yielding*: when a car merges in front of it (within 40 m), it brakes to let it in;
  - *aggressive*: it speeds up (6 m/s²) and closes the gap. Merging in front of it usually crashes.
  - Both drive identically until someone merges, so the type can't be seen in advance.
- **Ego actions:** change left, change right, keep going, faster, slower.
  A built-in cruise control prevents silly rear-end crashes.
- **Episode:** 20 steps (1 per second).
- **Reward per step:** `0.4 × speed score + 0.6 × in target lane`; a crash gives −2 and ends the episode.

## 3. Messages

A car that is asked replies with 4 numbers (`encode_message` in
`communication/getMessage.py`):

| Value | Meaning |
| --- | --- |
| driver type | +1 yielding, −1 aggressive, 0 other |
| intent acceleration | its current acceleration [m/s²] |
| target lane | the lane it is heading for |
| time to collision | seconds until it reaches the car ahead (max 10) |

The driver type is the one that matters; it can't be learned from watching.

## 4. How the system works

```text
each step:  DQN picks whom to ask  ->  message  ->  planner imagines the future with MLPx  ->  driving action
```

**Data** (`scripts/collect_multi_request_data.py`): 10,000 episodes
(125,792 steps). The ego merges at a random moment, and every visible car is
asked with 50% probability, so MLPx sees the same situations with and
without each message.

**MLPx, the world model** (`models/MLPx/mlpx.py`). Input: what the ego sees,
the action, who was asked, and the messages. Output: next observation,
reward, "crash this step?" and "crash within the next 3 steps?". Each car is
encoded *together with its own message* by one shared network; the cars are
then pooled into a scene summary.

**Planner** (`planner/MCP.py`): tries every 3-step action sequence in MLPx
and picks the best first action. A step's value is
`p_crash × (−2) + (1 − p_crash) × (reward + 0.99 × best future)`.
For the first step, `p_crash` is the "crash within 3 steps" probability, which
catches a merge that only goes wrong two steps later.

**Request DQN** (`policies/DQN.py`, `scripts/train_DQN.py`): each step it
asks one visible car or nobody (11 actions). It scores every car with one
shared network, like MLPx. Reward = driving reward − 0.05 per question.
Training: 3,000 episodes, Double DQN, 3-step returns, and exploration that
mostly asks cars in the target lane.

**Evaluation** (`scripts/evaluate_merge_policies.py`): every request policy
drives the same 200 unseen seeds with the same planner.

## 5. Current results

**Is the information hidden and useful?** (`scripts/check_information_gap.py`,
hand-written rules, no learning, 300 seeds)

- A classifier guessing a car's type from what the ego sees: 51.4% (chance 51.4%).
- Without asking, the ego must gamble (merge at once: 5.17, 52% crashes) or
  wait for a gap that is safe against any driver (4.64, 0% crashes).
- The safe rule plus asking the rear car: **7.35**, 0% crashes. So the
  information is worth about +2.7.

**Request policies with the learned planner** (200 unseen seeds, cost 0.05
per question; `results/merge_policies/cost0.05_dqn_v3_nstep.json`):

| Policy | Total score | Crashes | Merged | Questions / episode | Share to rear car |
| --- | ---: | ---: | ---: | ---: | ---: |
| ask nobody | 5.79 | 18% | 78% | 0 | – |
| ask everyone | 2.03 | 0% | 97% | 152 | 13% |
| random car each step | 5.58 | 15% | 82% | 18.7 | 12% |
| random, same budget as DQN | 5.53 | 17% | 80% | 12.7 | 12% |
| rear-car rule (hand-written) | **8.47** | 0% | 97% | 6.7 | 100% |
| **DQN** | **6.77** | 2% | 94% | 13.5 | 57% |

- **The planner uses messages well:** the rear-car rule gains +2.68 over
  asking nobody, which is close to the true value (+2.7).
- **The DQN learned to ask:** it beats asking nobody by +0.98 (about 3
  standard errors) and cuts crashes from 18% to 2%.
- **But it's not there yet:** it asks twice as often as needed, only half of
  its questions go to the rear car, and it merges late (7.8 s vs 4.9 s).

## 6. History

| Step | What changed | Result | What it taught us |
| --- | --- | --- | --- |
| Information-gap check | Hand-written rules | Asking the rear car: 7.35 vs 4.64 (safe, no asking) | The information is hidden and worth asking for |
| MLPx v1 | Plain MLP, 2k episodes | Planner crashes ~50% with or without messages | Crash risk is washed out by later rewards |
| MLPx v2 | Added a crash head | Crash AUC 0.96, messages still help little | Not enough data |
| MLPx v3 | 10k episodes | Drives better, ignores messages | A flat MLP can't pair a car with its message |
| MLPx v4 | Per-vehicle encoder | Rear rule +0.66 over none | Crashes 2 steps later get blurred |
| DQN run 1 | First request DQN (on v4) | 6.27, same as asking nobody | Planner gained too little from messages |
| MLPx v5 | "Crash within 3 steps" head | Rear rule +2.68 over none | Planner now uses messages fully |
| DQN run 2 | Same DQN on v5 | Learned to never ask (5.78) | It never saw a question pay off |
| **DQN run 3** | Target-lane exploration + 3-step returns | **6.77, crashes 2%** | First DQN that learned to ask |

## 7. Where things are

| What | Where |
| --- | --- |
| Data | `data/mergeMultiRequestData.csv` |
| Current MLPx (v5) | `checkpoints/MLPx/best.pt` (run `results/MLPx/experiments/20261115_185956_mlpx/`) |
| Current DQN (run 3) | `checkpoints/DQN/best.pt` (run `results/DQN/experiments/20261115_195934_dqn_cost0.05/`) |
| Older models | `checkpoints/MLPx_merge_history/`, `checkpoints/DQN_history/` |
| Evaluation results | `results/merge_policies/` |

## 8. Next steps

1. Train the DQN longer: its training score was still rising.
2. Cost sweep: train at several question costs and check that it asks less when questions are expensive.
3. Optional: the same idea on a 4-lane highway with normal traffic.
