# Project Progress Log

## 2026-11-05

- Configured the custom HighwayEnv with one ego vehicle and ten neighboring vehicles.
- Added stable vehicle IDs for tracking vehicles across observations.
- Added vehicle intent fields for acceleration, steering, and lane changes.
- Simplified the observation to Kinematics only with shape `(10, 9)`.
- Implemented vehicle communication messages containing vehicle state, action, and intent.
- Completed the random-policy data collector and CSV transition format.
- Added JSON preprocessing through `WorldModelDataset`.
- Started the MLP world model and training pipeline.
- Added checkpoint saving, best/latest checkpoints, and resume support.
- Updated data collection to append safely without duplicate headers or episode IDs.

## 2026-11-12

- Added the MLP world-model checkpoint and training-metric graphs under `results/MLP/`.
- Added the five-step MCP planner for selecting driving actions from MLP predictions.
- Changed the DQN into a request policy with visible-neighbor action masking.
- Added a no-request action and communication cost to the DQN reward.
- Updated data collection to store request actions, visibility-consistent messages, and episode-end flags.
- Added DQN training logs and reward, loss, request, and communication-cost graphs under `results/DQN/`.
- Added a renderer that displays the requested vehicle and its intent.
- Added automatic experiment folders with configurations, notes, metrics, graphs, and checkpoint copies.
- Added dynamic request masking so the DQN can request only visible neighboring vehicles.
- Added resumable DQN training with replay-buffer restoration.
- Added rendered policy video recording, a Play/Pause button, and a visual highlight for the requested vehicle.
- Resumed DQN training from 1000 to 2000 episodes.
- Observed average reward improvement from `6.42` to `8.67` and reduced communication cost.
- Recorded continued Q-loss instability and identified no-request baseline evaluation as the next step.

## 2026-11-12 — Multi-request DQNx

- Added multi-request data collection with up to ten visible request slots.
- Added the MLPx world model for observation, reward, and multi-request message inputs.
- Added DQNx, which predicts request/no-request values independently for each visible slot.
- Kept driving control separate: MLPx plus the five-step MCP planner selects the driving action.
- Added automatic DQNx/MLPx experiment folders, checkpoints, metrics, graphs, and resume support.
- Added DQNx rendering with requested-vehicle highlighting, IDs, and intent overlays.
- Documented that the current `0.1` cost per requested vehicle can encourage the learned policy to request nobody; evaluation should include a no-request baseline.
- Separated public kinematics from private messages: observations now contain position/velocity fields and IDs, while messages contain intent, target lane, and capped time-to-collision.
- Updated MLPx input size from `205` to `115` to match the new four-value message format.

## 2026-11-13 — TTC MLPx/DQNx evaluation

- Trained MLPx on the TTC dataset for 50 epochs; best test loss was `0.1693`
  at epoch 23.
- Trained DQNx for 1,000 episodes with communication cost `0.1`.
- Reward improved during training, but the final policy requested no vehicles
  in a ten-seed evaluation.
- Compared no-request, all-visible-request, and trained-DQNx policies using
  the same seeds. The no-request and trained-DQNx policies both achieved mean
  total reward `5.54`; all-visible requests achieved `-0.53`.
- The current experiment therefore reduces communication but does not yet show
  a measurable communication benefit. The next candidate is a `0.05` cost
  experiment with separate driving-reward logging.

## 2026-11-14 — MLPx rollout evaluation

- Compared MLPx predictions with real-environment rollouts from the same state
  and candidate driving action.
- Five-step planning achieved `23.1%` action-ranking agreement.
- Three-step planning achieved `66.7%` agreement without messages and `42.9%`
  with all messages.
- Set the active MCP planner horizon to three steps because shorter rollouts
  reduce accumulated world-model error.
- Re-evaluated the current MLPx checkpoint with three-step rollouts: action
  ranking agreement was `70.6%` without messages and `45.5%` with all messages.
- Expanded the random multi-request WM dataset to 5,000 episodes; MLPx has
  not yet been retrained on that expanded dataset.

## 2026-11-15 — Cooperative merge scenario with hidden driver types

- Diagnosed why requests never helped: messages (current acceleration, TTC)
  were largely inferable from kinematics, and the highway task rarely made
  the best action depend on them.
- Added `CooperativeMergeEnv`: the ego lane is blocked by a stalled car, so the
  ego must merge into a dense target-lane platoon.
- Added `DriverTypeIDMVehicle` with a private `yielding` / `aggressive` type
  that only shows when someone merges in front of it, and a cruise-controlled
  ego (`CruiseControlledVehicle`) so crashes come from merge decisions, not
  rear-ending.
- Messages now carry driver type through a single `encode_message()` helper
  (slot 1 replaces intent steering; MLPx input size unchanged).
- Collectors take `--env merge|highway`, write merge data to new files, and
  use merge-biased exploration. The matched collector adds a `rear` branch.
- Added `scripts/check_information_gap.py`. On 300 seeds, driver type is
  unpredictable from observation before a merge (51.4% vs 51.4% chance).
  Asking the rear car gave return 7.35 vs 4.64 for the best no-information
  rule, with 0% crashes and merging 4 s sooner.
- Next: collect merge data, retrain MLPx, and repeat the check with the MLPx
  planner instead of hand-written rules.

## 2026-11-15 — MLPx crash head

- Trained the first merge MLPx (2,000 episodes). It reacted to driver-type
  messages only weakly; the MLPx planner crashed about 50% of the time with or
  without messages (`scripts/evaluate_merge_planner.py`).
- Added a crash head to MLPx (BCE on `terminated`), trained the state and
  reward heads on non-crash steps only, and switched the reward target to the
  driving reward. Crash AUC on test: 0.96.
- The MCP planner now weighs each imagined step by its crash probability, and
  `--crash-value` sets how bad an imagined crash is.
- With crash value −6, asking the rear car cut crashes from 21% to 8%
  (return 4.23 → 4.86), still well below the hand-written ask-rear rule
  (7.35, 0% crashes).
- Diagnosis: MLPx predicts the right average merge risk (~45%) and raises it
  for "aggressive" (65%), but does not lower it for "yielding" (44% vs 0%
  real). Next: collect more merge data so "yielding" answers are better
  represented.
- Collected 8,000 more merge episodes (10,000 total) and retrained: test loss
  0.32, crash AUC 0.99. The MLPx planner improved to returns of about 7.5–8.0,
  but messages now barely change its predictions.
- Confirmed that the data contains the signal (crash within 3 steps of
  merging: aggressive 76%, not asked 51%, yielding 31%), so the flat MLPx input
  is the bottleneck: it must match each car's observation row to its message
  slot by itself. Next: a per-vehicle encoder that pairs each car with its
  own message.
- Replaced the flat MLPx with a per-vehicle encoder that pairs each car's
  kinematics with its own message (`FlatMLPxWorldModel` kept for old
  checkpoints). Test loss 0.286, crash AUC 0.998.
- First clear message benefit with the learned planner: asking the rear car
  raised return from 5.21 to 6.53 and cut crashes from 38% to 26% (100 seeds).
  Still below the hand-written ask-rear rule (7.35, 0% crashes), and the
  no-message baseline is worse than with the flat model.
- Remaining issue: crashes happen 1–2 steps after the merge action, so the
  planner predicts them through imagined intermediate states, which blurs the
  message's effect. Candidate fix: a "crash within k steps" head.

## 2026-11-15 — Cleanup and first request DQN in the merge scenario

- Archived everything from the highway experiments: data, checkpoints,
  results, and scripts in `archive/highway_ttc_multirequest/`, write-ups in
  `docs/archive/`. Earlier merge MLPx checkpoints moved to
  `checkpoints/MLPx_merge_history/`.
- Made the code merge-only: shared request helpers in `request_utils.py`,
  removed the legacy single-request dataset, highway env option, and
  multi-request DQNx. Renamed scripts to `train_DQN.py`, `render_DQN.py`,
  and `evaluate_merge_policies.py`.
- New request DQN: at most one car per step (11 actions), per-vehicle
  Q-network, Double DQN. Trained 2,000 episodes at cost 0.05.
- On 200 unseen seeds the DQN (total 6.27) matched "ask nobody" (6.23) and
  random at the same budget (6.29). It asked 2.2 times per episode, with 29% of
  requests to the rear car (random: 14%). The hand-written rear rule was best
  (6.89), but only 0.66 above "none": the planner still gets little value
  from messages, so the request policy has little to learn.

## 2026-11-15 — MLPx v5 (crash-within-3-steps head) and second DQN run

- Added a crash-soon head to MLPx: probability that the episode crashes
  within the next 3 steps, from the real observation and messages. The
  planner uses it as the first step's risk. Crash-soon AUC 0.91.
- With this planner, asking the rear car is worth +2.68 over asking nobody
  (8.47 vs 5.79, 0% crashes), up from +0.66 with MLPx v4 and close to the
  +2.7 in the true environment.
- Retrained the request DQN (2,000 episodes, cost 0.05). It collapsed to
  almost never asking (0.3 requests per episode) and scored like "none"
  (5.78). The planner side now works; the request policy's exploration and
  credit assignment are the next problem.

## 2026-11-15 — Request DQN v3: target-lane exploration + 3-step returns

- Exploration now asks target-lane cars most of the time when it asks, and
  the DQN learns from 3-step returns so delayed merge rewards reach the
  request. Trained 3,000 episodes at cost 0.05.
- On 200 unseen seeds: DQN 6.77 vs "none" 5.79 (+0.98), crashes 2% vs 18%,
  57% of requests to the rear car. First learned policy that beats asking
  nobody. Still below the hand-written rear rule (8.47): it asks twice as
  often and merges later. Training reward was still rising.

## 2026-11-15 — Code simplified for readability

- Shorter, plainer code with a short "what this file does" header in every
  main file. Behaviour is unchanged: the environment, messages, MLPx
  predictions, planner values, rule and DQN results, and the dataset give
  exactly the same numbers as before.
- Removed things no longer used: the old flat MLPx and its loading logic,
  the unused highway base environment and intent vehicles, MLPx `--resume`,
  `--architecture` and the planner's `use_crash_soon` switch (always on).
  `build_planner` now lives in `planner/MCP.py`.
- The code before this change is kept in
  `archive/code_snapshot_before_simplify_2026-11-15/` (needed to load the
  older flat MLPx checkpoints).
