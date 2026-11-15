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
