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
