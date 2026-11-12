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
- Added DQN training logs and reward, loss, request, and communication-cost graphs under `results/dqn/`.
- Added a renderer that displays the requested vehicle and its intent.
- Added automatic experiment folders with configurations, notes, metrics, graphs, and checkpoint copies.
- Added dynamic request masking so the DQN can request only visible neighboring vehicles.
- Added resumable DQN training with replay-buffer restoration.
- Added rendered policy video recording, a Play/Pause button, and a visual highlight for the requested vehicle.
- Resumed DQN training from 1000 to 2000 episodes.
- Observed average reward improvement from `6.42` to `8.67` and reduced communication cost.
- Recorded continued Q-loss instability and identified no-request baseline evaluation as the next step.
