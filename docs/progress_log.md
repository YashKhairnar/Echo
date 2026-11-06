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
