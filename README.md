# Echo: Co-operative Driving

This project explores cooperative driving with HighwayEnv, where multiple vehicles interact on a highway. Vehicles share their kinematics, current actions, and driving intent. A trained MLP world model predicts future traffic states and rewards, an MCP planner selects driving actions, and a DQN learns when to request information from visible neighboring vehicles while accounting for communication cost.

## Progress

See the [project progress log](docs/progress_log.md).

## Documentation

- [Data collection configuration](docs/data_collection_phase1_config.md)
- [MLP world model](docs/mlp_world_model.md)
- [DQN request policy](docs/dqn_request_policy.md)
- [MCP planner and evaluation](docs/planning_and_evaluation.md)
- [Experiment tracking](docs/experiment_tracking.md)

## DQN Experiments

<details>
<summary><strong>Experiment 1 — DQN request policy (1,000 episodes)</strong></summary>

The current DQN request-policy experiment uses:

- `1000` training episodes
- `1e-5` learning rate
- `0.3` communication cost per request
- Visible-neighbor request masking
- Five-step MCP planning
- MLP checkpoint: `checkpoints/mlp/best.pt`

The DQN learns to request information from visible neighboring vehicles or choose no request. Training results are stored under `results/dqn/`.

<video controls width="800">
  <source src="https://raw.githubusercontent.com/YashKhairnar/Echo/main/docs/images/dqn_1000_policy.mp4" type="video/mp4">
  Your browser does not support embedded videos. [View the video directly](https://raw.githubusercontent.com/YashKhairnar/Echo/main/docs/images/dqn_1000_policy.mp4).
</video>

[Open the DQN policy video directly](https://raw.githubusercontent.com/YashKhairnar/Echo/main/docs/images/dqn_1000_policy.mp4)

### Training graphs

<table>
  <tr>
    <td><img src="docs/images/dqn_1000_rewards.png" width="480" alt="DQN reward"></td>
    <td><img src="docs/images/dqn_1000_training.png" width="480" alt="DQN loss and exploration"></td>
  </tr>
  <tr>
    <td><img src="docs/images/dqn_1000_requests.png" width="480" alt="DQN request actions"></td>
    <td><img src="docs/images/dqn_1000_communication_cost.png" width="480" alt="DQN communication cost"></td>
  </tr>
</table>

</details>

<details>
<summary><strong>Experiment 2 — DQN resumed training (1,000 → 2,000 episodes)</strong></summary>

The resumed run achieved an average reward of `8.67`, compared with `6.42` for the previous 1000-episode run. Communication cost decreased as the policy requested information less often. The run is stored under:

```text
results/dqn/experiments/dqn_resume_1000_2000/
```

The reward improved, but the DQN loss remained unstable, so the policy should be evaluated against a no-request baseline before drawing final conclusions.

[Read the resumed-experiment notes](docs/dqn_resume_1000_2000.md)

<video controls width="800">
  <source src="https://raw.githubusercontent.com/YashKhairnar/Echo/main/docs/images/dqn_resume_1000_2000_policy.mp4" type="video/mp4">
  Your browser does not support embedded videos. [View the video directly](https://raw.githubusercontent.com/YashKhairnar/Echo/main/docs/images/dqn_resume_1000_2000_policy.mp4).
</video>

[Open the resumed-policy video directly](https://raw.githubusercontent.com/YashKhairnar/Echo/main/docs/images/dqn_resume_1000_2000_policy.mp4)

### Training graphs

<table>
  <tr>
    <td><img src="docs/images/dqn_resume_1000_2000_rewards.png" width="480" alt="Resumed DQN reward"></td>
    <td><img src="docs/images/dqn_resume_1000_2000_training.png" width="480" alt="Resumed DQN loss and exploration"></td>
  </tr>
  <tr>
    <td><img src="docs/images/dqn_resume_1000_2000_requests.png" width="480" alt="Resumed DQN request actions"></td>
    <td><img src="docs/images/dqn_resume_1000_2000_communication_cost.png" width="480" alt="Resumed DQN communication cost"></td>
  </tr>
</table>

</details>
