# MLP World Model

The current world model is a multilayer perceptron that predicts the next Kinematics state and driving reward from one collected transition. It is frozen during DQN training and used by the MCP planner to evaluate candidate driving actions.

## Model input

Each input combines:

```text
observation_t: 10 × 9 = 90 values
action_t:      5 one-hot values
message_t:     10 values
```

The complete input is therefore a `105`-value vector:

```text
90 + 5 + 10 = 105
```

The observation contains vehicle position, velocity, intent, and vehicle ID. The message contains the communicating vehicle's position, velocity, action, intent, and ID.

## Architecture

The shared MLP is:

```text
105 → 256 → 256 → 128
```

Each layer is followed by a ReLU activation. The final shared representation is passed to two output heads:

```text
next-state head: 128 → 90
reward head:     128 → 1
```

The model therefore returns:

```python
predicted_next_state, predicted_reward = model(inputs)
```

## Training

`WorldModelDataset` reads the collected CSV and returns:

```text
input:       105 values
next_state:  90 values
reward:       1 value
```

The training script uses:

- An 80/20 random train/test split.
- Mean squared error for next-state prediction.
- Mean squared error for reward prediction.
- Total loss equal to state loss plus reward loss.
- Adam optimizer with a default learning rate of `1e-3`.
- Batch size `64`.
- Default training length of `50` epochs.

Training metrics are saved under:

```text
results/MLP/
```

The CSV contains total, next-state, and reward losses. The script also saves loss graphs. New runs are versioned instead of overwriting previous logs; resuming with the same `--log` path appends to that run's history.

## Checkpoints

Checkpoints are saved in:

```text
checkpoints/mlp/
```

The files are:

```text
latest.pt  # Most recent checkpoint
best.pt    # Checkpoint with the lowest test loss
```

Each checkpoint stores the model parameters, optimizer parameters, epoch, training loss, and test loss.

To resume training from the latest checkpoint:

```bash
PYTHONPATH=. .venv/bin/python -m scripts.train_mlp_wm \
    --resume checkpoints/mlp/latest.pt \
    --epochs 100
```

To train from the current request-policy dataset:

```bash
PYTHONPATH=. .venv/bin/python -m scripts.train_mlp_wm \
    --data data/randomPolicyData_requestPolicy.csv
```

The MLP input does not include `request_action_t` because the message already represents the information received. Communication cost is applied separately by the request-policy DQN.
