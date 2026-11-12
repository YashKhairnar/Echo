# DQN Request Policy

The DQN learns which visible neighboring vehicle, if any, should be requested at each timestep.

## Action space

The DQN has `11` output actions:

```text
0–9   visible-neighbor slots
10    no request
```

The mapping is dynamic. A request slot refers to a visible neighboring vehicle in the current observation, not to a fixed global vehicle ID. Invalid slots are masked before action selection and when calculating target Q-values.

The same visibility rule is used by data collection and rendering. A vehicle outside the current observation cannot be requested.

The environment's driving action space is separate and has `5` actions. The DQN does not select the driving action.

## Interaction flow

```text
observation
    ↓
DQN selects visible request slot or no request
    ↓
requested vehicle message
    ↓
MLP world model + MCP planner select driving action
    ↓
real environment executes driving action
    ↓
real transition trains the DQN
```

The DQN is trained only on real environment transitions. Predicted MLP transitions are not placed in the replay buffer.

## Reward

The request policy uses:

```text
total reward = driving reward − communication cost
```

The current communication cost is:

```python
COMMUNICATION_COST = 0.3
```

No-request actions have zero communication cost.

## Training

The training script uses:

- Replay buffer size: `10,000`
- Batch size: `128`
- Learning rate: `1e-5`
- Discount factor: `0.99`
- Initial exploration: `1.0`
- Minimum exploration: `0.01`
- Target-network update frequency: `100` steps
- Default training length: `1000` episodes

Run training with:

```bash
PYTHONPATH=. .venv/bin/python -m scripts.train_DQN
```

The DQN checkpoint is saved to:

```text
checkpoints/dqn/latest.pt
```

Training logs and graphs are saved under:

```text
results/dqn/
```

Logged metrics include reward, loss, exploration probability, episode length, request count, no-request count, and communication cost.

## Latest training result

The resumed `1000→2000` episode experiment achieved:

```text
Average reward:       8.67
Previous run average: 6.42
Average episode length: 10.73 steps
```

Communication decreased during the resumed run:

```text
Requests:       1.5 → 0.8 per episode
Communication cost: 0.45 → 0.24 per episode
```

The reward improved overall, but the average DQN loss increased from approximately `1.76` to `7.63`. This indicates that Q-value learning is still unstable. The next evaluation should use `epsilon = 0` and compare the trained policy against a no-request baseline.
