# Request DQN policy

*Experiment 1 · Baseline request-policy training*

The Request DQN is a learned communication policy. It decides **which visible vehicle to ask**, or whether to ask nobody. It does not directly choose the ego's driving action.

After the DQN selects a request, the frozen MLPx planner uses the returned message to choose the ego action with a three-step planning horizon.

## Experiment status

The current results are from the **first Request DQN training experiment**.
It was trained for 1,000 episodes with the first MLPx checkpoint frozen. This
run establishes the communication-policy baseline; it is not yet the final
policy.

The next DQN phase will use the improved MLPx checkpoint, multiple training
seeds, and fixed validation environments so that return, request count, crash
rate, truncation rate, and loss can be compared fairly with this first run.

## Request actions

The DQN has nine possible actions:

```text
left-front, left-beside, left-rear,
same-lane-front, same-lane-rear,
right-front, right-beside, right-rear,
ask-nobody
```

Invalid spatial slots are masked during action selection. The DQN receives a 45-value state representation containing the ego and eight ordered neighbor slots.

## Training reward

```text
DQN reward = real driving reward - communication cost when a vehicle is asked
```

The DQN uses replay memory, an online Q-network, a target Q-network, epsilon-greedy exploration, and Huber loss. The target network is currently synchronized every 50 episodes.

Train it with:

```bash
.venv/bin/python -m scripts.training.train_request_dqn \
  --episodes 1000 \
  --planning-horizon 3 \
  --communication-cost 0.05
```

Training logs are saved under `results/RequestDQN/experiments/` with return, loss, epsilon, request count, crash, and truncation columns. The current DQN improves return during training, but its crash rate and rising loss show that it still needs stabilization before deployment.
