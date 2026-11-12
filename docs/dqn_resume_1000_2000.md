# DQN Resumed Training: 1,000 to 2,000 Episodes

## What changed

- Resumed the DQN request policy after the first 1,000 episodes.
- Continued training to a total target of 2,000 episodes.
- Kept the learning rate at `1e-5`.
- Used a communication cost of `0.3` per request.
- Continued using visible-vehicle request masking and five-step MCP planning.

## Results

- Average reward during the resumed run: `8.67`.
- Previous 1,000-episode average reward: `6.42`.
- Average episode length: `10.73` steps.
- Reward peaked in the middle of the resumed run and declined somewhat near the end.
- Average DQN loss increased from approximately `1.76` to `7.63`.

## What worked

- Overall reward improved compared with the previous run.
- The policy reduced communication use during training.
- Requests decreased from approximately `1.5` to `0.8` per episode.
- Communication cost decreased from approximately `0.45` to `0.24` per episode.

## What to improve

- Evaluate with `epsilon = 0` across multiple fixed seeds.
- Compare against a no-request baseline.
- Investigate the increasing DQN loss and Q-value instability.
- Save a separate best-evaluation-reward checkpoint instead of relying only on the latest checkpoint.
