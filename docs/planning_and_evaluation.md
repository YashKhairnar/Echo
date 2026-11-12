# MCP Planning and Evaluation

## MCP planner

The MCP planner uses the trained MLP world model to evaluate driving actions. It plans five steps ahead and returns only the first driving action to the real environment.

```text
observation + requested message
    ↓
MLP predicts next state and reward
    ↓
MCP compares candidate five-step action sequences
    ↓
first driving action is executed in the real environment
```

The planner uses `5^5 = 3125` candidate action sequences in the current exhaustive implementation, so rendering and training can be slow.

## Rendering

Render the trained request policy with:

```bash
PYTHONPATH=. .venv/bin/python -m scripts.render_dqn
```

The overlay shows the ego driving action, requested slot, actual vehicle ID, and the requested vehicle's intent. The requested vehicle is highlighted in the road view. The window includes a Play/Pause button.

Record the rendered behavior as an MP4 video with:

```bash
PYTHONPATH=. .venv/bin/python -m scripts.render_dqn \
    --episodes 10 \
    --video results/dqn/policy_1000episodes.mp4
```

The renderer uses a random seed by default. Pass `--seed` to reproduce the same episodes.

Evaluation uses `epsilon = 0`; no learning occurs during rendering.

## Evaluation metrics

For each evaluation episode, report:

- Total reward after communication cost
- Driving reward
- Communication cost
- Number of requests
- Number of no-request actions
- Episode length

Use multiple fixed random seeds and compare the trained DQN request policy against at least a random-request policy. Reward should be evaluated with learning disabled and exploration set to zero.
