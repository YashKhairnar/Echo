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

The original single-request renderer is archived in
`archive/legacy_single_request/scripts/render_dqn.py`. The current renderer is
the DQNx renderer documented below.

Historical single-request rendering used:

```bash
PYTHONPATH=. .venv/bin/python -m archive.legacy_single_request.scripts.render_dqn
```

The overlay shows the ego driving action, requested slot, actual vehicle ID, and the requested vehicle's intent. The requested vehicle is highlighted in the road view. The window includes a Play/Pause button.

Record the rendered behavior as an MP4 video with:

```bash
PYTHONPATH=. .venv/bin/python -m archive.legacy_single_request.scripts.render_dqn \
    --episodes 10 \
    --video docs/images/dqn_policy_1000episodes.mp4
```

The renderer uses a random seed by default. Pass `--seed` to reproduce the same episodes.

Evaluation uses `epsilon = 0`; no learning occurs during rendering.

Render the multi-request DQNx policy with its MLPx world model:

```bash
PYTHONPATH=. .venv/bin/python -m scripts.render_DQNx
```

The DQNx renderer highlights every requested visible vehicle and displays its
slot, vehicle ID, and intent. Record it with:

```bash
PYTHONPATH=. .venv/bin/python -m scripts.render_DQNx \
    --video docs/images/dqnx_policy.mp4
```

See the [DQNx renderer documentation](dqnx_request_policy.md#rendering-and-video)
for all renderer options, reproducible seeds, checkpoint selection, and video
recording notes.

The current DQNx policy video can be recorded with:

```bash
PYTHONPATH=. .venv/bin/python -m scripts.render_DQNx \
    --episodes 1 \
    --delay 0.1 \
    --video docs/images/dqnx_current_policy.mp4
```

Run this command from a normal graphical terminal. The `--headless` option is
available for environments with a working RGB-array renderer, but it may not
produce a visible HighwayEnv scene on systems without a display backend.

The current DQNx policy can select no requests. In that case the video still
shows the ego vehicle moving, but no neighboring vehicle is highlighted.

## Evaluation metrics

For each evaluation episode, report:

- Total reward after communication cost
- Driving reward
- Communication cost
- Number of requests
- Number of no-request actions
- Episode length

Use multiple fixed random seeds and compare the trained DQN request policy against at least a random-request policy. Reward should be evaluated with learning disabled and exploration set to zero.
