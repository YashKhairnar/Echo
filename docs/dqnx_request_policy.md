# DQNx Multi-Request Policy

DQNx learns which visible neighboring vehicles should be queried at each
timestep. It can request any subset of the currently visible vehicles, or
request nobody. A request slot is positional within the current observation;
it is not a permanent vehicle ID.

## Observation and message design

The ego observation contains only locally visible kinematics:

```text
presence, x, y, vx, vy, vehicle_id
```

The requested vehicle sends information that is not included in that public
observation:

```text
intent acceleration, intent steering, target lane, time to collision
```

Each message slot therefore contains four values. With ten observation rows,
MLPx receives `60 + 5 + 10 + 40 = 115` input values and predicts a 60-value
next observation plus reward. The vehicle ID is kept for routing, logging, and
rendering; the model uses the request slot to identify the message.

TTC is capped at 10 seconds. A capped value means that no immediate collision
is predicted. It is calculated from simulator positions and velocities and
does not require a radar sensor.

## Decision flow

```text
observation
    ↓
DQNx outputs 10 × 2 Q-values
    ↓
visible-slot mask removes cars outside the current frame
    ↓
request mask selects zero or more visible cars
    ↓
MLPx + five-step MCP planner selects the driving action
    ↓
real environment returns next observation and driving reward
    ↓
communication cost is subtracted and the real transition trains DQNx
```

The DQNx policy chooses communication requests only. The driving action is
chosen by the MCP planner using the trained MLPx world model.

## Output and masking

For each of the ten request slots, DQNx outputs two values:

```text
[Q(no request), Q(request)]
```

Only slots containing visible neighboring vehicles can be requested. Invalid
slots are forced to the no-request choice. The renderer highlights every
vehicle selected by the request mask and shows its slot, vehicle ID, and
intent.

## Reward

The training reward is:

```text
total reward = driving reward − 0.1 × number of requested vehicles
```

The current cost is defined in `scripts/train_DQNx.py` as
`COMMUNICATION_COST = 0.1`. A policy that requests nobody can therefore have a
reasonable reward while using no communication. This is expected behavior,
but it should be compared with a no-request baseline and a useful-request
baseline before concluding that the policy has learned good communication.

## Training

Train from a fresh run:

```bash
PYTHONPATH=. .venv/bin/python -m scripts.train_DQNx \
    --mlpx-checkpoint checkpoints/MLPx/best.pt
```

The script trains DQNx on real environment transitions. MLPx predictions are
used by MCP to choose the driving action, but imagined transitions are not
added to the DQN replay buffer.

The active checkpoints are:

```text
checkpoints/DQNx/latest.pt
checkpoints/DQNx/best.pt
```

Each run also receives an automatic folder under
`results/DQNx/experiments/`, containing the CSV metrics, graphs, metadata,
notes, and checkpoint copies. Resume without naming a run folder:

```bash
PYTHONPATH=. .venv/bin/python -m scripts.train_DQNx \
    --resume checkpoints/DQNx/latest.pt \
    --episodes 2000
```

The resume checkpoint restores the online network, target network, optimizer,
epsilon, episode position, best reward, and replay buffer.

## Metrics

The training CSV and graphs record:

- total reward after communication cost;
- DQNx loss;
- exploration probability;
- episode length;
- number of requests; and
- total communication cost.

Increasing loss means the Q-value estimates are unstable; it is not by itself
evidence that the driving behavior is improving. Evaluate with exploration
disabled and compare several fixed seeds.

## Rendering and video

The renderer is implemented in `scripts/render_DQNx.py`. It runs the trained
DQNx policy without updating its weights, while MLPx and MCP continue to select
the driving action for the real environment.

Render the current best policy:

```bash
PYTHONPATH=. .venv/bin/python -m scripts.render_DQNx
```

Record an MP4 from a normal graphical terminal:

```bash
PYTHONPATH=. .venv/bin/python -m scripts.render_DQNx \
    --episodes 1 \
    --delay 0.1 \
    --video docs/images/dqnx_current_policy.mp4
```

Useful options are `--seed` for reproducible episodes, `--fps` for video
speed, `--width` and `--height` for window size, and `--dqnx-checkpoint` or
`--mlpx-checkpoint` for selecting different checkpoints.

The saved current-policy video is available at
`docs/images/dqnx_current_policy.mp4` when it has been generated locally and
included in the repository. Because the current learned policy may select no
requests, a valid video can show vehicle motion without yellow request
highlights.

### Renderer options

Useful command-line options are:

| Option | Purpose | Default |
| --- | --- | --- |
| `--episodes` | Number of episodes to render | `1` |
| `--delay` | Delay between environment steps in seconds | `0.1` |
| `--width` | Window/video width | `1600` |
| `--height` | Window/video height | `900` |
| `--fps` | Video frame rate | `10` |
| `--seed` | Reproduce the same initial episodes | random |
| `--video PATH` | Save frames as an MP4 | disabled |
| `--dqnx-checkpoint PATH` | DQNx checkpoint to load | `checkpoints/DQNx/best.pt` |
| `--mlpx-checkpoint PATH` | MLPx checkpoint to load | `checkpoints/MLPx/best.pt` |
| `--headless` | Use RGB-array rendering without a visible window | disabled |

Example with a reproducible seed and a slower video:

```bash
PYTHONPATH=. .venv/bin/python -m scripts.render_DQNx \
    --episodes 3 \
    --seed 42 \
    --delay 0.2 \
    --fps 5 \
    --video docs/images/dqnx_policy_seed42.mp4
```

The overlay displays the selected driving action, number of requested cars,
request slots, actual vehicle IDs, and each requested vehicle's acceleration,
steering, and lane-change intent. Requested visible vehicles are highlighted in
yellow.

The renderer uses greedy inference (`epsilon = 0`); it does not train DQNx or
change any checkpoint. The `--headless` mode requires a working HighwayEnv
RGB-array backend. On systems without a display backend it may create a video
with blank frames, so use a normal graphical terminal for reliable recordings.
