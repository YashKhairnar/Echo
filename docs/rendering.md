# Rendering

The rendering script shows the trained Request DQN and the frozen three-step MLPx planner driving the HighwayEnv scene.

## Run the renderer

```bash
.venv/bin/python -m scripts.evaluation.render_request_dqn \
  --experiment 1 \
  --episodes 1 \
  --delay 0.15 \
  --width 1800 \
  --height 800
```

The default window is `1800x800` with scaling `6.0`. You can customize it:

```bash
--width 1800 --height 700 --scaling 5.5
```

## What the display shows

- Green: controlled ego vehicle
- Orange: aggressive vehicle
- Blue: defensive vehicle
- Yellow: IDM vehicle
- Red: any vehicle involved in a crash
- Yellow circle: the vehicle whose message was requested during that step
- Yellow ID label: identifies the requested vehicle on the road

The top overlay shows the episode, step, ego action, and requested vehicle ID.
It displays the exact response supplied to the planner: behavior type, target
lane (zero-based), target speed in m/s, acceleration in m/s², steering in
radians, and the crash flag. These are previewed intentions at request time.
When the ego asks nobody, it shows `Request: nobody` and `Response: None`.
The terminal frame displays the episode outcome and the last request/response.

The overlay is included in both the interactive window and saved videos.

The terminal also prints the visible vehicle count, requested vehicle ID, and ego action selected by the planner.
At the end of each episode it prints the completed step count, whether the ego
crashed or drove to the episode limit, total reward, and total requests sent.

To record an MP4 for the documentation or later inspection, pass `--video`:

```bash
SDL_VIDEODRIVER=dummy .venv/bin/python -m scripts.evaluation.render_request_dqn \
  --experiment 1 \
  --episodes 1 \
  --seed 7 \
  --video docs/images/dqn_experiment_1_policy.mp4
```

## Experiment 2 · Rendering and recording

Select Experiment 2 to load its DQN and MLPx checkpoints:

```bash
.venv/bin/python -m scripts.evaluation.render_request_dqn \
  --experiment 2 --episodes 1 --seed 7 --delay 0.15
```

Record the episode with its request/response overlay:

```bash
SDL_VIDEODRIVER=dummy .venv/bin/python -m scripts.evaluation.render_request_dqn \
  --experiment 2 \
  --episodes 1 \
  --seed 7 \
  --delay 0.01 \
  --video-fps 2 \
  --video docs/images/dqn_experiment_2_policy.mp4
```

The recording saves one frame per decision plus the terminal scene. At two
frames per second, playback advances two decisions per second; it is not a
continuous recording of the simulation's intermediate physics frames.
The default recording rate is eight frames per second unless overridden.

Seed 7 completed 14/40 steps before crashing, with driving reward 13.039 and
five requests. See the [DQN policy page](dqn_policy.html) for the embedded
video and the full evaluation results. The seed controls the initial scene;
use a different `--seed` value to inspect a different episode.

## Terminal-only rendering check

For a terminal-only smoke test:

```bash
.venv/bin/python -m scripts.evaluation.render_request_dqn \
  --experiment 1 \
  --episodes 1 \
  --no-render
```
