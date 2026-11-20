# Rendering

The rendering script shows the trained Request DQN and the frozen three-step MLPx planner driving the HighwayEnv scene.

This page covers usage and display options. Experiment videos, episode
statistics, and policy comparisons live on the [DQN policy page](dqn_policy.html).

## Select the experiment

Run from the project root and select an experiment with both MLPx and DQN
checkpoints available:

```bash
ECHO_EXPERIMENT=2
```

The renderer loads that experiment's `MLPx/best.pt` and `RequestDQN/best.pt`
from `checkpoints/experiment_N/`. Use `--mlpx-checkpoint PATH` or
`--dqn-checkpoint PATH` to select explicit checkpoints.

## Run the renderer

```bash
.venv/bin/python -m scripts.evaluation.render_request_dqn \
  --experiment "$ECHO_EXPERIMENT" \
  --episodes 1 \
  --seed 7 \
  --delay 0.15 \
  --width 1800 \
  --height 800
```

The default window is `1800x800` with scaling `6.0`. Change `--width`,
`--height`, or `--scaling` to adjust the view. The seed controls the initial
scene; use another `--seed` value to inspect a different episode. Multiple
episodes use consecutive seeds.

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

## Record a video

Recording requires `ffmpeg` on your PATH. Save an MP4 with the request/response
overlay using an experiment-specific filename:

```bash
SDL_VIDEODRIVER=dummy .venv/bin/python -m scripts.evaluation.render_request_dqn \
  --experiment "$ECHO_EXPERIMENT" \
  --episodes 1 \
  --seed 7 \
  --delay 0.01 \
  --video-fps 2 \
  --video "docs/images/dqn_experiment_${ECHO_EXPERIMENT}_policy.mp4"
```

The recording saves one frame per decision plus the terminal scene. At two
frames per second, playback advances two decisions per second; it is not a
continuous recording of the simulation's intermediate physics frames.
The default recording rate is eight frames per second unless overridden.

Recording overwrites the chosen file if it already exists. Use a new filename
to preserve an earlier recording, then embed the selected video and its
episode statistics in the corresponding experiment on the DQN policy page.

## Terminal-only rendering check

For a terminal-only smoke test:

```bash
.venv/bin/python -m scripts.evaluation.render_request_dqn \
  --experiment "$ECHO_EXPERIMENT" \
  --episodes 1 \
  --no-render
```
