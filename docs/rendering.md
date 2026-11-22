# Rendering

The rendering script shows the trained Request DQN and the frozen three-step MLPx planner driving the HighwayEnv scene.

This page covers usage and display options. Experiment videos, episode
statistics, and policy comparisons live on the [DQN policy page](dqn_policy.html).

## Select the experiment

Run from the project root and select an experiment with both MLPx and DQN
checkpoints available:

```bash
ECHO_EXPERIMENT=6
```

The renderer loads `RequestDQN/best.pt` from the selected experiment. Modern
DQN checkpoints also store the exact frozen MLPx path used during training, so
Experiment 6 automatically reuses Experiment 5's `intent_stage_best.pt`.
Use `--mlpx-checkpoint PATH` or `--dqn-checkpoint PATH` only for deliberate
transfer tests.

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

To inspect the optional geometry guard, add --safety-guard to any rendering
command. The overlay and terminal identify braking or boundary-mask reasons,
and the episode summary includes override counts. See [Planner](planner.html)
for the guard rules and separate evaluation results. It is disabled by default.

The full optional guard also accepts `--max-target-speed 30` and
`--lane-change-guard`, both requiring `--safety-guard`. The overlay reports
speed-cap, destination-gap, maneuver-in-progress, braking, and boundary
override reasons. For a historical transfer example, inspect frozen WM4 with
the preserved Experiment 3 DQN:

```bash
.venv/bin/python -m scripts.evaluation.render_request_dqn \
  --experiment 4 --dqn-checkpoint checkpoints/experiment_3/RequestDQN/best.pt \
  --episodes 1 --seed 100220 \
  --safety-guard --max-target-speed 30 --lane-change-guard
```

This is a transfer configuration, not a trained Experiment 4 DQN. The same
flags work in `scripts.evaluation.evaluate_request_dqn`; see
[Request DQN](dqn_policy.html) for the paired baseline comparison.

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

## Render Experiment 6

Render the selective Experiment 6 policy with its training guard:

```bash
.venv/bin/python -m scripts.evaluation.render_request_dqn \
  --experiment 6 --episodes 1 --seed 1200000 --delay 0.15 \
  --safety-guard --max-target-speed 30 --lane-change-guard --comfort-ttc 6
```

Record the same policy with its request/response overlay:

```bash
.venv/bin/python -m scripts.evaluation.render_request_dqn \
  --experiment 6 --episodes 1 --seed 1200000 --delay 0 \
  --safety-guard --max-target-speed 30 --lane-change-guard --comfort-ttc 6 \
  --video docs/images/dqn_experiment_6_policy.mp4 --video-fps 8
```

The `--comfort-ttc 6` mode begins reacting before the 3-second emergency brake
threshold. It uses safe lane changes when they improve forward clearance,
holds a five-step lane-change cooldown, and blocks acceleration for three steps
after braking. The verified terminal-only run completed 40/40 steps without a
crash and earned 36.394 driving reward. One
episode is a visual inspection, not an aggregate policy benchmark; use the
[Experiment 6](experiment_6.html) evaluations for policy conclusions.

The DQN checkpoint supplies its versioned ego-control state encoding and
frozen intent-stage WM5 path. An explicit `--mlpx-checkpoint` overrides that
path for deliberate transfer tests. Older checkpoints without a stored path
use the selected experiment's default WM. Guard flags remain explicit; the
renderer warns if a guarded-trained DQN is rendered without the guard.

For a terminal-only smoke test without opening a window:

```bash
.venv/bin/python -m scripts.evaluation.render_request_dqn \
  --experiment "$ECHO_EXPERIMENT" \
  --episodes 1 \
  --no-render
```
