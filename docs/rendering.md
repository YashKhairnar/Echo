# Rendering

The rendering script shows the trained Request DQN and the frozen three-step MLPx planner driving the HighwayEnv scene.

## Run the renderer

```bash
.venv/bin/python -m scripts.evaluation.render_request_dqn \
  --episodes 1 \
  --delay 0.15 \
  --width 1400 \
  --height 600
```

The default window is now larger than the original HighwayEnv view. You can customize it:

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

The terminal also prints the visible vehicle count, requested vehicle ID, and ego action selected by the planner.

For a terminal-only smoke test:

```bash
.venv/bin/python -m scripts.evaluation.render_request_dqn \
  --episodes 1 \
  --no-render
```
