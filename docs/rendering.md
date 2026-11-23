# Rendering & Visualizing Policies

The rendering tool visualizes the driving environment in real-time or renders MP4 policy videos. It displays vehicle trajectories, safety guard interventions, and communication queries between vehicles.

- **Main Script:** `scripts/evaluation/render_request_dqn.py`
- **Output Videos:** Saved to `docs/images/`

---

## 1. Interactive Visualizer (Watch Live)

Run this from the project root to open a live Pygame window:

```bash
.venv/bin/python -m scripts.evaluation.render_request_dqn \
  --experiment 7 \
  --episodes 1 \
  --seed 1200000 \
  --delay 0.15 \
  --safety-guard \
  --max-target-speed 30 \
  --lane-change-guard
```

- `--delay 0.15`: Pauses 0.15 seconds between steps for comfortable viewing.
- `--seed <N>`: Sets the initial road and traffic seed.

---

## 2. Headless Video Recording (Save to MP4)

To record a policy run to an `.mp4` file without opening a GUI window:

```bash
SDL_VIDEODRIVER=dummy .venv/bin/python -m scripts.evaluation.render_request_dqn \
  --experiment 7 \
  --episodes 1 \
  --seed 1200000 \
  --safety-guard \
  --max-target-speed 30 \
  --lane-change-guard \
  --video-fps 2 \
  --video "docs/images/dqn_experiment_7_policy.mp4"
```

---

## 3. Rendering Cut-In Challenge Scenarios

To render the matched-intent cut-in scenario where neighboring vehicles attempt an aggressive merge:

```bash
SDL_VIDEODRIVER=dummy .venv/bin/python -m scripts.evaluation.render_request_dqn \
  --experiment 7 \
  --scenario paired-intent \
  --hidden-mode merge \
  --seed 1300051 \
  --safety-guard \
  --max-target-speed 30 \
  --lane-change-guard \
  --video-fps 2 \
  --video "docs/images/cut_in_demo.mp4"
```

---

## What the Visualizer Shows

### Vehicle Colors
- 🟩 **Green:** Ego vehicle (controlled by MLPx World Model + Safety Guard)
- 🟧 **Orange:** Aggressive vehicle
- 🟦 **Blue:** Defensive vehicle
- 🟨 **Yellow:** Standard IDM background vehicle
- 🟥 **Red:** Any vehicle involved in a collision

### Communication Overlays
- **Yellow Circle & Label:** Highlights the specific neighboring vehicle that the ego car queried for intent at that timestep.
- **Top HUD Panel:** Shows the ego's current speed, chosen driving action, the queried vehicle ID, and its received intent message (`target_lane`, `target_speed`, `acceleration`).
