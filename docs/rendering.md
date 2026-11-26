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
  --dqn-checkpoint checkpoints/experiment_7/best.pt \
  --mlpx-checkpoint checkpoints/experiment_5/MLPx/intent_stage_best.pt \
  --episodes 1 \
  --seed 120 \
  --duration 100 \
  --delay 0.15 \
  --safety-guard \
  --max-target-speed 30 \
  --lane-change-guard \
  --comfort-ttc 6 \
  --batched-planner
```

- `--delay 0.15`: Pauses 0.15 seconds between steps for comfortable viewing.
- `--seed <N>`: Sets the initial road and traffic seed.

---

## 2. Headless Video Recording (Save to MP4)

To record a policy run to an `.mp4` file without opening a GUI window:

```bash
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
.venv/bin/python -m scripts.evaluation.render_request_dqn \
  --experiment 7 \
  --dqn-checkpoint checkpoints/experiment_7/best.pt \
  --mlpx-checkpoint checkpoints/experiment_5/MLPx/intent_stage_best.pt \
  --episodes 1 \
  --seed 120 \
  --duration 100 \
  --delay 0 \
  --safety-guard \
  --max-target-speed 30 \
  --lane-change-guard \
  --comfort-ttc 6 \
  --batched-planner \
  --video-fps 8 \
  --video "docs/images/dqn_experiment_7_policy.mp4"
```

All generic Experiment 1–8 policy videos use seed **120** and a maximum duration of **100 decision steps** so their behavior is visually comparable on the same initial road. Purpose-built cut-in demonstrations retain their selected challenge seeds because they illustrate specific hidden-intent events.

### Shared seed-120 video set

| Experiment | Outcome | Steps | Reward | Requests | Video |
| :--- | :--- | ---: | ---: | ---: | :--- |
| Experiment 1 | Crashed | 6 / 100 | 4.773 | 3 | [Watch](images/dqn_experiment_1_policy.mp4) |
| Experiment 2 | Crashed | 10 / 100 | 8.532 | 4 | [Watch](images/dqn_experiment_2_policy.mp4) |
| Experiment 3 | Crashed | 6 / 100 | 4.749 | 4 | [Watch](images/dqn_experiment_3_policy.mp4) |
| Experiment 4 | Drove to end | 100 / 100 | 87.145 | 74 | [Watch](images/dqn_experiment_4_guarded_policy.mp4) |
| Experiment 5 | Drove to end | 100 / 100 | 87.145 | 63 | [Watch](images/dqn_experiment_5_policy.mp4) |
| Experiment 6 | Drove to end | 100 / 100 | 90.661 | 52 | [Watch](images/dqn_experiment_6_policy.mp4) |
| Experiment 7 | Drove to end | 100 / 100 | 90.661 | 69 | [Watch](images/dqn_experiment_7_policy.mp4) |
| Experiment 8 | Drove to end | 100 / 100 | 90.836 | 82 | [Watch](images/dqn_experiment_8_policy.mp4) |

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
