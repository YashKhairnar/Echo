# Training Runbook

This guide covers the step-by-step workflow for data collection, world-model training, request-policy learning, and evaluation.

---

## Workflow Overview

```text
Step 1: Collect Data       Step 2: Train MLPx        Step 3: Evaluate MLPx
┌────────────────────┐     ┌───────────────────┐     ┌───────────────────┐
│ Intent & Random    │ ──> │ Neural World      │ ──> │ 3-Step Rollout    │
│ Transitions (JSONL)│     │ Model (MSE Loss)  │     │ Agreement Check   │
└────────────────────┘     └───────────────────┘     └───────────────────┘
                                                               │
                                                               ▼
Step 6: Render Live        Step 5: Evaluate Policy   Step 4: Train Request DQN
┌────────────────────┐     ┌───────────────────┐     ┌───────────────────┐
│ Visualizer with    │ <── │ 4-Way Policy      │ <── │ Q-Learning with   │
│ Intent Overlays    │     │ Comparison        │     │ Frozen MLPx       │
└────────────────────┘     └───────────────────┘     └───────────────────┘
```

Set the active experiment ID in your terminal:
```bash
export ECHO_EXPERIMENT=5
```

---

## Step 1: Collect Transitions

Collect intent-aware transitions where neighbors preview their planned actions:

```bash
# 1. Collect intent-informed transitions (including no-request steps)
.venv/bin/python -m scripts.data_collection.collect_intent_aware_dataset \
  --experiment "$ECHO_EXPERIMENT" --episodes 1000 --include-none --visibility 60

# 2. Collect random no-message baseline transitions
.venv/bin/python -m scripts.data_collection.collect_random_no_message_dataset \
  --experiment "$ECHO_EXPERIMENT" --episodes 1000 --visibility 60

# 3. Merge both into a single training dataset
.venv/bin/python -m scripts.data_collection.merge_world_model_datasets \
  --experiment "$ECHO_EXPERIMENT"
```

*Data is written to:* `data/experiment_<id>/combined_world_model_dataset.jsonl`

---

## Step 2: Train the MLPx World Model

Train the multi-layer perceptron to jointly predict the next road state and immediate driving reward:

```bash
.venv/bin/python -m scripts.training.train_mlpx_wm \
  --experiment "$ECHO_EXPERIMENT" \
  --epochs 50 \
  --batch-size 128 \
  --lr 1e-3 \
  --balance-by action-message-stage
```

*The best checkpoint is saved to:* `checkpoints/experiment_<id>/MLPx/best.pt`

---

## Step 3: Evaluate World Model Rollouts

Verify open-loop rollout accuracy and action agreement before training downstream policies:

```bash
.venv/bin/python -m scripts.evaluation.evaluate_mlpx_rollout \
  --experiment "$ECHO_EXPERIMENT" \
  --episodes 100 \
  --horizon 3
```

- **Pass criteria:** 3-step action agreement should exceed **80%** against the ground-truth simulator.

---

## Step 4: Train the Request DQN

Train the Q-learning policy to decide which neighbor to query, using the frozen MLPx planner for driving actions:

```bash
.venv/bin/python -m scripts.training.train_request_dqn \
  --experiment "$ECHO_EXPERIMENT" \
  --episodes 1000 \
  --planning-horizon 3 \
  --communication-cost 0.05 \
  --safety-guard --max-target-speed 30 --lane-change-guard
```

### Resuming Training
To continue training an existing model for an additional 1,000 episodes:
```bash
.venv/bin/python -m scripts.training.train_request_dqn \
  --experiment "$ECHO_EXPERIMENT" \
  --resume \
  --episodes 1000
```

*The trained model is saved to:* `checkpoints/experiment_<id>/RequestDQN/best.pt`

### Selective-request training

Experiment 6 adds a training-only teacher penalty when a requested response
does not change the planner's driving action. The deployed DQN still observes
only public state.

```bash
.venv/bin/python -m scripts.training.train_request_dqn \
  --experiment 6 --episodes 1000 --seed 0 \
  --mlpx-checkpoint checkpoints/experiment_5/MLPx/intent_stage_best.pt \
  --scenario-mix mixed --ordinary-seed 1000000 --challenge-seed 1100000 \
  --request-state-encoding ego_control_v1 \
  --communication-cost 0.05 --redundant-request-penalty 0.10 \
  --safety-guard --max-target-speed 30 --lane-change-guard --batched-planner
```

---

## Step 5: Evaluate Policy Performance

Compare the trained Request DQN against three baselines across identical test seeds:
1. **Ask Nobody (Blind):** Zero communication cost.
2. **Ask Random:** Queries a random visible vehicle.
3. **Ask All:** Queries every visible vehicle.

```bash
.venv/bin/python -m scripts.evaluation.evaluate_request_dqn \
  --experiment "$ECHO_EXPERIMENT" \
  --episodes 100 \
  --planning-horizon 3 \
  --safety-guard --max-target-speed 30 --lane-change-guard
```

*Summary JSON & CSV are written to:* `results/experiment_<id>/evaluation/request_dqn_guarded/`

---

## Step 6: Visual Rendering & Inspection

Launch the interactive renderer to watch the car drive with live visual overlays showing requested vehicle IDs, replied intentions, and safety-guard interventions:

```bash
.venv/bin/python -m scripts.evaluation.render_request_dqn \
  --experiment "$ECHO_EXPERIMENT" \
  --episodes 1 \
  --seed 42 \
  --safety-guard --max-target-speed 30 --lane-change-guard
```

---

## Artifacts & Directory Structure

```text
Echo/
├── checkpoints/experiment_<id>/
│   ├── MLPx/best.pt              # Trained world model
│   └── RequestDQN/best.pt        # Trained communication policy
├── data/experiment_<id>/
│   ├── intent_aware_dataset.jsonl
│   ├── random_no_message_dataset.jsonl
│   └── combined_world_model_dataset.jsonl
└── results/experiment_<id>/
    ├── MLPx/                     # Training loss logs and curves
    ├── RequestDQN/               # DQN rewards, epsilons, and losses
    └── evaluation/               # Rollout metrics, benchmark comparisons
```
