# Training

This page is the run guide for collecting data, training, and evaluating an
experiment. Results and graphs live on their topic pages:

| Topic | Documentation |
| --- | --- |
| Dataset sources, statistics, and coverage | [Dataset dashboard](dataset_dashboard.html) |
| MLPx configuration, losses, graphs, and rollout results | [MLPx world model](mlpx.html) |
| DQN configuration, losses, graphs, evaluation, and videos | [Request DQN](dqn_policy.html) |
| Interactive rendering and recording | [Rendering](rendering.html) |
| Experiment history and implementation changes | [Progress log](progress_log.html) |

## Select the experiment

Run commands from the project root. Set the same experiment number for every
stage in the terminal session:

```bash
ECHO_EXPERIMENT=3
```

Use a new number for a new experiment to preserve previous results. Training
creates a timestamped run directory, but replaces the selected experiment's
active `best.pt`. Collection and evaluation overwrite their default output
files when rerun.

Experiments 1 and 2 contain historical packed informed-message records. The
corrected adapter rejects those records because missing spatial positions
cannot be recovered from vehicle IDs alone. Experiment 3 uses corrected
eight-slot encoding; its empty slots remain in place. Existing no-message
records remain usable.

## 1. Prepare the dataset

Collect informed transitions, including the no-request case:

```bash
.venv/bin/python -m scripts.data_collection.collect_intent_aware_dataset \
  --experiment "$ECHO_EXPERIMENT" --episodes 1000 --include-none
```

Either collect a new random baseline:

```bash
.venv/bin/python -m scripts.data_collection.collect_random_no_message_dataset \
  --experiment "$ECHO_EXPERIMENT" --episodes 1000

.venv/bin/python -m scripts.data_collection.merge_world_model_datasets \
  --experiment "$ECHO_EXPERIMENT"
```

Or reuse the Experiment 2 no-message baseline, as Experiment 3 did:

```bash
.venv/bin/python -m scripts.data_collection.merge_world_model_datasets \
  --experiment "$ECHO_EXPERIMENT" \
  --random data/experiment_2/random_no_message_dataset.jsonl
```

Choose one baseline route. See [Intent-aware collection](intent_aware_data_collection.html)
for the collection process and [Dataset dashboard](dataset_dashboard.html)
for each experiment's saved data.

## 2. Train MLPx

```bash
.venv/bin/python -m scripts.training.train_mlpx_wm \
  --experiment "$ECHO_EXPERIMENT" \
  --epochs 50 --batch-size 128 --lr 1e-3
```

The trainer selects the checkpoint with the lowest held-out loss.
See [MLPx](mlpx.html) for architecture, per-experiment settings, and results.

## 3. Evaluate MLPx

```bash
.venv/bin/python -m scripts.evaluation.evaluate_mlpx_rollout \
  --experiment "$ECHO_EXPERIMENT" \
  --episodes 100 --states-per-episode 5 \
  --sample-every 10 --action-mode planner
```

Inspect the results before training DQN. The corrected evaluator tests the
model's preferred continuation for each first action, retains early-terminal
outcomes, and assigns zero future real reward after an episode ends.
It does not exhaustively search every action sequence in the real simulator.
Historical Experiment 1/2 rollout tables used a different evaluation protocol;
see the model page for interpretation.

## 4. Train the request DQN

```bash
.venv/bin/python -m scripts.training.train_request_dqn \
  --experiment "$ECHO_EXPERIMENT" \
  --episodes 1000 --planning-horizon 3 --communication-cost 0.05
```

The trainer loads the same experiment's MLPx checkpoint and keeps it frozen.
The corrected optimizer masks unavailable next requests in its target values.
See [Request DQN](dqn_policy.html) for the policy design and experiment results.

### Resume DQN training

Continue the selected experiment for 1,000 additional episodes:

```bash
.venv/bin/python -m scripts.training.train_request_dqn \
  --experiment "$ECHO_EXPERIMENT" --resume --episodes 1000 --save-every 50
```

For Experiment 3, set `ECHO_EXPERIMENT=3`. A checkpoint completed at episode
1,000 continues with episode 1,001 and trains through episode 2,000.
`--episodes` always counts additional episodes for this invocation.

`--resume` loads `latest.pt` beside the active DQN checkpoint, falling back to
`best.pt` when no full-state checkpoint exists. To select a specific save:

Experiment 3 currently selects the 1,000-episode model as inference best.pt,
while latest.pt retains the 2,000-episode training state. Plain --resume
continues from 2,000. To start a new continuation from the selected 1,000-episode
policy, explicitly pass --resume checkpoints/experiment_3/RequestDQN/best.pt;
that file supports weights-only continuation. Keep new runs' evaluation output
separate when comparing checkpoints.

```bash
.venv/bin/python -m scripts.training.train_request_dqn \
  --experiment "$ECHO_EXPERIMENT" \
  --resume "checkpoints/experiment_${ECHO_EXPERIMENT}/RequestDQN/latest.pt" \
  --episodes 1000
```

New `latest.pt` files restore online and target networks, Adam state, replay
memory, epsilon, episode count, return history, and Python/NumPy/PyTorch random
states. They restore training settings, including seed, learning rate, batch
size, planning horizon, and request cost, from the saved configuration. The
trainer verifies that the frozen World Model file has not changed.

The selected Experiment 3 best.pt is weights-only: its first continuation
restores weights and episode count but restarts optimizer, target, replay,
and random states. Epsilon is inferred from the original decay schedule
(`0.05` after 1,000 episodes). A warning identifies this partial continuation.
Compressed-action checkpoints cannot be resumed into fixed spatial actions.

Each invocation creates a new timestamped run directory and CSV with cumulative
episode numbers. Full-state saves occur at startup, every `--save-every`
completed episodes, and on successful completion. Ctrl-C leaves the last saved
boundary available; episodes after that boundary must be repeated. Set
`--save-every 1` for a save after every episode, at the cost of extra disk IO.

Full saves are written to the run directory and beside the active output as
`latest.pt`. Successful completion updates the active `best.pt` and the run's
`best.pt` with final inference weights. Despite its name, this DQN file is not
selected by validation score. Graph export follows the latest run's resume
history and combines related logs into one full-episode graph, marking resume
boundaries. It checks action encoding and consecutive episode numbers. If
more than one parent run is possible, pass the chronological CSV paths with
`--dqn-logs` rather than guessing which runs belong together.

## 5. Evaluate the request DQN

```bash
.venv/bin/python -m scripts.evaluation.evaluate_request_dqn \
  --experiment "$ECHO_EXPERIMENT" \
  --episodes 200 --planning-horizon 3
```

This compares DQN, no-request, random-request, and ask-all policies on the same
seed range. Review return, driving reward, request counts, and crash rate
together.

## 6. Inspect the policy

```bash
.venv/bin/python -m scripts.evaluation.render_request_dqn \
  --experiment "$ECHO_EXPERIMENT" --episodes 1 --seed 7
```

The frame displays the requested vehicle ID, intent response, and ego action.
See [Rendering](rendering.html) for video recording commands.

## Output locations

| Artifact | Path |
| --- | --- |
| Combined dataset | `data/experiment_N/combined_world_model_dataset.jsonl` |
| Active MLPx checkpoint | `checkpoints/experiment_N/MLPx/best.pt` |
| Active DQN checkpoint | `checkpoints/experiment_N/RequestDQN/best.pt` |
| Resumable DQN training state | `checkpoints/experiment_N/RequestDQN/latest.pt` |
| MLPx run logs, checkpoints, and loss plot | `results/experiment_N/MLPx/experiments/<run>/` |
| DQN run logs and checkpoint | `results/experiment_N/RequestDQN/experiments/<run>/` |
| MLPx evaluation CSV and JSON | `results/experiment_N/evaluation/mlpx_rollout/` |
| DQN evaluation CSV and JSON | `results/experiment_N/evaluation/request_dqn/` |

`N` is the selected experiment number. Earlier archived outputs remain under
`results/archive/`.

## Export documentation graphs

Export dataset charts after collection:

```bash
.venv/bin/python -m scripts.docs.plot_dataset_graphs \
  --experiment "$ECHO_EXPERIMENT"
```

Export matching MLPx and DQN training charts once both training logs exist:

```bash
.venv/bin/python -m scripts.docs.plot_training_graphs \
  --experiment "$ECHO_EXPERIMENT"
```

The MLPx trainer also saves its loss plot in the run directory immediately
after training. Rebuild HTML after editing documentation sources:

```bash
.venv/bin/python -m scripts.docs.build_docs_site
```
