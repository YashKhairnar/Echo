# Experiment Tracking

Each new MLP or DQN training run creates a timestamped folder containing:

```text
config.json       hyperparameters, dataset, checkpoint, and git commit
notes.md          space for observations and conclusions
training.csv      per-epoch or per-episode metrics
graphs            saved training plots
checkpoints       copy of the trained checkpoint
```

The top-level result files remain available for convenience. They are not a replacement for the experiment folders; the folders preserve each run independently.

MLP runs are stored under:

```text
results/MLP/experiments/
```

DQN runs are stored under:

```text
results/dqn/experiments/
```

Run training normally to create a new experiment folder:

```bash
PYTHONPATH=. .venv/bin/python -m scripts.train_mlp_wm
PYTHONPATH=. .venv/bin/python -m scripts.train_DQN
```

To continue an MLP experiment, pass its original experiment directory and log:

```bash
PYTHONPATH=. .venv/bin/python -m scripts.train_mlp_wm \
    --resume checkpoints/mlp/latest.pt \
    --experiment-dir results/MLP/experiments/<run-name> \
    --log results/MLP/experiments/<run-name>/mlp_training.csv
```

After each run, edit `notes.md` with what changed, what worked, and what should be improved.

The latest resumed DQN experiment is stored at:

```text
results/dqn/experiments/dqn_resume_1000_2000/
```

Its results should be interpreted using both reward and communication cost. A higher reward with fewer requests is preferable, but increasing DQN loss should be investigated before treating the policy as fully converged.

## Resuming DQN training

DQN checkpoints restore the networks, optimizer, exploration probability, episode number, and replay buffer. Resume using the same experiment directory and log:

```bash
PYTHONPATH=. .venv/bin/python -m scripts.train_DQN \
    --resume results/dqn/experiments/<run-name>/checkpoints/latest.pt \
    --experiment-dir results/dqn/experiments/<run-name> \
    --log results/dqn/experiments/<run-name>/dqn_training.csv \
    --episodes 2000
```
