# Experiment Tracking

Every training run now creates an automatically named experiment folder. The
folder name contains a timestamp and model name, so fresh runs never overwrite
older runs:

```text
results/
├── MLP/
│   └── experiments/<timestamp>_mlp/
├── MLPx/
│   └── experiments/<timestamp>_mlpx/
├── DQN/
│   └── experiments/<timestamp>_dqn/
└── DQNx/
    └── experiments/<timestamp>_dqnx/
```

Each experiment folder contains its fixed filenames:

```text
results/<model>/experiments/<run-name>/
├── config.json
├── notes.md
├── <model>_training.csv
├── training graphs
├── best.pt
└── latest.pt
```

The active checkpoint shortcuts remain in:

```text
checkpoints/
├── MLP/
├── MLPx/
├── DQN/
└── DQNx/
```

Train the new multi-request models with:

```bash
PYTHONPATH=. .venv/bin/python -m scripts.train_mlpx_wm \
    --data data/randomMultiRequestData_ttc.csv

PYTHONPATH=. .venv/bin/python -m scripts.train_DQNx \
    --mlpx-checkpoint checkpoints/MLPx/best.pt
```

The multi-request collector used for the MLPx dataset is:

```bash
PYTHONPATH=. .venv/bin/python -m scripts.collect_multi_request_data
```

The active TTC dataset is `data/randomMultiRequestData_ttc.csv`. Older datasets
are archived under `archive/`.

You do not need to provide a folder name for a normal run. The scripts create
the run folder automatically. `--experiment-dir` remains available only when
you explicitly need a custom location.

Run training normally to create a new experiment folder automatically:

The original single-request commands are archived in
`archive/legacy_single_request/`. The active multi-request commands are
documented in [the DQNx experiment guide](dqnx_request_policy.md).

To continue the latest MLPx experiment, pass only its active checkpoint:

```bash
PYTHONPATH=. .venv/bin/python -m scripts.train_mlpx_wm \
    --resume checkpoints/MLPx/latest.pt
```

After each run, edit `notes.md` with what changed, what worked, and what should be improved.

Its results should be interpreted using both reward and communication cost. A higher reward with fewer requests is preferable, but increasing DQN loss should be investigated before treating the policy as fully converged.

## Resuming DQNx training

DQNx checkpoints restore the networks, optimizer, exploration probability,
episode number, and replay buffer:

```bash
PYTHONPATH=. .venv/bin/python -m scripts.train_DQNx \
    --resume checkpoints/DQNx/latest.pt \
    --episodes 2000
```
