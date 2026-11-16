# MLPx Rollout Evaluation

The MLPx world model was tested against real `HighwayEnv` transitions. From
the same sampled state, the evaluator compared all five driving actions over a
short constant-action rollout.

Run the evaluation with:

```bash
PYTHONPATH=. .venv/bin/python -m scripts.evaluate_mlpx_rollout \
    --episodes 10 \
    --states-per-episode 3 \
    --horizon 3 \
    --message-mode none
```

The evaluator reports next-state MAE, rollout reward MAE, and action-ranking
agreement: how often MLPx selects the same best action as the real environment.

## Results

Some sampled episodes ended before the complete horizon, so only complete
rollouts were included.

| Horizon | Message mode | State MAE | Reward MAE | Action-ranking agreement |
| ---: | --- | ---: | ---: | ---: |
| 5 | none | `0.238` | `0.723` | `23.1%` (`3/13`) |
| 5 | all | `0.377` | `0.635` | `23.1%` (`3/13`) |
| 3 | none | `0.229` | `0.336` | `66.7%` (`8/12`) |
| 3 | all | `0.245` | `0.383` | `42.9%` (`6/14`) |

## Interpretation

Three-step planning is currently more reliable than five-step planning. The
shorter horizon reduces accumulated prediction error and improves reward
prediction and action ranking.

The active MCP planner therefore uses a three-step horizon. MLPx is useful for
short-horizon planning, but it should not yet be treated as a reliable
five-step simulator.

Detailed evaluation files are saved under `results/MLPx/`:

```text
mlpx_rollout_evaluation.csv
mlpx_rollout_evaluation_all.csv
mlpx_rollout_evaluation_3step_none.csv
mlpx_rollout_evaluation_3step_all.csv
```

## Current checkpoint evaluation

The current `checkpoints/MLPx/best.pt` was evaluated again with the active
three-step horizon. This checkpoint was trained before the WM dataset was
expanded to 5,000 episodes.

| Message mode | State MAE | Three-step reward MAE | Action-ranking agreement |
| --- | ---: | ---: | ---: |
| None | `0.187` | `0.426` | `70.6%` (`12/17`) |
| All visible messages | `0.195` | `0.294` | `45.5%` (`5/11`) |

The model is reasonably accurate for short no-message rollouts, but action
ranking is less reliable when all messages are included. The next evaluation
should be performed after retraining MLPx on the expanded 5,000-episode WM
dataset.

Detailed current-checkpoint results:

```text
results/MLPx/mlpx_rollout_evaluation_current_3step_none.csv
results/MLPx/mlpx_rollout_evaluation_current_3step_all.csv
```
