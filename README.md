# Echo: Cooperative Driving

A self-driving car learns **when to ask nearby cars for information, and whom
to ask**. Every request can cost something, so it should only ask when the
answer can change how it drives. Built on
[HighwayEnv](https://github.com/Farama-Foundation/HighwayEnv).

## Active scenario

The active environment has four lanes, one controlled ego vehicle, and ten
background vehicles. Background traffic uses HighwayEnv's aggressive and
defensive behavior classes. Vehicles have stable IDs, and the ego can inspect
up to eight nearby spatial slots within 60 m: left-front, left-beside,
left-rear, same-lane-front, same-lane-rear, and the corresponding right-lane
slots.

The older hidden-driver merge scenario is preserved in the documentation
archive as historical experiment material.

## The main parts

| Part | Job | Code |
| --- | --- | --- |
| **CustomEnv** | Creates the four-lane scene, vehicle types, IDs, visibility, and intent previews | `src/cooperative_driving/environments/customEnv.py` |
| **MLPx** (world model) | Predicts the next observation and reward from the current state, ego action, and ordered vehicle messages | `src/cooperative_driving/models/MLPx/mlpx.py` |
| **Planner** | Imagines three steps for each candidate ego action and picks the best one | `src/cooperative_driving/planner/MCP.py` |
| **Data-collection policy** | Uses visible neighbors and requested intents to choose one of five ego actions | `scripts/policy/intent_aware_policy.py` |
| **Request DQN** | Chooses one of eight vehicle slots or asks nobody | `src/cooperative_driving/policies/request_dqn.py` |
| **Renderer** | Shows the learned request policy, vehicle types, crashes, and requested vehicles | `scripts/evaluation/render_request_dqn.py` |

## Current snapshot

| Item | Current state |
| --- | --- |
| Combined dataset | 26,597 informed and random no-message transitions |
| MLPx input | 116 values: 55 state, 5 ego-action, and 56 message values |
| MLPx output | 55-value next state plus reward |
| Planner horizon | Three steps |
| Request DQN actions | Eight spatial request slots plus ask-nobody |
| Active checkpoints | `checkpoints/experiment_1/{MLPx,RequestDQN}/` |

Held-out MLPx evaluation currently reports one-step next-state RMSE `0.0677`
and reward RMSE `0.0828`. Open-loop state RMSE is `0.0935` at three steps and
`0.1201` at five steps, which is why the active planner uses three steps.

The current Request DQN training run logs return, loss, epsilon, request count,
crashes, and truncations. Its latest run still shows unstable loss and frequent
crashes, so it is an active experiment rather than a final policy.

## Project timeline

<a href="https://yashkhairnar.github.io/Echo/timeline.html"><img src="docs/images/project_timeline.svg" width="100%" alt="Echo project timeline"></a>

[Open the interactive timeline](https://yashkhairnar.github.io/Echo/timeline.html) directly.

## Run it

```bash
# 1. Collect intent-aware transitions
.venv/bin/python -m scripts.data_collection.collect_intent_aware_dataset \
  --episodes 100 --include-none --visibility 60 --experiment 1
# 2. Collect random no-message transitions
.venv/bin/python -m scripts.data_collection.collect_random_no_message_dataset \
  --episodes 1000 --visibility 60 --experiment 1
# 3. Merge informed and uninformed transition datasets
.venv/bin/python -m scripts.data_collection.merge_world_model_datasets --experiment 1
# 4. Train MLPx
.venv/bin/python -m scripts.training.train_mlpx_wm --experiment 1 --epochs 50
# 5. Train the request DQN with a frozen three-step MLPx planner
.venv/bin/python -m scripts.training.train_request_dqn \
  --experiment 1 --episodes 1000 --planning-horizon 3 --communication-cost 0.05
# 6. Render the learned request policy (default window: 1800x800)
.venv/bin/python -m scripts.evaluation.render_request_dqn --experiment 1 --episodes 1 --seed 42
# 7. Rebuild the HTML documentation site
.venv/bin/python -m scripts.docs.build_docs_site
```

## Docs

- [docs/index.html](docs/index.html): HTML documentation site
- [docs/mlpx.html](docs/mlpx.html): MLPx world-model definition and evaluation
- [docs/planner.html](docs/planner.html): three-step model-predictive planner
- [docs/policy.html](docs/policy.html): rule-based data-collection policy
- [docs/dqn_policy.html](docs/dqn_policy.html): learned Request DQN policy
- [docs/rendering.html](docs/rendering.html): renderer controls, colors, and highlights
- [docs/training.html](docs/training.html): MLPx and DQN training commands and graphs
- [docs/intent_aware_data_collection.html](docs/intent_aware_data_collection.html): current vehicle, intent, rule, and dataset workflow
- [docs/dataset_dashboard.html](docs/dataset_dashboard.html): dataset statistics dashboard
- [docs/progress_log.html](docs/progress_log.html): dated log of what was done
- [docs/archive/](docs/archive/): earlier highway and merge experiments

## Publish the docs

The repository includes a GitHub Pages workflow at
`.github/workflows/deploy-docs.yml`. In the repository settings, set
**Pages → Build and deployment → Source** to **GitHub Actions**. Every push to
`main` then publishes the `docs/` folder at:

<https://yashkhairnar.github.io/Echo/>
