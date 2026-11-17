# Echo: Cooperative Driving

A self-driving car learns **when to ask other cars for information, and whom
to ask**. Every question costs something, so it should only ask when the answer
changes how it drives. Built on [HighwayEnv](https://github.com/Farama-Foundation/HighwayEnv).

## The scenario

The ego's lane is blocked by a stalled car, so it has to merge into a line of
traffic. Each car in that line is secretly **yielding** (lets you in) or
**aggressive** (speeds up and closes the gap). They look the same until you
try to merge, so the only way to know in advance is to ask.

## The three parts

| Part | Job | Code |
| --- | --- | --- |
| **MLPx** (world model) | Predicts what happens next from what the car sees plus any messages, including "crash within 3 steps?" | `src/cooperative_driving/models/MLPx/mlpx.py` |
| **Planner** | Tries out moves in MLPx and picks the best one | `src/cooperative_driving/planner/MCP.py` |
| **Request DQN** | Each step: ask one car, or nobody | `src/cooperative_driving/policies/DQN.py` |

The environment, vehicles and messages are in
`src/cooperative_driving/environments/`, `vehicles/` and `communication/`.

## Current results

200 unseen episodes, cost 0.05 per question:

| Policy | Total score | Crashes | Questions / episode |
| --- | ---: | ---: | ---: |
| ask nobody | 5.79 | 18% | 0 |
| ask everyone | 2.03 | 0% | 152 |
| rear-car rule (hand-written) | **8.47** | 0% | 6.7 |
| **request DQN** (average of 3 training runs) | **7.09** | 3% | 18.2 |

The planner uses messages well, and the DQN has learned to ask (crashes fall
from 18% to about 3%), but it asks far too often and is still below the
hand-written rule. Details: [docs/merge_scenario.md](docs/merge_scenario.md).

## Timeline

Click the graphic for the interactive version, where each step opens to show what changed, the result, and what it taught us.

<a href="https://yashkhairnar.github.io/Echo/timeline.html"><img src="docs/images/project_timeline.svg" width="100%" alt="Project timeline: highway experiments (Nov 5–14) where asking never helped, the turning point, then the merge scenario (Nov 15) where the world model learned to use messages and request DQN run 3 became the first policy that learned to ask."></a>

## Run it

```bash
# 1. Check that the hidden information matters (hand-written rules, no learning)
.venv/bin/python -m scripts.check_information_gap --episodes 300
# 2. Collect world-model data
.venv/bin/python -m scripts.collect_multi_request_data --episodes 10000
# 3. Train MLPx
.venv/bin/python -m scripts.train_mlpx_wm --epochs 50
# 4. Train the request DQN (MLPx and the planner stay frozen)
.venv/bin/python -m scripts.train_DQN --episodes 3000 --communication-cost 0.05
# 5. Compare request policies on unseen seeds
.venv/bin/python -m scripts.evaluate_merge_policies --episodes 200 --communication-cost 0.05
# 6. Watch it drive (--policy dqn | rear | none | all; add --video out.mp4 to save)
.venv/bin/python -m scripts.render_DQN --episodes 3
```

## Docs

- [docs/merge_scenario.md](docs/merge_scenario.md): how it works, results, history, next steps
- [docs/progress_log.md](docs/progress_log.md): dated log of what was done
- [docs/archive/](docs/archive/): earlier highway experiments ([summary](docs/archive/earlier_experiments.md)) and the full merge history
