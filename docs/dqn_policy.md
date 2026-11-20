# Request DQN policy

The Request DQN is a learned communication policy. It decides **which visible
vehicle to ask**, or whether to ask nobody. It does not directly choose the
ego's driving action.

After the DQN selects a request, the frozen MLPx planner uses the returned
message to choose the ego action with a three-step planning horizon.

## Shared policy design

### Request actions

The DQN has nine possible actions:

```text
left-front, left-beside, left-rear,
same-lane-front, same-lane-rear,
right-front, right-beside, right-rear,
ask-nobody
```

Invalid spatial slots are masked during action selection. The DQN receives a
45-value state representation containing the ego and eight ordered neighbor
slots.

### Training setup

```text
DQN reward = real driving reward - communication cost when a vehicle is asked
```

The DQN uses replay memory, an online Q-network, a target Q-network,
epsilon-greedy exploration, and Huber loss. The target network is synchronized
every 50 episodes. Each experiment has its own checkpoint and result folder.

## Experiment 1 · Baseline request-policy training

Experiment 1 trained the Request DQN for 1,000 episodes with the Experiment 1
MLPx checkpoint frozen, a three-step planner, and a communication cost of
`0.05`.

```bash
.venv/bin/python -m scripts.training.train_request_dqn \
  --experiment 1 \
  --episodes 1000 \
  --planning-horizon 3 \
  --communication-cost 0.05
```

| Metric | Result |
| --- | ---: |
| Episodes | `1,000` |
| Mean return | `10.621` |
| Mean return, last 100 episodes | `12.167` |
| Best episode return | `19.526` at episode `643` |
| Mean training loss | `0.655` |
| Mean loss, last 100 episodes | `1.307` |
| Mean requests per episode | `5.392` |
| Mean requests, last 100 episodes | `5.460` |
| Episodes with crash flag | `826` (`82.60%`) |
| Episodes with truncation flag | `200` (`20.00%`) |
| Final epsilon | `0.05` |

The baseline shows improving late returns, but the crash rate and late loss
remain high. It is useful as a first communication-policy checkpoint, not as a
reliable deployment policy. The Experiment 1 log contains 26 episodes with
both flags, so the crash and truncation percentages are reported independently
and are not mutually exclusive.

### Experiment 1 validation

Evaluation compared the learned request policy against no-request,
random-request, and ask-all baselines over 200 episodes each, starting at seed
`100000`, with communication cost `0.05`:

```bash
.venv/bin/python -m scripts.evaluation.evaluate_request_dqn \
  --experiment 1 \
  --episodes 200 \
  --planning-horizon 3
```

| Request policy | Mean return | Mean driving reward | Mean requests per episode | Crashes | Crash rate | Truncations |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DQN | `14.955` | `15.207` | `5.040` | `184 / 200` | `92.00%` | `16` |
| Ask nobody | `13.070` | `13.070` | `0.000` | `198 / 200` | `99.00%` | `2` |
| Random requests | `6.941` | `7.359` | `8.355` | `200 / 200` | `100.00%` | `0` |
| Ask all visible | `4.331` | `4.959` | `12.560` | `200 / 200` | `100.00%` | `0` |

The DQN improves mean return over ask-nobody by `1.884` (about `14.41%`)
and reduces crashes by 14 episodes, or `7` percentage points. This evaluation
provides evidence that the learned request strategy helps the Experiment 1
system relative to its no-request baseline. The remaining `92%` crash rate is
still high, and uncertainty across independent training runs has not been
measured.

Results are saved under `results/experiment_1/evaluation/request_dqn/`.

### Experiment 1 rendering

The replacement rendering uses Experiment 1 checkpoints, seed `7`, and the
three-step planner:

| Metric | Result |
| --- | ---: |
| Steps completed | `20 / 40` |
| Outcome | Crashed |
| Crashed | Yes |
| Reward | `18.732` |
| Requests sent | `5` |

The recording is generated with:

```bash
SDL_VIDEODRIVER=dummy .venv/bin/python -m scripts.evaluation.render_request_dqn \
  --experiment 1 \
  --episodes 1 \
  --seed 7 \
  --delay 0.01 \
  --video docs/images/dqn_experiment_1_policy.mp4
```

![Experiment 1 Request DQN rendering](images/dqn_experiment_1_policy.mp4)

## Experiment 2 · Larger World Model DQN training

Experiment 2 trained the Request DQN for 1,000 episodes with the larger
Experiment 2 MLPx checkpoint frozen, the same three-step planner, and a
communication cost of `0.05`.

```bash
.venv/bin/python -m scripts.training.train_request_dqn \
  --experiment 2 \
  --episodes 1000 \
  --planning-horizon 3 \
  --communication-cost 0.05
```

| Metric | Result |
| --- | ---: |
| Episodes | `1,000` |
| Mean return | `10.263` |
| Mean return, last 100 episodes | `11.423` |
| Best episode return | `38.658` at episode `615` |
| Mean training loss | `0.618` |
| Mean loss, last 100 episodes | `1.407` |
| Mean requests per episode | `6.749` |
| Mean requests, last 100 episodes | `7.190` |
| Episodes with crash flag | `984` (`98.40%`) |
| Episodes with truncation flag | `16` (`1.60%`) |
| Final epsilon | `0.05` |

The return in the last 100 episodes was higher than the overall average, but
the very high crash rate and rising late-training loss show that this
checkpoint is not yet reliable. Request frequency also remains substantial,
so the policy has not yet demonstrated efficient communication.

The Experiment 2 checkpoint is saved at
`checkpoints/experiment_2/RequestDQN/best.pt`. Its training log is
`results/experiment_2/RequestDQN/experiments/20261119_170031_request_dqn/request_dqn_training.csv`.

### Experiment 2 validation results

Evaluation compared four policies on 200 episodes each, starting at seed
`100000`, with a three-step planning horizon and communication cost `0.05`:

```bash
.venv/bin/python -m scripts.evaluation.evaluate_request_dqn \
  --experiment 2 \
  --episodes 200 \
  --planning-horizon 3
```

| Request policy | Mean return | Mean driving reward | Mean requests per episode | Crashes | Crash rate | Truncations |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DQN | `12.326` | `12.556` | `4.615` | `189 / 200` | `94.50%` | `11` |
| Ask nobody | `11.973` | `11.973` | `0.000` | `186 / 200` | `93.00%` | `14` |
| Random requests | `7.692` | `8.144` | `9.050` | `199 / 200` | `99.50%` | `1` |
| Ask all visible | `6.855` | `7.705` | `16.985` | `200 / 200` | `100.00%` | `0` |

Return includes the communication penalty; driving reward excludes it. Request
counts are messages sent per episode, so ask-all can send multiple messages
in one step.

The DQN has the highest mean return, exceeding ask-nobody by `0.352`
(about `2.94%`), while sending fewer requests than random or ask-all.
However, it crashes in three more episodes than ask-nobody, a difference of
`1.5` percentage points. This run shows a small average return advantage,
but no safety advantage from communication. Statistical uncertainty has not
been quantified, so the return difference alone does not establish a reliable
improvement.

All four policies crash in at least 93% of episodes. The shared driving planner
and World Model need investigation alongside the request policy; these results
do not isolate which component causes the failures.

The evaluation CSV and JSON summary are saved under
`results/experiment_2/evaluation/request_dqn/`.

### Experiment 2 rendering

This recording uses the Experiment 2 DQN and MLPx checkpoints, seed `7`, and
the three-step planner. Playback is two frames per second, with one frame per
decision plus the final scene, so the actions are easier to inspect.

| Metric | Result |
| --- | ---: |
| Steps completed | `14 / 40` |
| Outcome | Crashed |
| Driving reward | `13.039` |
| Requests sent | `5` |

This is one illustrative episode; the 200-episode evaluation above describes
overall performance.

```bash
SDL_VIDEODRIVER=dummy .venv/bin/python -m scripts.evaluation.render_request_dqn \
  --experiment 2 \
  --episodes 1 \
  --seed 7 \
  --delay 0.01 \
  --video-fps 2 \
  --video docs/images/dqn_experiment_2_policy.mp4
```

![Experiment 2 Request DQN rendering](images/dqn_experiment_2_policy.mp4)

## Comparison across experiments

Both evaluations used 200 episodes starting at seed `100000`, a three-step
planner, and communication cost `0.05`. Each experiment used its own MLPx and
DQN checkpoints, so this compares the complete systems rather than isolating
the DQN or the World Model.

| Metric | Experiment 1 DQN | Experiment 2 DQN |
| --- | ---: | ---: |
| Mean return | `14.955` | `12.326` |
| Mean requests per episode | `5.040` | `4.615` |
| Crash rate | `92.00%` | `94.50%` |
| Return gain over own no-request baseline | `1.884` | `0.352` |
| Crash-rate reduction versus own no-request baseline | `7.0` percentage points | `-1.5` percentage points |

Experiment 1 currently performs better in these complete-system evaluations.
Experiment 2 sends slightly fewer requests but has lower return and a higher
crash rate. The larger World Model dataset has therefore not translated into
better DQN-planner performance in this comparison. Further controlled runs
are needed to identify the cause.
