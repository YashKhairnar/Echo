# Project Progress Log

## Current snapshot

| Area | Current state |
| --- | --- |
| Environment | Four lanes, one ego, aggressive and defensive traffic, stable vehicle IDs |
| Visibility | Up to 60 m, ego lane plus immediate left/right lanes, eight spatial slots |
| Dataset | Experiment 5: 144,945 paired-intent/control-state transitions; earlier datasets preserved |
| World model | Experiment 5 stage-balanced 118-input/57-output MLPx is frozen for Experiment 6 request learning |
| Planner | Reward-only MLPx planner with a three-step horizon |
| Request policy | DQN6 is the strongest learned request candidate: fewer requests and better same-seed return than DQN5, but below no-request after cost |
| Documentation | HTML site with dedicated system, data, training, rendering, and history pages |
| Intent-sensitive scenarios | Experiment 5 has separate scenario benchmarks and a grouped paired-intent training dataset |

## Active pipeline

```text
CustomEnv
  → visible spatial neighbor slots
  → optional intent requests
  → Request DQN chooses whom to ask
  → MLPx imagines three-step ego action outcomes
  → HighwayEnv executes the selected ego action
  → transition and training metrics are logged
```

Latest inspected trained artifacts:

- MLPx checkpoint: `checkpoints/experiment_5/MLPx/intent_stage_best.pt`
- Request DQN checkpoint: `checkpoints/experiment_6/RequestDQN/best.pt`
- Latest combined dataset: `data/experiment_5/combined_world_model_dataset.jsonl`
- Training logs: `results/experiment_6/RequestDQN/experiments/20261120_150538_request_dqn/`
- Ordinary DQN evaluation: `results/experiment_6/evaluation/selective_ordinary/`
- Paired-intent DQN evaluation: `results/experiment_6/evaluation/selective_intent/`
- Documentation home: `docs/index.html`

## Latest milestone

## Experiment 6 · Smooth proactive driving controller

- Added opt-in `--comfort-ttc 6` behavior without changing historical
  experiment defaults. It acts before the 3-second emergency threshold,
  requires a safe destination lane with at least equal forward clearance,
  and falls back to slowing when no useful lane exists.
- Added a five-step lane-change cooldown to stop left/right ping-pong and a
  three-step acceleration cooldown after braking to stop immediate
  `SLOWER → FASTER` reversals.
- Across the same 100 ordinary roads, completed lane changes increased
  171→240, requests fell 16.88→13.17, and crashes remained 0. Adjusted return
  was nearly unchanged at 35.748 versus 35.801; mean speed changed
  27.527→27.401 m/s.
- Across the same 200 paired-intent trials, crashes remained 9, requests stayed
  1.04 per trial, and return was 2.594 versus 2.597. All 63 tests pass.

## Experiment 6 · Selective request teacher

- Added a training-only counterfactual signal: a requested message receives an
  extra 0.10 penalty when the planner chooses the same move with and without
  that response. The DQN input remains public; hidden intent is never exposed
  at inference time.
- Trained for 1,000 mixed episodes using frozen Experiment 5
  `intent_stage_best.pt`, source-balanced replay, the full guard, and disjoint
  ordinary/challenge seeds. The original Experiment 5 checkpoints remain
  unchanged.
- On 100 fresh ordinary roads, DQN6/none adjusted returns were 35.801/36.566,
  with zero crashes for both. DQN6 sent 16.88 requests per episode—about half
  DQN5's same-seed 33.31. Its return improved from DQN5's 34.943.
- On 100 fresh paired scenes (200 trials), DQN6/none adjusted returns were
  2.597/2.643 and crashes were 9/14. DQN6 avoided five baseline crashes and
  introduced none, while sending 1.04 requests per trial.
- The safety/selectivity direction improved, and DQN6 beat random/all request
  baselines. On the same intent seeds it also improved over DQN5: requests
  1.88→1.04, return 2.563→2.597, and crashes 10→9. Asking nobody still won
  adjusted return, so DQN6 is the best learned request candidate rather than
  the overall policy winner.

## Experiment 5 · Mixed-traffic request DQN training and evaluation

- Trained DQN5 for 1,000 episodes against frozen intent_stage_best.pt:
  500 ordinary roads and 500 short stay/merge episodes from 250 paired scenes.
- Shared the speed cap and full geometry guard between trainer and evaluator.
  Added source-balanced replay so long ordinary episodes do not dominate.
  Versioned the 45-value DQN state to include ego target speed and target lane;
  old DQNs retain their zero-filled ego control slots and existing action maps.
- Training produced 19,967 ordinary steps and 1,473 intent steps. Logs include
  loss, requests, traffic type, hidden mode for audit only, seeds, and overrides.
  Full resume restores both replay pools, RNGs, frozen WM hash, and configuration.
- On 100 fresh ordinary roads, DQN5/none returns were 34.840/36.441, both with
  one crash. DQN5 requested 33.26 vehicles per episode without a safety gain.
- On 100 fresh paired scenes (200 trials), DQN5/none returns were 2.500/2.580
  and crashes 18/20. DQN5 avoided two baseline crashes, but request costs still
  exceeded its reward gain. Its return beat random/all requesting in both sets.
- Checked matched initial scenes across all four policies and clustered paired
  return analysis by scene. First requests: 40 scenes nobody, 47 relevant
  neighbor, 13 rear distractor. Identical hidden variants had identical initial
  requests; the pre-request state does not expose private intent.
- Added training/source plots, evaluation comparisons, renderer usage, and
  candidate limitations to the docs. All 54 tests pass. This run is not promoted
  as an overall winner; prior experiment checkpoints remain intact.

## Experiment 5 · WM training and message sensitivity

- Collected 144,945 transitions: 116,320 paired-intent tree records and 28,625
  ordinary-traffic counterfactuals. Each of five actions has 28,989 records;
  73,710 records have no messages. Grouping prevents hidden-mode/branch leakage.
- Added a versioned 57-value state with ego target speed and target lane;
  MLPx now supports 118-input/57-output checkpoints while preserving old
  116-input/55-output weights. Normal evaluators and renderer encode by version.
- Trained two 50-epoch runs on 115,680 records, holding out 29,265. Initial
  action/message balancing selected epoch 50 (loss 0.005619); a source/stage
  correction selected epoch 38 (loss 0.006208), emphasizing early decisions.
- On fresh paired seeds 420000–420099, WM4 crashed 13/200 with and without
  messages. Initial WM5 had 13 without / 11 with; stage-balanced WM5 had
  12 without / 6 with. The stage candidate changed 32 first actions on request.
- Correct messages modestly beat missing messages and clearly beat swapped
  messages on the stage candidate's held-out root next-state probe. Always
  asking still reduced adjusted return by about 0.017 on the final challenge.
- On 100 ordinary guarded episodes, WM4/initial WM5/stage WM5 had 1/0/1
  crashes and mean rewards 36.537/36.874/36.322. Kept initial WM5 as default
  best.pt and saved the stage candidate as intent_stage_best.pt; prior models
  remain unchanged. DQN5 was subsequently trained and evaluated in the newer
  milestone above.
- Updated MLPx, scenario, dataset, and training pages. All 51 tests pass.

## Experiment 5 · Matched-intent scenario validation

- Added a separate challenge-scene builder; ordinary environment defaults are
  unchanged. Visible kinematics and request states match across stay/merge
  variants, while genuine preview responses expose the private lane target.
- Used stock aggressive/defensive controllers with assigned committed targets
  and MOBIL retargeting disabled, plus adjacent-lane rear blockers. Four lanes,
  60 m visibility, real dynamics, collisions, and rewards remain in use.
- Validated 100 scene pairs on seeds 400000–400099 after a separate 20-pair
  pilot. Sixteen pairs changed the oracle action and justified the 0.05 cost.
  Always asking had negative mean net value versus the best blind oracle.
- A privileged selective-request oracle achieved return 2.714 versus 2.692
  for the best blind oracle, requesting in 16% of scenes. This is an upper
  benchmark, not a learned policy result.
- The message rule avoided 16 aggressive-baseline crashes but lost adjusted
  reward to the stronger blind oracle. Frozen WM4 had 16/200 mode-trial crashes
  with and without messages and changed no first actions on receiving them.
- Added an Intent Scenarios documentation tab with the sampler, controls,
  outcome table, example, and chart. All 46 tests passed at this benchmark-only
  milestone; the Experiment 5 models and production dataset were created later.

## Experiment 4 · Guarded request-policy transfer evaluation

- Integrated optional speed cap and destination-lane checks into the normal
  request evaluator and renderer. They share the guard implementation and
  require `--safety-guard --max-target-speed 30 --lane-change-guard`.
- Compared frozen WM4 plus the selected 1,000-episode DQN3 against no-request,
  random-visible, and all-visible policies on fresh seeds 100220–100319.
  Verified initial-scene hashes match across all four policies.
- All four completed 100/100 full episodes with no crashes. Adjusted returns:
  none 37.003, DQN3 36.254, random 34.994, all 32.295.
- DQN3 requested 16.05 vehicles per episode but gained only 0.054 driving
  reward over none. Its adjusted return lost on 97/100 paired seeds; the mean
  difference was −0.749. This is a transfer result, not DQN4 training.
- Speeds remained 27.70–27.75 m/s across policies, with 207–216 settled lane
  changes per 100 episodes. Requesting showed no safety benefit in this batch.
- Added evaluation progress/movement counters, checkpoint hashes, paired
  comparison statistics, and a chart to Request DQN, Experiment 4. Renderer
  smoke check completed 40/40 steps without crashing. All 43 tests pass.

## Experiment 4 · Destination-lane guard and paired validation

- Added opt-in destination-lane front/rear gap projection and a lock against
  stacked or reversing lane commands while a maneuver is unfinished.
- Preserved the 30 m/s diagnostic command cap and current-lane braking.
  All checks use the existing visible public geometry; WM4 remains frozen.
- On 100 fresh paired seeds (100120–100219), both old and new cap/guard
  controllers had zero crashes. The new guard averaged 27.44 m/s, reward
  36.622, and completed 165 settled changes: 47 left, 118 right.
- On the earlier 100-seed regression set, crashes fell from 6/100 to 0/100;
  mean reward was 36.915 and speed 27.56 m/s, with 174 settled changes.
- Verified paired initial-scene hashes for both sets. The new guard had
  0/200 crashes overall; fresh validation alone does not demonstrate a crash
  reduction because its baseline also had zero crashes.
- Added command, lane-index-crossing, and settled-maneuver counters to
  diagnostic outputs. All 40 tests pass. These are no-message tests; no DQN4
  was trained. Guard additions remain opt-in through the diagnostic script.

## Experiment 4 · Planner crash investigation

- Identified repeated FASTER commands raising ego target speed as high as
  50 m/s. The collection rule's target speed is not an environment cap.
- Demonstrated identical WM inputs can have different next states/rewards
  because ego target speed and target lane are omitted from the observation.
- Replayed 20 pre-crash scenes: 13 model-preferred sequences crashed while
  privileged simulator search found non-crashing alternatives in every scene.
- Tested horizon reduction, a target-speed command cap, the existing guard,
  and the cap/guard combination with frozen WM4. One-step planning alone
  still crashed on all 20 initial diagnostic seeds.
- Validated on 100 fresh paired seeds: cap plus guard reduced no-message
  crashes from 97% to 6%, with mean reward 36.212 and speed 27.31 m/s.
  Remaining collisions involved lateral motion; target-lane checks are absent.
- Added a reusable replay investigation script and optional diagnostic speed
  cap. All 33 tests pass. Defaults and selected WM3/DQN3 weights remain intact.
- Full protocols, ablations, caveats, and artifact paths are in Planner,
  Experiment 4. No DQN4 was trained or evaluated.

## Experiment 4 · All-action collection, training, and evaluation

- Collected 566,710 records from 1,000 episodes and 32,968 physical states;
  each action has exactly 113,342 records, with 164,840 no-message records.
- Included 24,102 counterfactual crash flags. Actual collection rollouts had
  269 crashes; counterfactual records are not independent episodes.
- Trained for 50 epochs using action/message balanced sampling; best held-out
  loss was 0.008617 at epoch 35. WM3 artifacts remain unchanged.
- On the same 110,995 held-out records, reward MAE improved from WM3's
  0.04724 to 0.01681; crash-record reward MAE improved from 0.62133 to 0.14087.
- Three-step rollout agreement improved to 86.62% with messages and 86.29%
  without; five-step agreement worsened to 73.58% and 77.59%.
- Unguarded, no-message WM4 driving crashed on all 20 diagnostic seeds,
  versus 19/20 for WM3. Better one-step prediction has not produced safer
  driving. No DQN4 was trained; WM3/DQN3 remains selected.
- Full metrics and graphs are in MLPx and Dataset Dashboard, Experiment 4.

## Action coverage · Evaluation and collection tooling

- Added one-step MLPx evaluation by action, message presence, and crash
  outcome using the training episode-group holdout. Experiment 3 results are
  in evaluation/mlpx_actions/20261119_234823_actions.
- No-message lane-change reward MAE is 0.034–0.041 versus 0.009 for IDLE;
  collision-record errors are much larger across actions (0.30–0.56). No
  message-conditioned collision records occur in this held-out subset.
- Added an all-action counterfactual collector with fixed spatial slots,
  none/single/all request bundles, mixed rule/random real rollouts, explicit
  boundary-command labels, and episode progress logging.
- Added optional action or action-message balanced sampling on training
  records only. Test distribution and episode-group isolation are retained.
- Verified a 55-record smoke dataset had exactly 11 records per action and
  completed a one-epoch balanced-training smoke test. Temporary training
  artifacts were moved to /tmp; no production dataset or model was replaced.
- Documented proposed next-experiment commands. No large collection or
  production retraining was launched.

## Experiment 3 · Optional braking and boundary guard

- Added a public-geometry filter: mask boundary lane changes, then select
  SLOWER for front clearance below 8 m or positive-closing TTC below 3 s.
  It checks the first action only; target-lane safety remains unimplemented.
- Compared five policies on matched seeds 100000–100019 in diagnostic run
  20261119_234104_driving. WM/no-message crashes fell from 19/20 to 6/20;
  WM/DQN crashes from 17/20 to 7/20. Rule/no-message remained at zero.
- Guarded mean driving rewards were 30.239 and 29.690. Logged 206 and 201
  action overrides respectively and no boundary lane commands.
- Added opt-in --safety-guard to policy evaluation and rendering. Guarded
  evaluator outputs default to a separate request_dqn_guarded folder. No
  checkpoints or baseline planner defaults were changed.
- These are 20-seed diagnostic results; the 200-seed guarded evaluation is
  still a separate validation step. Documented remaining front and lateral
  conflict failures on the planner page.

## Experiment 3 · Paired driving diagnostics

- Added scripts.evaluation.diagnose_driving with matched initial-scene checks,
  full step traces, one-step WM errors, and five-decision crash tails.
- Ran seeds 100000–100019 using WM3 and the selected 1,000-episode DQN.
  Both rule policies completed all 20 episodes without crashes; WM/no-message
  crashed in 19 and WM/DQN in 17.
- Most WM crashes had front-contact geometry. Mean predicted collision-step
  rewards were about 0.85, versus actual rewards about 0.10. Found 27 and 40
  boundary lane-change commands respectively in the WM/no-message and DQN runs.
- Saved run: results/experiment_3/diagnostics/20261119_233504_driving.
  Added results and a concrete seed-100000 trace to the planner page. Contact
  labels are heuristic; no driving-policy or safety-filter changes were made.

## Documentation audit · Selected policy and experiment conclusions

- Updated README and documentation home to the Experiment 3 dataset and
  selected 1,000-episode fixed-action policy. Clarified that latest.pt still
  contains the 2,000-episode resume state.
- Updated interactive and README timelines through the encoding corrections,
  continuation regression, checkpoint selection, and seed-10 recordings.
- Removed stale baseline totals and old script paths from the collection
  guide, linking experiment statistics to the dataset dashboard.
- Clarified comparison conclusions: modest observed overall progress versus
  Experiment 1, greater request usage, and no established repeatable safety
  benefit from learned communication. Historical log entries below describe
  the state at the time of each milestone, not the current selection.

## Same-seed policy videos · Experiments 1–3

- Recorded all three experiments with seed 10, the same current 40-second
  environment setup, three-step planning, 1800×800 frames, and 2 fps playback.
- Experiment 1 crashed at step 15, driving reward 12.769, five requests;
  Experiment 2 at step 8, reward 6.790, one request; Experiment 3 at step 10,
  reward 8.594, six requests.
- Experiment 3 uses the selected 1,000-episode fixed-action checkpoint.
  Historical Experiment 1/2 DQN action semantics remain preserved.
- Updated each experiment's video and episode table on the policy page and
  added a common seed-10 comparison. Seed-7 files remain available separately.
  These single-episode outcomes do not replace aggregate evaluations.

## Experiment 3 · Selected policy video recorded

- Recorded seed 7 using the selected 1,000-episode fixed-action DQN and frozen
  Experiment 3 MLPx with a three-step horizon.
- Completed 40/40 steps without crashing, driving reward 37.058, and ten
  requests. This is one illustrative episode, not an aggregate safety result.
- Added the video, per-episode statistics, and reproduction command to the
  DQN policy page. Verified first and final frames show the scene and intent
  overlay. Video: docs/images/dqn_experiment_3_policy.mp4, 1800×800, 41 frames.

## Experiment 3 · Selected the 1,000-episode fixed-action policy

- Restored run 20261119_225244_request_dqn/best.pt as the active checkpoint
  at checkpoints/experiment_3/RequestDQN/best.pt after verifying its episode
  count and spatial_v1 encoding.
- The 2,000-episode checkpoint remains in its run folder. latest.pt still
  resumes the 2,000-episode training state; inference defaults to the selected
  1,000-episode model.
- Saved evaluation files currently describe the 2,000-episode model; the
  policy page retains both reported evaluation summaries.

## Experiment 3 · Continuation evaluation shows regression

- Evaluated the 2,000-episode checkpoint on the same 200 seeds. Mean return
  was 15.142, driving reward 15.515, requests 7.460, and crash rate 90.5%
  (181 crashes). There were 21 truncation flags.
- The 1,000-episode fixed-action baseline had return 15.585, requests 7.015,
  and 88% crashes. Additional training produced five more crashes and lower
  return in this comparison despite higher late-training return.
- Updated DQN evaluation and comparison tables. The earlier checkpoint remains
  in run 20261119_225244_request_dqn; the active checkpoint is still the
  2,000-episode model. No rollback was performed.

## Experiment 3 · Full 2,000-episode training graph

- Combined the initial fixed-action run and its continuation in the DQN
  training graph, with a dashed marker after episode 1,000.
- Updated graph export to resolve resume ancestry, exclude unrelated action
  encodings, and reject missing or overlapping episode ranges. Ambiguous
  histories can be supplied explicitly with --dqn-logs.

## Experiment 3 · DQN continuation completed at episode 2,000

- Run `20261119_230820_request_dqn` completed episodes 1,001–2,000. Mean
  return was 14.392, mean loss 1.577, and requests averaged 7.782 per episode.
- Last-100 mean return reached 15.678, while loss rose to 1.982. Crash flags
  occurred in 926 continuation episodes (92.6%); 75 had truncation flags,
  including one episode with both.
- Active latest.pt records 2,000 completed episodes, epsilon 0.05, and 16,217
  replay transitions. This save supports full-state resume; the initial
  continuation used the older weights-only checkpoint.
- Updated the Experiment 3 policy training results and continuation graph.
  Saved evaluation still belongs to the 1,000-episode policy; evaluate the
  new checkpoint before assessing whether the additional training helped.

## Request DQN · Resumable training

- Added --resume with cumulative episode numbering and separate timestamped
  continuation logs. --episodes counts additional episodes.
- New latest.pt saves include online/target weights, optimizer, replay, epsilon,
  return history, settings, and random states. Saves occur at completed-episode
  boundaries; the frozen MLPx file is checked against its saved hash.
- Existing fixed-action weights-only checkpoints continue with a warning:
  optimizer/replay/random states restart and epsilon is inferred. Legacy
  compressed-action policies require retraining.
- Verified split/resumed training matches uninterrupted training on a small
  deterministic environment. No additional Experiment 3 training was launched
  as part of this implementation.

## Experiment 3 · Fixed-action DQN evaluation complete

- Summary metadata confirms spatial_v1 request actions. Across 200 episodes,
  greedy DQN return was 15.585, driving reward 15.936, and requests averaged
  7.015 per episode. There were 176 crashes (88%) and 25 truncation flags;
  one episode had both flags.
- Asking nobody returned 14.092 with 177 crashes (88.5%). DQN improves mean
  return by 10.59%, but the one-episode crash difference is inconclusive.
- Experiment 3 now has the highest observed DQN return and lowest DQN crash
  rate across experiments. It uses more requests; multiple training seeds
  are needed to assess repeatability.
- Latest evaluation files replaced the previous run in the default folder.
  Updated the policy evaluation and experiment comparison tables.

## Experiment 3 · Fixed-action DQN retrained

- Completed the new 1,000-episode run at
  `results/experiment_3/RequestDQN/experiments/20261119_225244_request_dqn/`.
  The active checkpoint confirms request_action_encoding: spatial_v1 and
  masked next-request targets; Experiment 3 MLPx remains frozen.
- Mean training return was 13.379; last-100 mean return 14.749. Mean loss was
  0.572 overall and 1.137 over the last 100 episodes. Requests averaged 7.404.
- Crash flags occurred in 949 episodes and truncation flags in 54, with three
  overlapping episodes. These are exploratory training outcomes.
- Updated the policy page and matching graphs, preserving the previous run's
  graph, training statistics, and evaluation. New fixed-action evaluation is
  pending; the earlier 89% evaluation crash rate belongs to the old policy.

## Request DQN · Fixed spatial action mapping

- Corrected request actions to use stable spatial indices 0–7 and ask-nobody
  at 8. Empty slots are excluded during exploration, greedy selection, and
  bootstrap target calculation.
- Updated message resolution, random-request evaluation, and renderer vehicle
  highlighting to use the same indices.
- New DQN checkpoints record request_action_encoding: spatial_v1. Earlier
  checkpoints retain compressed_v0 behavior with a warning so their learned
  outputs are not silently reinterpreted.
- This correction follows Experiment 3 and requires DQN retraining. Reuse the
  corrected Experiment 3 MLPx checkpoint; no new WM data is required.

## Experiment 3 · Request DQN evaluation complete

- Evaluated 200 episodes per policy starting at seed 100000. DQN achieved
  mean return 14.869, driving reward 15.199, 6.605 requests per episode,
  178 crashes (89%), and 22 truncations.
- Asking nobody achieved return 14.092 with 177 crashes (88.5%). Random
  requests achieved 9.216 with 99.5% crashes; ask-all achieved 6.047 with
  100% crashes.
- Experiment 3 has the lowest observed DQN crash rate across the three runs,
  but no safety benefit over its own no-request baseline. Updated the policy
  results and comparison table; Experiment 3 recording remains pending.

## Experiment 3 · Request DQN training complete

- Completed 1,000 episodes with the frozen Experiment 3 MLPx checkpoint,
  three-step planning, request cost 0.05, and masked next-request targets.
- Mean training return was 13.475, increasing to 14.366 over the last 100
  episodes. Mean loss was 0.563 overall and 1.136 over the last 100.
- Averaged 7.310 requests per episode. Crash flags occurred in 945 episodes
  and truncation flags in 60; five episodes had both flags.
- Added matching training graphs and full statistics to the DQN policy page.
  Fixed-seed request-policy evaluation and video recording remain pending.

## Experiment 3 · MLPx training and rollout evaluation complete

- Completed 50 MLPx training epochs with spatial_v1 message encoding; best
  checkpoint was epoch 25, with test loss 0.004056.
- Evaluated 100 episodes with up to five states per episode, sampled every
  ten steps. Every horizon/message group contains 299 states, including
  early-terminal candidate outcomes.
- Three-step no-message/all-message state MAE was 0.0323/0.0341, cumulative
  reward MAE 0.0752/0.0819, and action agreement 81.61%/81.94%.
- Five-step errors are higher. Message modes perform similarly at three
  steps; full-episode policy evaluation is needed to assess crash reduction.
- Results use the corrected sequence evaluator and are not directly
  comparable to Experiment 1/2's filtered repeated-action tests.

## Experiment 3 · Corrected dataset collected and merged

- Collected 237,553 intent-aware records from 1,000 episodes, including
  40,000 no-message scenarios. Message slots now preserve absent neighbors.
- Reused the Experiment 2 random baseline's 25,822 records from 1,000 episodes;
  merged output contains 263,375 records with mean reward 0.873, 526 crash
  flags, and 6,684 truncation flags.
- Verified eight-slot structure and requested-vehicle membership for every
  message-bearing record in the informed and combined files.
- Added Experiment 3 dataset statistics to the dashboard and training docs.
  The dataset correction requires retraining; previous metrics remain
  historical results from the earlier encoding.

## 2026-11-19 — Experiment 2 training, evaluation, and intent overlays

- Collected 197,553 intent-aware and 25,822 random no-message transitions
  across 1,000 episodes per source; merged them into 223,375 records.
- Intent-aware rollouts completed 40,000 steps without crashes, using
  40-second episodes at one decision per second. The random source contains
  526 crash transitions. Collection duration describes this saved run;
  future runs use the environment configuration at execution time.
- Trained MLPx for 50 epochs; best test loss was 0.004263 at epoch 42.
  Three-step no-message evaluation achieved state MAE 0.0213, cumulative
  reward MAE 0.0422, and 96.55% best-action agreement on 29 valid states.
  All-message agreement was 68.97%; five-step tests had only 11 valid states.
- Trained the Experiment 2 request DQN for 1,000 episodes, using its frozen
  MLPx checkpoint and a three-step planner. Mean training return was 10.263;
  984 episodes had the crash flag.
- Evaluated both experiments on 200 episodes per request policy, starting
  at seed 100000. Experiment 1 DQN achieved return 14.955 and 92% crashes,
  versus 13.070 and 99% for asking nobody. Experiment 2 DQN achieved return
  12.326 and 94.5% crashes, versus 11.973 and 93% for asking nobody.
- Experiment 1 currently has better complete-system performance. Short-rollout
  prediction metrics do not establish safety over full episodes, especially
  because incomplete candidate rollouts are excluded by the WM evaluator.
- Added matching MLPx and DQN training graphs for Experiment 2, yellow
  experiment headings, and dedicated experiment sections in the docs.
- Recorded Experiment 2 seed 7: crashed at 14/40 steps, driving reward 13.039,
  and five requests. Its video includes the requested vehicle ID, yellow
  circle, exact intent response, and ego action. The overlay also appears in
  future interactive renders and recordings; no-request frames show None.
- Next investigation: compare imagined and real outcomes near collisions,
  including early-terminal cases, to explain the high full-episode crash rate.

## 2026-11-19 — Request DQN, rendering, and documentation polish

- Added a Request DQN that chooses among eight ordered spatial request slots or
  asks nobody. The frozen three-step MLPx planner then chooses the ego's
  driving action.
- Added per-episode DQN logging for return, mean loss, epsilon, request count,
  crashes, and truncations under `results/experiment_1/RequestDQN/experiments/`.
- Added open-loop WM evaluation at three and five steps. Three-step prediction
  is more accurate and substantially cheaper than five-step planning, so the
  current planner uses a three-step horizon.
- Added a larger policy renderer with configurable `--width`, `--height`, and
  `--scaling` options. The renderer shows the ego in green, aggressive cars in
  orange, defensive cars in blue, IDM cars in yellow, and crashed vehicles in
  red. Requested vehicles are circled in yellow.
- Added MLPx and Request DQN training graphs to the HTML Training page.
- Separated the documentation into dedicated MLPx, Planner, Data-collection
  Policy, Request DQN, Rendering, Training, dashboard, and timeline pages.
- Reorganized scripts into `data_collection`, `training`, `evaluation`,
  `policy`, `docs`, and `utils` folders, and updated all current commands and
  imports.
- Improved Phase 3 of the project timeline to describe the complete informed
  world-model pipeline from data creation through DQN training and policy
  inspection.

## 2026-11-19 — HTML documentation site

- Added a static HTML documentation site with a shared sidebar, topic pages,
  archive pages, the intent-aware collection guide, dataset dashboard, and
  project timeline.
- Added `scripts/docs/build_docs_site.py` so the Markdown source pages can be
  regenerated into the shared HTML layout.

## 2026-11-19 — Project cleanup

- Kept only the active four-lane environment, custom vehicles, MLPx model,
  reward-only planner, intent-aware data collectors, dataset adapter, and
  documentation tooling.
- Removed generated Python caches, obsolete CSV/message artifacts, old code
  snapshots, legacy DQN checkpoints, and unused request-policy/communication
  source packages.
- Kept the active MLPx checkpoint at `checkpoints/experiment_1/MLPx/` and preserved older
  experiment results under `results/archive/`.

## 2026-11-19 — Intent-aware vehicle scenarios and dataset collection

- Rebuilt the active environment around standard HighwayEnv behavior classes:
  one controlled ego vehicle plus configurable aggressive and defensive
  background vehicles.
- Added stable vehicle IDs and lookup tables for retrieving each vehicle and
  its behavior type.
- Limited intent-aware visibility to 60 m, the ego lane, and the immediate
  left/right lanes. The selector keeps at most eight vehicles.
- Added `get_vehicle_state()` for current state inspection and
  `preview_next_action()` for calculating a neighbor's next action on a copied
  road without modifying the real environment.
- Added smoke tests for messages, visible neighbors, and current state versus
  previewed action.
- Added a rule-based intent-aware ego policy supporting all five discrete
  actions: left, idle, right, faster, and slower.
- Added request-subset enumeration. For `n` visible vehicles it evaluates all
  `2^n - 1` non-empty request subsets, plus an optional no-message scenario.
- Added counterfactual evaluation on copied environments and JSONL collection
  of state, requested messages, ego action, next state, reward, termination,
  and truncation.
- The initial 100-episode collection produced 4,824 scenario records and 400
  counterfactual terminated scenarios (8.29%). This is a scenario crash rate,
  not the actual full-message policy crash rate.
- Updated the collector to report counterfactual scenario crashes separately
  from actual rollout crashes.
- Replaced the fixed goal-lane behavior with natural lane changes: the ego
  accelerates until a slower vehicle enters the caution range, then selects a
  safe left/right lane or slows if neither is safe.
- Staggered background lead vehicles by lane so the safe escape direction can
  vary and the dataset includes natural left-lane and right-lane examples.
- Randomized the lane-to-lead-offset assignment on every reset after the first
  dataset remained biased toward `LANE_LEFT`.
- Added `collect_random_no_message_dataset.py`, which requests no vehicles and
  samples random ego actions to collect uninformed safe and crash transitions
  for World Model training.
- Merged the intent-aware and random no-message datasets into
  `data/experiment_1/combined_world_model_dataset.jsonl` with source labels, source episode
  IDs, global episode IDs, and unique record IDs: 26,597 records total.

## Earlier chronological history

## 2026-11-05

- Configured the custom HighwayEnv with one ego vehicle and ten neighboring vehicles.
- Added stable vehicle IDs for tracking vehicles across observations.
- Added vehicle intent fields for acceleration, steering, and lane changes.
- Simplified the observation to Kinematics only with shape `(10, 9)`.
- Implemented vehicle communication messages containing vehicle state, action, and intent.
- Completed the random-policy data collector and CSV transition format.
- Added JSON preprocessing through `WorldModelDataset`.
- Started the MLP world model and training pipeline.
- Added checkpoint saving, best/latest checkpoints, and resume support.
- Updated data collection to append safely without duplicate headers or episode IDs.

## 2026-11-12

- Added the MLP world-model checkpoint and training-metric graphs under `results/MLP/`.
- Added the five-step MCP planner for selecting driving actions from MLP predictions.
- Changed the DQN into a request policy with visible-neighbor action masking.
- Added a no-request action and communication cost to the DQN reward.
- Updated data collection to store request actions, visibility-consistent messages, and episode-end flags.
- Added DQN training logs and reward, loss, request, and communication-cost graphs under `results/DQN/`.
- Added a renderer that displays the requested vehicle and its intent.
- Added automatic experiment folders with configurations, notes, metrics, graphs, and checkpoint copies.
- Added dynamic request masking so the DQN can request only visible neighboring vehicles.
- Added resumable DQN training with replay-buffer restoration.
- Added rendered policy video recording, a Play/Pause button, and a visual highlight for the requested vehicle.
- Resumed DQN training from 1000 to 2000 episodes.
- Observed average reward improvement from `6.42` to `8.67` and reduced communication cost.
- Recorded continued Q-loss instability and identified no-request baseline evaluation as the next step.

## 2026-11-12 — Multi-request DQNx

- Added multi-request data collection with up to ten visible request slots.
- Added the MLPx world model for observation, reward, and multi-request message inputs.
- Added DQNx, which predicts request/no-request values independently for each visible slot.
- Kept driving control separate: MLPx plus the five-step MCP planner selects the driving action.
- Added automatic DQNx/MLPx experiment folders, checkpoints, metrics, graphs, and resume support.
- Added DQNx rendering with requested-vehicle highlighting, IDs, and intent overlays.
- Documented that the current `0.1` cost per requested vehicle can encourage the learned policy to request nobody; evaluation should include a no-request baseline.
- Separated public kinematics from private messages: observations now contain position/velocity fields and IDs, while messages contain intent, target lane, and capped time-to-collision.
- Updated MLPx input size from `205` to `115` to match the new four-value message format.

## 2026-11-13 — TTC MLPx/DQNx evaluation

- Trained MLPx on the TTC dataset for 50 epochs; best test loss was `0.1693`
  at epoch 23.
- Trained DQNx for 1,000 episodes with communication cost `0.1`.
- Reward improved during training, but the final policy requested no vehicles
  in a ten-seed evaluation.
- Compared no-request, all-visible-request, and trained-DQNx policies using
  the same seeds. The no-request and trained-DQNx policies both achieved mean
  total reward `5.54`; all-visible requests achieved `-0.53`.
- The current experiment therefore reduces communication but does not yet show
  a measurable communication benefit. The next candidate is a `0.05` cost
  experiment with separate driving-reward logging.

## 2026-11-14 — MLPx rollout evaluation

- Compared MLPx predictions with real-environment rollouts from the same state
  and candidate driving action.
- Five-step planning achieved `23.1%` action-ranking agreement.
- Three-step planning achieved `66.7%` agreement without messages and `42.9%`
  with all messages.
- Set the active MCP planner horizon to three steps because shorter rollouts
  reduce accumulated world-model error.
- Re-evaluated the current MLPx checkpoint with three-step rollouts: action
  ranking agreement was `70.6%` without messages and `45.5%` with all messages.
- Expanded the random multi-request WM dataset to 5,000 episodes; MLPx has
  not yet been retrained on that expanded dataset.

## 2026-11-15 — Cooperative merge scenario with hidden driver types

- Diagnosed why requests never helped: messages (current acceleration, TTC)
  were largely inferable from kinematics, and the highway task rarely made
  the best action depend on them.
- Added `CooperativeMergeEnv`: the ego lane is blocked by a stalled car, so the
  ego must merge into a dense target-lane platoon.
- Added `DriverTypeIDMVehicle` with a private `yielding` / `aggressive` type
  that only shows when someone merges in front of it, and a cruise-controlled
  ego (`CruiseControlledVehicle`) so crashes come from merge decisions, not
  rear-ending.
- Messages now carry driver type through a single `encode_message()` helper
  (slot 1 replaces intent steering; MLPx input size unchanged).
- Collectors take `--env merge|highway`, write merge data to new files, and
  use merge-biased exploration. The matched collector adds a `rear` branch.
- Added `scripts/check_information_gap.py`. On 300 seeds, driver type is
  unpredictable from observation before a merge (51.4% vs 51.4% chance).
  Asking the rear car gave return 7.35 vs 4.64 for the best no-information
  rule, with 0% crashes and merging 4 s sooner.
- Next: collect merge data, retrain MLPx, and repeat the check with the MLPx
  planner instead of hand-written rules.

## 2026-11-15 — MLPx crash head

- Trained the first merge MLPx (2,000 episodes). It reacted to driver-type
  messages only weakly; the MLPx planner crashed about 50% of the time with or
  without messages (`scripts/evaluate_merge_planner.py`).
- Added a crash head to MLPx (BCE on `terminated`), trained the state and
  reward heads on non-crash steps only, and switched the reward target to the
  driving reward. Crash AUC on test: 0.96.
- The MCP planner now weighs each imagined step by its crash probability, and
  `--crash-value` sets how bad an imagined crash is.
- With crash value −6, asking the rear car cut crashes from 21% to 8%
  (return 4.23 → 4.86), still well below the hand-written ask-rear rule
  (7.35, 0% crashes).
- Diagnosis: MLPx predicts the right average merge risk (~45%) and raises it
  for "aggressive" (65%), but does not lower it for "yielding" (44% vs 0%
  real). Next: collect more merge data so "yielding" answers are better
  represented.
- Collected 8,000 more merge episodes (10,000 total) and retrained: test loss
  0.32, crash AUC 0.99. The MLPx planner improved to returns of about 7.5–8.0,
  but messages now barely change its predictions.
- Confirmed that the data contains the signal (crash within 3 steps of
  merging: aggressive 76%, not asked 51%, yielding 31%), so the flat MLPx input
  is the bottleneck: it must match each car's observation row to its message
  slot by itself. Next: a per-vehicle encoder that pairs each car with its
  own message.
- Replaced the flat MLPx with a per-vehicle encoder that pairs each car's
  kinematics with its own message (`FlatMLPxWorldModel` kept for old
  checkpoints). Test loss 0.286, crash AUC 0.998.
- First clear message benefit with the learned planner: asking the rear car
  raised return from 5.21 to 6.53 and cut crashes from 38% to 26% (100 seeds).
  Still below the hand-written ask-rear rule (7.35, 0% crashes), and the
  no-message baseline is worse than with the flat model.
- Remaining issue: crashes happen 1–2 steps after the merge action, so the
  planner predicts them through imagined intermediate states, which blurs the
  message's effect. Candidate fix: a "crash within k steps" head.

## 2026-11-15 — Cleanup and first request DQN in the merge scenario

- Archived everything from the highway experiments: data, checkpoints,
  results, and scripts in `archive/highway_ttc_multirequest/`, write-ups in
  `docs/archive/`. Earlier merge MLPx checkpoints moved to
  `checkpoints/MLPx_merge_history/`.
- Made the code merge-only: shared request helpers in `request_utils.py`,
  removed the legacy single-request dataset, highway env option, and
  multi-request DQNx. Renamed scripts to `train_DQN.py`, `render_DQN.py`,
  and `evaluate_merge_policies.py`.
- New request DQN: at most one car per step (11 actions), per-vehicle
  Q-network, Double DQN. Trained 2,000 episodes at cost 0.05.
- On 200 unseen seeds the DQN (total 6.27) matched "ask nobody" (6.23) and
  random at the same budget (6.29). It asked 2.2 times per episode, with 29% of
  requests to the rear car (random: 14%). The hand-written rear rule was best
  (6.89), but only 0.66 above "none": the planner still gets little value
  from messages, so the request policy has little to learn.

## 2026-11-15 — MLPx v5 (crash-within-3-steps head) and second DQN run

- Added a crash-soon head to MLPx: probability that the episode crashes
  within the next 3 steps, from the real observation and messages. The
  planner uses it as the first step's risk. Crash-soon AUC 0.91.
- With this planner, asking the rear car is worth +2.68 over asking nobody
  (8.47 vs 5.79, 0% crashes), up from +0.66 with MLPx v4 and close to the
  +2.7 in the true environment.
- Retrained the request DQN (2,000 episodes, cost 0.05). It collapsed to
  almost never asking (0.3 requests per episode) and scored like "none"
  (5.78). The planner side now works; the request policy's exploration and
  credit assignment are the next problem.

## 2026-11-15 — Request DQN v3: target-lane exploration + 3-step returns

- Exploration now asks target-lane cars most of the time when it asks, and
  the DQN learns from 3-step returns so delayed merge rewards reach the
  request. Trained 3,000 episodes at cost 0.05.
- On 200 unseen seeds: DQN 6.77 vs "none" 5.79 (+0.98), crashes 2% vs 18%,
  57% of requests to the rear car. First learned policy that beats asking
  nobody. Still below the hand-written rear rule (8.47): it asks twice as
  often and merges later. Training reward was still rising.

## 2026-11-15 — Code simplified for readability

- Shorter, plainer code with a short "what this file does" header in every
  main file. Behaviour is unchanged: the environment, messages, MLPx
  predictions, planner values, rule and DQN results, and the dataset give
  exactly the same numbers as before.
- Removed things no longer used: the old flat MLPx and its loading logic,
  the unused highway base environment and intent vehicles, MLPx `--resume`,
  `--architecture` and the planner's `use_crash_soon` switch (always on).
  `build_planner` now lives in `planner/MCP.py`.
- The code before this change is kept in
  `archive/code_snapshot_before_simplify_2026-11-15/` (needed to load the
  older flat MLPx checkpoints).

## 2026-11-15 — Docs simplified

- README cut to the essentials: the idea, the three parts, current results,
  commands, and links. The old highway experiments moved to
  `docs/archive/earlier_experiments.md`.
- `docs/merge_scenario.md` rewritten as: how it works now, current results, a
  one-line-per-version history table, where things are, next steps. The
  original step-by-step write-up is `docs/archive/merge_scenario_full_history.md`.

## 2026-11-16 — Request DQN run 4: 6,000 episodes

- Same settings as run 3, twice as long. After exploration ended the
  100-episode training score swung between about 5.9 and 7.1 instead of
  rising; best checkpoint at episode 3,637.
- On 200 unseen seeds: 6.28 (run 3: 6.77), crashes 13% (run 3: 2%),
  17 questions per episode. Longer training does not help; run 3 stays the
  current DQN (`checkpoints/DQN/best.pt`), run 4 is in
  `checkpoints/DQN_history/v4_6000/`.
- Next: measure seed-to-seed variance and make DQN training steadier before
  the cost sweep.

## 2026-11-16 — DQN seed check and a steadier variant

- `train_DQN.py` gained `--learning-rate`, `--target-update-steps` and
  `--checkpoint-dir` (defaults unchanged), so several runs can train side by
  side without overwriting `checkpoints/DQN/best.pt`.
- Run 3's settings with training seeds 1 and 2: 6.87 (crashes 14%) and 5.96
  (crashes 16%, almost never asks). With run 3 (6.77): mean 6.53 over three
  seeds, +0.74 over asking nobody, but a spread of about 0.9 between runs.
- Steadier settings (learning rate 5e-5, target copy every 2,000 steps,
  seed 0): 6.11, crashes 10%. Training scores were smoother, but the result
  was worse than run 3 with the same seed.
- Results in `results/merge_policies/cost0.05_dqn_seed_check.json`;
  checkpoints in `checkpoints/DQN_seeds/`. Run 3 stays the current DQN.
- Next: choose checkpoints on fixed validation roads instead of the noisy
  training average, and use several seeds per setting in the cost sweep.

## 2026-11-16 — DQN checkpoint selection on validation roads

- `train_DQN.py` now keeps the model that scores best on 100 fixed
  validation roads (seeds 200,000+, separate from training and test roads),
  checked every 250 episodes once epsilon is at most 0.2 and at the end.
  Scores go to `dqn_validation.csv` in the run folder. Also fixed a plotting
  crash for runs shorter than 50 episodes.
- Retrained training seeds 0, 1 and 2 (training itself is identical to the
  seed check; only the selection differs). Test scores: 6.92 / 7.45 / 6.90,
  mean 7.09 (old selection: 6.77 / 6.87 / 5.96, mean 6.53). Crashes 1–4%.
  Every run now beats asking nobody, by +1.30 on average.
- Still asks a lot (18 questions per episode vs 6.7 for the rear-car rule)
  and stays 1.4 below that rule.
- Seed 0 (best validation score, 7.98) is now `checkpoints/DQN/best.pt`.
  Results in `results/merge_policies/cost0.05_dqn_val_selection.json`.

## 2026-11-16 — Where the DQN asks

- Added `scripts/check_dqn_requests.py`. It labels each step as decisive
  (the planner would act differently with the rear car's answer), not
  decisive, no rear car, or after merge, and records whom the DQN asked.
- Over the three validation-selected DQNs on the 200 test roads:
  - decisive steps are 21–25% of all steps; there it asks the rear car only
    36–63% of the time and the front car 27–51% (the front car's type
    doesn't matter);
  - it asks as much at non-decisive steps as at decisive ones;
  - 41–59% of all its questions come after the merge, when they are useless.
- Likely cause: a 0.05 question cost is tiny next to the ±4 episode noise,
  and the rear and front cars look alike. Next: the cost sweep, then the
  front/rear confusion.

## 2026-11-17 — Cost sweep

- Trained the request DQN at question costs 0, 0.1, 0.2 and 0.5 (3 training
  seeds each, validation selection) and evaluated them with the 0.05 runs on
  the 200 test roads (`scripts/evaluate_cost_sweep.py`, plot in
  `docs/images/cost_sweep.png`). The 12 runs took about 11.5 hours, 6 at a time.
- DQN totals (mean of seeds): 7.60 / 7.09 / 6.18 / 5.70 / 5.79 at costs
  0 / 0.05 / 0.1 / 0.2 / 0.5. Ask nobody: 5.79. Rear-car rule: 8.80 / 8.47 /
  8.13 / 7.46 / 5.44.
- All or nothing: each DQN asks ~15–20 times per episode or never. From cost
  0.2 every run stops asking, although asking the rear car pays up to a cost
  of about 0.45. Only at 0.5 is never asking the right choice.
- Even at cost 0 the DQN (7.60) stays below the rule (8.80): whom to ask is
  the bigger problem, not the cost.
- Fixed run folders: two runs started in the same second used to collide;
  folders now include the seed and get a number added if needed.

## 2026-11-17 — Value-of-information request rule

- Added `src/cooperative_driving/policies/voi.py`: for each target-lane car,
  the planner imagines both answers (yielding / aggressive) and the rule asks
  the car whose answer is worth most, if that beats the question's cost. No
  training.
- Added `predict_batch` to MLPx and `action_values_batched` to the planner:
  3 batched calls instead of 155 single ones, 17× faster, same best move in
  368 of 368 test plans. The driving planner is unchanged.
- On the 200 test roads (cost 0 / 0.05 / 0.1 / 0.2 / 0.5): 8.53 / 8.27 /
  7.70 / 7.33 / 5.87, with 14.5 / 5.6 / 4.4 / 3.2 / 0.6 questions per
  episode. Beats the DQN and asking nobody at every cost; within about
  0.13–0.43 of the hand-written rear-car rule up to cost 0.2 and above it at
  0.5. Crashes 4–7% (rule: 0%).
- `evaluate_cost_sweep.py` now includes the rule and reuses saved results.

## 2026-11-19 — Simplified MLPx world model

- Replaced the previous world-model implementation with a simple MLP in
  `src/cooperative_driving/models/MLPx/mlpx.py`.
- The model receives 116 values: 55 state values, a five-value one-hot ego
  action, and eight seven-value message slots. It predicts only the 55-value
  next state and the next reward; crash heads are not used.
- Message slots now follow a fixed spatial order: left front/beside/rear,
  same-lane front/rear, and right front/beside/rear. Collectors save this
  ordering as `visible_vehicle_slots`.
- The merged dataset contains 26,597 transitions. The trained checkpoint is
  `checkpoints/experiment_1/MLPx/best.pt`.
- Held-out one-step metrics were next-state RMSE `0.0677` and reward RMSE
  `0.0828`. Open-loop evaluation gave state RMSE `0.0935` at three steps and
  `0.1201` at five steps on the random no-message sequences.
- Older experiment outputs were consolidated under `results/archive/`.
  The active run is `results/experiment_1/MLPx/experiments/20261119_110113_mlpx`.

## 2026-11-17 — Value-of-information rule, v2

- Replayed the 8 crashes of the first version (cost 0.05). In 6, the ego
  merged in front of an aggressive car 11–21 m behind without asking: with
  either answer the planner would have waited, so asking looked worthless,
  but with no message it gambled and merged. The imagined messages matched
  the real ones.
- Fix: measure the value of asking against the move the ego actually makes
  without asking (the planner's no-message choice), judged under both answers.
- On the 200 test roads (cost 0 / 0.05 / 0.1 / 0.2 / 0.5): 9.01 / 8.52 /
  8.35 / 7.46 / 6.78 with 16.4 / 9.5 / 8.0 / 6.5 / 2.7 questions per episode,
  crashes 2–4%. Matches or beats the hand-written rear-car rule at every cost
  (clearly at 0.5: 6.78 vs 5.44) and beats the DQN and asking nobody everywhere.
