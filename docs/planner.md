# Planner

The planner compares possible ego actions by asking MLPx to imagine their consequences.

## One-step decision

For each of the five discrete ego actions, the planner sends the current state, action, and message bundle to MLPx. MLPx returns a predicted next state and reward.

## Multi-step decision

With a planning horizon, the predicted next state becomes the input to the next imagined step. The planner accumulates discounted rewards and selects the first action from the highest-value sequence.

```text
current state
    -> try each action in MLPx
    -> predicted next state
    -> try the next actions
    -> sum discounted rewards
    -> choose the best first action
```

Because the simplified MLPx has no crash head, the current planner uses predicted reward only. Crash and truncation flags are still retained in the real dataset for evaluation.

The same message bundle is reused at every imagined step. A three-step search
considers all 125 action sequences, using discount 0.99; action agreement in
the corrected evaluator ranks the model's selected continuations in the real
simulator. It does not establish optimality across all real action sequences.

The constructor's crash_reward argument is currently stored but not used in
the score. Changing it alone does not alter the planner's chosen action.
See [MLPx](mlpx.html) for prediction metrics and [Request DQN](dqn_policy.html)
for complete-system crash rates. Good short-rollout prediction does not imply
safe full episodes.

The planner implementation is `src/cooperative_driving/planner/MCP.py`.
