# Data-collection policy

The current driving policy is the rule-based intent-aware policy in `scripts/policy/intent_aware_policy.py`.

## Inputs

The policy uses:

- ego speed and lane;
- visible vehicles within 60 m;
- the fixed left, same-lane, and right spatial slots;
- requested preview messages when available;
- distance, closing speed, and time-to-collision estimates.

## Action rules

1. If a front vehicle creates a caution or collision risk, choose the safer adjacent lane when one exists.
2. If neither adjacent lane is safe, choose `SLOWER`.
3. If the lane is safe and the ego is below its target speed, choose `FASTER`.
4. Otherwise choose `IDLE`.

The five actions are `LANE_LEFT`, `IDLE`, `LANE_RIGHT`, `FASTER`, and `SLOWER`. The ego can start in any of the four lanes, so both lane-change directions are represented in the dataset.
