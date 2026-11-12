# Data Collection Configuration

This document describes the current configuration used to collect cooperative-driving data.

## Environment

The project uses a custom `HighwayEnv` environment with:

- `1` controlled ego vehicle
- `10` neighboring vehicles
- `11` vehicles total
- Stable vehicle IDs assigned from `0` to `10`
- `1` action selected per second
- A maximum episode duration of `40` seconds
- `15` internal simulation updates per second

The collector uses random driving actions for the ego vehicle:

```python
action = env.action_space.sample()
```

The current collector runs for `1000` episodes by default and uses:

```python
CustomHighwayEnv(render_mode="rgb_array")
```

Only the Kinematics observation is currently used. Lidar has been removed from the active configuration.

## Kinematics observation

The environment returns up to `10` vehicle rows for each observation:

- Row `0` is the ego vehicle.
- The remaining rows contain up to `9` nearby vehicles.
- Empty rows have `presence = 0`.
- The observed vehicle IDs can change as vehicles move, but each ID remains stable for the episode.
- One of the `10` neighboring vehicles may therefore be outside the current observation.

Each row contains these nine values:

```text
presence
x
y
vx
vy
intent_acceleration
intent_steering
intent_lane_change
vehicle_id
```

The observation shape is:

```text
(10, 9)
```

Coordinates and velocities are provided in the Kinematics observation format. Intent values describe the vehicle's current acceleration, steering, and lane-change commands.

## Communication messages

At each timestep, the collector randomly selects one visible neighboring-vehicle slot or selects the no-request action. The request actions are:

```text
0–9   visible neighboring-vehicle slots
10    no request
```

The collector never requests a vehicle outside the current observation frame. It calls `getMessage()` only when a visible vehicle is selected.

The message contains:

```text
vehicle_id
position
velocity
action
intent
```

## Collected transition

Each CSV row stores one transition:

```text
episode_id
timestep
observation_t
request_action_t
action_t
message_t
observation_t+1
reward_t+1
terminated
truncated
done
```

The observations and messages are stored as JSON strings inside the CSV file so that their array and dictionary structures are preserved. The current dataset is:

```text
data/randomPolicyData_requestPolicy.csv
```

`request_action_t` describes the randomly selected request. The MLP currently uses the resulting message rather than the request action directly.
