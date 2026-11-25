"""Train a DQN that learns which neighboring vehicle to ask.

The frozen three-step MLPx planner chooses the ego's driving action after the
DQN selects a message request. The DQN actions are eight spatial vehicle
slots plus ask-nobody.

Run:

    .venv/bin/python -m scripts.training.train_request_dqn --episodes 1000

Oracle teaching (labels from scripts.data_collection.collect_oracle_requests):

- ``--oracle-demos FILE ... --bc-epochs N`` gives a behaviour-cloning warm
  start: cross-entropy between the masked request scores and the oracle choice.
- ``--demo-batch-size K`` adds K oracle records to every DQN update
  (DQfD-style): the normal TD loss plus a large-margin loss that keeps the
  oracle-best options ahead of the others by ``--demo-margin``.
- ``--episodes 0`` with ``--bc-epochs`` saves a behaviour-cloning-only policy.
"""

import argparse
import csv
import hashlib
import json
import os
import random
import warnings
from collections import deque
from itertools import chain
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from scripts.data_collection.show_visible_neighbors import SLOT_ORDER, ordered_visible_ids, visible_neighbors
from src.cooperative_driving.environments.customEnv import CustomEnv
from src.cooperative_driving.models.MLPx.mlpx import MLPxWorldModel
from src.cooperative_driving.planner.MCP import MCPPlanner
from src.cooperative_driving.planner.controller import driving_action as choose_driving
from src.cooperative_driving.planner.safety import update_controller_state
from src.cooperative_driving.environments.intent_scenarios import intent_episode
from src.cooperative_driving.policies import RequestDDQN, RequestDQN
from scripts.utils.experiment_utils import (
    add_experiment_argument,
    create_run_dir,
    experiment_checkpoint_dir,
    experiment_results_dir,
    save_run_metadata,
)


ASK_NOBODY = 8


def request_state(environment, visible, state_encoding='kinematics_v1'):
    """Encode ego and eight spatial neighbor slots into 45 values."""
    ego = environment.vehicle
    features = [1.0, float(ego.lane_index[2]) / 3.0, ego.speed / 30.0, 0.0, 0.0]
    if state_encoding == 'ego_control_v1':
        features[3:] = [ego.target_speed / 30., ego.target_lane_index[2] / 3.]
    elif state_encoding != 'kinematics_v1':
        raise ValueError('Unknown request state encoding')
    for slot in SLOT_ORDER:
        candidate = visible.get(slot)
        if candidate is None:
            features.extend([0.0] * 5)
            continue
        vehicle = candidate["vehicle"]
        relative = vehicle.position - ego.position
        relative_velocity = vehicle.velocity - ego.velocity
        features.extend([
            1.0,
            float(relative[0]) / 60.0,
            float(relative[1]) / 4.0,
            float(relative_velocity[0]) / 30.0,
            float(relative_velocity[1]) / 10.0,
        ])
    return np.asarray(features, dtype=np.float32)


def valid_actions(visible, action_encoding="spatial_v1"):
    if action_encoding == "compressed_v0":
        return list(range(len(ordered_visible_ids(visible)))) + [ASK_NOBODY]
    if action_encoding != "spatial_v1":
        raise ValueError(f"Unknown request action encoding: {action_encoding}")
    return [index for index, slot in enumerate(SLOT_ORDER) if slot in visible] + [ASK_NOBODY]


def resolve_requested_vehicle(visible, action, action_encoding="spatial_v1"):
    """Resolve an action without ever compressing fixed spatial positions."""
    if action == ASK_NOBODY:
        return None
    if action not in valid_actions(visible, action_encoding):
        raise ValueError(f"Request action {action} targets an empty or invalid slot")
    if action_encoding == "compressed_v0":
        occupied = [visible[slot]["vehicle"] for slot in SLOT_ORDER if slot in visible]
        return occupied[action]
    return visible[SLOT_ORDER[action]]["vehicle"]


def requested_message(environment, visible, action, action_encoding="spatial_v1"):
    vehicle = resolve_requested_vehicle(visible, action, action_encoding)
    if vehicle is None:
        return None
    messages = [None] * len(SLOT_ORDER)
    vehicle_id = vehicle.vehicle_id
    slot = next(index for index, key in enumerate(SLOT_ORDER) if key in visible and visible[key]["vehicle"].vehicle_id == vehicle_id)
    messages[slot] = environment.request_neighbor_action(vehicle_id)
    return {"messages_by_slot": messages}


def select_action(model, state, valid, epsilon, rng):
    if rng.random() < epsilon:
        return rng.choice(valid)
    with torch.no_grad():
        values = model(torch.tensor(state).unsqueeze(0)).squeeze(0)
        masked = torch.full_like(values, -torch.inf)
        masked[valid] = values[valid]
        return int(masked.argmax())


def redundant_request_penalty(request_action, informed_action, blind_action,
                              penalty):
    """Penalize a training-time request that changes no planner decision.

    The deployed policy still sees only public state. The private response is
    used here only as a teacher signal during training.
    """
    if request_action == ASK_NOBODY or informed_action != blind_action:
        return 0.0
    return -float(penalty)


class ScenarioReplay:
    """Two FIFO pools with equal sampling mass despite unequal episode lengths."""
    def __init__(self, capacity):
        self.pools = {'ordinary': deque(maxlen=capacity // 2),
                      'intent': deque(maxlen=capacity - capacity // 2)}

    def append(self, transition):
        self.pools[transition[-1]].append(transition)

    def extend(self, transitions):
        for transition in transitions:
            self.append(transition)

    def __len__(self):
        return sum(map(len, self.pools.values()))

    def __iter__(self):
        return chain.from_iterable(self.pools.values())

    def sample(self, count):
        ordinary, intent = self.pools.values()
        k_intent = min(len(intent), count // 2 + int(count % 2 and random.random() < .5))
        k_ordinary = min(len(ordinary), count - k_intent)
        k_intent = count - k_ordinary
        batch = random.sample(list(ordinary), k_ordinary) + random.sample(list(intent), k_intent)
        return [item[:-1] for item in batch]


def td_loss(online, target, states, actions, rewards, next_states, dones, next_mask,
            gamma, td_target="dqn"):
    """Huber TD loss with masked DQN or Double-DQN bootstrap targets."""
    current = online(states).gather(1, actions[:, None]).squeeze(1)
    with torch.no_grad():
        target_values = target(next_states).masked_fill(~next_mask, -torch.inf)
        if td_target == "double-dqn":
            # Select with the online network, evaluate with the target network.
            next_actions = online(next_states).masked_fill(
                ~next_mask, -torch.inf
            ).argmax(dim=1)
            future = target_values.gather(1, next_actions[:, None]).squeeze(1)
        elif td_target == "dqn":
            future = target_values.max(dim=1).values
        else:
            raise ValueError(f"Unknown TD target: {td_target}")
        expected = rewards + gamma * future * (1.0 - dones)
    return F.smooth_l1_loss(current, expected)


ORACLE_DEMO_KEYS = ('state', 'target', 'valid', 'choice', 'reward', 'next_state',
                    'done', 'next_valid', 'source')


def load_oracle_demos(paths, state_encoding):
    """Load and join oracle label files from collect_oracle_requests."""
    parts = []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data['metadata']))
            if metadata.get('request_state_encoding') != state_encoding:
                raise ValueError(f'{path} uses request state encoding '
                                 f"{metadata.get('request_state_encoding')}, not {state_encoding}")
            parts.append({key: data[key] for key in ORACLE_DEMO_KEYS})
    demos = {key: np.concatenate([part[key] for part in parts]) for key in ORACLE_DEMO_KEYS}
    if not len(demos['state']):
        raise ValueError('Oracle demonstration files contain no records')
    demos['best'] = demos['target'] > 0
    return demos


def sample_demo_indices(demos, count, generator=np.random):
    """Sample oracle records with equal mass for ordinary and intent sources."""
    groups = [np.flatnonzero(demos['source'] == source) for source in ('ordinary', 'intent')]
    groups = [group for group in groups if len(group)]
    shares = [count // len(groups) + (index < count % len(groups)) for index in range(len(groups))]
    return np.concatenate([group[generator.randint(0, len(group), share)]
                           for group, share in zip(groups, shares)])


def demo_tensors(demos, indices):
    pick = {key: demos[key][indices] for key in ('state', 'target', 'valid', 'choice', 'reward',
                                                 'next_state', 'done', 'next_valid', 'best')}
    return {
        'state': torch.tensor(pick['state'], dtype=torch.float32),
        'target': torch.tensor(pick['target'], dtype=torch.float32),
        'valid': torch.tensor(pick['valid'], dtype=torch.bool),
        'best': torch.tensor(pick['best'], dtype=torch.bool),
        'choice': torch.tensor(pick['choice'], dtype=torch.long),
        'reward': torch.tensor(pick['reward'], dtype=torch.float32),
        'next_state': torch.tensor(pick['next_state'], dtype=torch.float32),
        'done': torch.tensor(pick['done'], dtype=torch.float32),
        'next_valid': torch.tensor(pick['next_valid'], dtype=torch.bool),
    }


def behaviour_cloning_loss(model, states, targets, valid, ask_weight=1.0):
    """Cross-entropy between masked request scores and the oracle target.

    ``ask_weight`` scales the records where the oracle asks somebody. Ask
    records are rare (about 5-9%), so a weight above 1 stops the network from
    learning to never ask.
    """
    logits = model(states).masked_fill(~valid, -1e9)
    per_record = -(targets * F.log_softmax(logits, dim=1)).sum(dim=1)
    weights = torch.where(targets[:, ASK_NOBODY] < 1.0, float(ask_weight), 1.0)
    return (weights * per_record).sum() / weights.sum()


def balanced_ask_weight(demos):
    """Inverse class frequency: no-ask records per ask record."""
    asks = int(np.sum(demos['target'][:, ASK_NOBODY] < 1.0))
    if asks == 0:
        return 1.0
    return float(len(demos['target']) - asks) / asks


def demo_margin_loss(q_values, best, valid, margin):
    """DQfD large-margin loss: oracle-best options must lead by ``margin``.

    max over valid a of [Q(a) + margin * (a not oracle-best)] minus the best
    Q among oracle-best options. It is zero when an oracle-best option is
    already ahead of every other option by the margin.
    """
    augmented = (q_values + margin * (~best).float()).masked_fill(~valid, -torch.inf)
    expert = q_values.masked_fill(~best, -torch.inf).max(dim=1).values
    return (augmented.max(dim=1).values - expert).mean()


def pretrain_behaviour_cloning(model, demos, epochs, batch_size, lr, ask_weight=1.0):
    """Warm start: fit the request network to oracle choices by cross-entropy."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    steps_per_epoch = max(1, len(demos['state']) // batch_size)
    history = []
    for epoch in range(epochs):
        losses = []
        for _ in range(steps_per_epoch):
            batch = demo_tensors(demos, sample_demo_indices(demos, batch_size))
            loss = behaviour_cloning_loss(model, batch['state'], batch['target'], batch['valid'],
                                          ask_weight)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        history.append(float(np.mean(losses)))
        print(f'BC epoch {epoch + 1}/{epochs} | cross-entropy={history[-1]:.4f}', flush=True)
    return history


def optimize(online, target, optimizer, replay, batch_size, gamma, demos=None,
             demo_batch_size=0, demo_margin=0.05, demo_margin_weight=0.0,
             td_target="dqn"):
    if len(replay) < batch_size:
        return None
    batch = replay.sample(batch_size) if isinstance(replay, ScenarioReplay) else random.sample(replay, batch_size)
    states, actions, rewards, next_states, dones, next_valid = zip(*batch)
    states = torch.tensor(np.asarray(states), dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.long)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    next_states = torch.tensor(np.asarray(next_states), dtype=torch.float32)
    dones = torch.tensor(dones, dtype=torch.float32)
    mask = torch.zeros((len(batch), RequestDQN.ACTION_SIZE), dtype=torch.bool)
    for index, valid in enumerate(next_valid):
        mask[index, list(valid)] = True
    loss = td_loss(
        online, target, states, actions, rewards, next_states, dones, mask,
        gamma, td_target,
    )
    if demos is not None and demo_batch_size > 0:
        # DQfD: oracle transitions get the same TD loss plus a large-margin loss.
        demo = demo_tensors(demos, sample_demo_indices(demos, demo_batch_size))
        loss = loss + td_loss(online, target, demo['state'], demo['choice'], demo['reward'],
                              demo['next_state'], demo['done'], demo['next_valid'], gamma,
                              td_target)
        if demo_margin_weight > 0:
            loss = loss + demo_margin_weight * demo_margin_loss(
                online(demo['state']), demo['best'], demo['valid'], demo_margin)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(online.parameters(), 5.0)
    optimizer.step()
    return float(loss.item())


def training_snapshot(online, target, optimizer, replay, rng, epsilon, completed, returns, config):
    """Capture a completed-episode boundary, including all training RNGs."""
    return {
        "checkpoint_version": 2,
        "model_state_dict": online.state_dict(),
        "target_state_dict": target.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "replay": list(replay),
        "episodes": completed,
        "epsilon": epsilon,
        "returns": list(returns),
        "rng_state": rng.getstate(),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "training_config": config,
        "planning_horizon": config["planning_horizon"],
        "communication_cost": config["communication_cost"],
        "mlpx_checkpoint": config["mlpx_checkpoint"],
        "masked_next_request_targets": True,
        "request_action_encoding": "spatial_v1",
        "request_state_encoding": config.get('request_state_encoding', 'kinematics_v1'),
    }


def restore_training(checkpoint, online, target, optimizer, replay, rng):
    if checkpoint.get("request_action_encoding") != "spatial_v1":
        raise ValueError("Cannot resume compressed-action DQN weights as fixed spatial actions")
    online.load_state_dict(checkpoint["model_state_dict"])
    completed = int(checkpoint.get("episodes", 0))
    if checkpoint.get("checkpoint_version") == 2:
        target.load_state_dict(checkpoint["target_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        replay.extend(checkpoint["replay"])
        rng.setstate(checkpoint["rng_state"])
        random.setstate(checkpoint["python_rng_state"])
        np.random.set_state(checkpoint["numpy_rng_state"])
        torch.set_rng_state(checkpoint["torch_rng_state"])
        return completed, checkpoint["epsilon"], list(checkpoint["returns"])
    warnings.warn(
        "Weights-only checkpoint: continuing learned spatial-action weights and episode count, "
        "but restarting optimizer, target network, replay, and random states. "
        "Epsilon is inferred from completed episodes; future latest.pt saves support full resume.",
        UserWarning, stacklevel=2,
    )
    target.load_state_dict(online.state_dict())
    return completed, max(0.05, 0.995 ** completed), []


def atomic_save(checkpoint, path):
    """Replace a checkpoint only after its new file is fully written."""
    temporary = path.with_name(path.name + ".tmp")
    torch.save(checkpoint, temporary)
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_experiment_argument(parser)
    parser.add_argument("--episodes", type=int, default=1000,
                        help="Number of additional episodes to train in this invocation")
    parser.add_argument("--resume", nargs="?", const="auto", default=None,
                        help="Resume a checkpoint; without a path uses latest.pt, then best.pt")
    parser.add_argument("--save-every", type=int, default=50,
                        help="Save full training state every N completed episodes and at the end")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--planning-horizon", type=int, default=3)
    parser.add_argument("--communication-cost", type=float, default=0.05)
    parser.add_argument(
        "--redundant-request-penalty",
        type=float,
        default=0.0,
        help=("Extra training-only penalty when a requested message does not "
              "change the planner's driving action"),
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--replay-size", type=int, default=50000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument('--td-target', choices=['dqn', 'double-dqn'], default='dqn',
                        help='Bootstrap target rule; double-dqn selects next actions online')
    parser.add_argument('--init-checkpoint', default=None,
                        help='Initialize policy weights without restoring replay/optimizer state')
    parser.add_argument('--mlpx-checkpoint')
    parser.add_argument('--scenario-mix', choices=['ordinary', 'mixed'], default='ordinary')
    parser.add_argument('--ordinary-seed', type=int, default=None)
    parser.add_argument('--challenge-seed', type=int, default=800000)
    parser.add_argument('--request-state-encoding', choices=['kinematics_v1', 'ego_control_v1'], default='kinematics_v1')
    parser.add_argument('--safety-guard', action='store_true')
    parser.add_argument('--lane-change-guard', action='store_true')
    parser.add_argument('--max-target-speed', type=float)
    parser.add_argument('--min-clearance', type=float, default=8.)
    parser.add_argument('--min-ttc', type=float, default=3.)
    parser.add_argument('--comfort-ttc', type=float, default=None,
                        help='Proactively change lanes or slow before the emergency TTC threshold')
    parser.add_argument('--batched-planner', action='store_true')
    parser.add_argument('--oracle-demos', nargs='+', default=None,
                        help='Oracle label files from collect_oracle_requests (enables the options below)')
    parser.add_argument('--bc-epochs', type=int, default=0,
                        help='Behaviour-cloning warm-start epochs on the oracle choices')
    parser.add_argument('--bc-batch-size', type=int, default=256)
    parser.add_argument('--bc-lr', type=float, default=1e-3)
    parser.add_argument('--bc-ask-weight', default='1',
                        help="Cross-entropy weight of oracle 'ask' records: a number, or 'balanced' "
                             "for inverse class frequency")
    parser.add_argument('--demo-batch-size', type=int, default=0,
                        help='Oracle records per DQN update (DQfD TD plus margin loss); 0 disables')
    parser.add_argument('--demo-margin', type=float, default=0.05)
    parser.add_argument('--demo-margin-weight', type=float, default=1.0)
    parser.add_argument('--initial-epsilon', type=float, default=1.0)
    parser.add_argument("--output", default=None)
    parser.add_argument("--log-path", default=None,
                        help="CSV path; defaults to the selected experiment's RequestDQN run folder")
    args = parser.parse_args()
    # A pure behaviour-cloning run may train zero DQN episodes.
    min_episodes = 0 if args.oracle_demos and args.bc_epochs > 0 and not args.resume else 1
    if args.episodes < min_episodes or args.save_every < 1:
        parser.error("episodes and save-every must be positive (episodes may be 0 only for BC-only)")
    if (args.bc_epochs > 0 or args.demo_batch_size > 0) and not args.oracle_demos:
        parser.error('bc-epochs and demo-batch-size require --oracle-demos')
    if args.resume and args.init_checkpoint:
        parser.error('--resume and --init-checkpoint are mutually exclusive')
    if min(args.bc_epochs, args.demo_batch_size) < 0 or args.bc_batch_size < 1 \
            or min(args.demo_margin, args.demo_margin_weight) < 0 or args.bc_lr <= 0 \
            or not 0.05 <= args.initial_epsilon <= 1.0:
        parser.error('oracle options must be nonnegative; initial-epsilon must be in [0.05, 1]')

    request_checkpoint_dir = experiment_checkpoint_dir(args.experiment, "RequestDQN")
    resume_checkpoint = None
    resume_path = None
    if args.resume:
        resume_path = Path(args.resume) if args.resume != "auto" else request_checkpoint_dir / "latest.pt"
        if args.resume == "auto" and not resume_path.exists():
            resume_path = request_checkpoint_dir / "best.pt"
        resume_checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
        if resume_checkpoint.get("request_action_encoding") != "spatial_v1":
            parser.error("Resume requires a spatial_v1 DQN checkpoint; compressed actions require retraining")
        saved_config = resume_checkpoint.get("training_config")
        if saved_config:
            if saved_config["experiment"] != args.experiment:
                parser.error("Resume checkpoint experiment differs from --experiment")
            for key in ("seed", "planning_horizon", "communication_cost", "batch_size", "replay_size", "lr"):
                setattr(args, key, saved_config[key])
            for key in ('scenario_mix', 'ordinary_seed', 'challenge_seed', 'request_state_encoding',
                        'safety_guard', 'lane_change_guard', 'max_target_speed', 'min_clearance', 'min_ttc', 'batched_planner',
                        'oracle_demos', 'demo_batch_size', 'demo_margin', 'demo_margin_weight'):
                if key in saved_config:
                    setattr(args, key, saved_config[key])
            args.td_target = saved_config.get('td_target', 'dqn')
            args.comfort_ttc = saved_config.get('comfort_ttc')
            args.redundant_request_penalty = saved_config.get(
                'redundant_request_penalty', 0.0
            )
            # Older full snapshots have six-field replay records and zeroed ego
            # control slots. Never reinterpret them as the new mixed contract.
            if resume_checkpoint.get('checkpoint_version') == 2:
                for key, default in {'scenario_mix': 'ordinary', 'request_state_encoding': 'kinematics_v1',
                                     'safety_guard': False, 'lane_change_guard': False,
                                     'max_target_speed': None, 'batched_planner': False}.items():
                    setattr(args, key, saved_config.get(key, default))
        else:
            for key in ("planning_horizon", "communication_cost"):
                setattr(args, key, resume_checkpoint.get(key, getattr(args, key)))

    if (args.lane_change_guard or args.max_target_speed is not None) and not args.safety_guard:
        parser.error('lane-change-guard and max-target-speed require safety-guard')
    if args.comfort_ttc is not None and (not args.safety_guard or not args.lane_change_guard):
        parser.error('comfort-ttc requires safety-guard and lane-change-guard')
    if not np.isfinite([args.min_clearance, args.min_ttc, args.communication_cost,
                        args.redundant_request_penalty, args.lr]).all() \
            or min(args.min_clearance, args.min_ttc, args.lr) <= 0 \
            or min(args.communication_cost, args.redundant_request_penalty) < 0:
        parser.error('Guard thresholds/lr must be positive; request costs finite and nonnegative')
    if args.max_target_speed is not None and (not np.isfinite(args.max_target_speed) or args.max_target_speed <= 0):
        parser.error('max-target-speed must be finite and positive')
    if args.comfort_ttc is not None and (not np.isfinite(args.comfort_ttc) or args.comfort_ttc <= args.min_ttc):
        parser.error('comfort-ttc must be finite and greater than min-ttc')
    if args.replay_size < max(2, args.batch_size) or args.batch_size < 1:
        parser.error('replay-size must cover batch-size; batch-size positive')
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    mlpx_checkpoint = Path(args.mlpx_checkpoint) if args.mlpx_checkpoint else experiment_checkpoint_dir(args.experiment, "MLPx") / "best.pt"
    if resume_checkpoint:
        mlpx_checkpoint = Path(resume_checkpoint.get("mlpx_checkpoint", mlpx_checkpoint))
    mlpx_hash = hashlib.sha256(mlpx_checkpoint.read_bytes()).hexdigest()
    if resume_checkpoint and resume_checkpoint.get("training_config", {}).get("mlpx_sha256", mlpx_hash) != mlpx_hash:
        parser.error("Frozen MLPx checkpoint changed since the saved training state")
    world_model = MLPxWorldModel.load(mlpx_checkpoint)
    planner = None
    policy_class = RequestDDQN if args.td_target == "double-dqn" else RequestDQN
    online = policy_class()
    online.request_state_encoding = args.request_state_encoding
    initialized_from = None
    if args.init_checkpoint:
        initial = torch.load(args.init_checkpoint, map_location='cpu', weights_only=False)
        if initial.get('request_action_encoding', 'spatial_v1') != 'spatial_v1':
            parser.error('init-checkpoint must use spatial_v1 request actions')
        initial_state_encoding = initial.get('request_state_encoding', 'kinematics_v1')
        if initial_state_encoding != args.request_state_encoding:
            parser.error('init-checkpoint request state encoding differs from this run')
        online.load_state_dict(initial['model_state_dict'])
        initialized_from = str(args.init_checkpoint)
    demos = load_oracle_demos(args.oracle_demos, args.request_state_encoding) if args.oracle_demos else None
    bc_history = []
    if demos is not None:
        if args.bc_ask_weight == 'balanced':
            args.bc_ask_weight = balanced_ask_weight(demos)
        else:
            try:
                args.bc_ask_weight = float(args.bc_ask_weight)
            except ValueError:
                parser.error("bc-ask-weight must be a number or 'balanced'")
            if not np.isfinite(args.bc_ask_weight) or args.bc_ask_weight <= 0:
                parser.error('bc-ask-weight must be finite and positive')
    if demos is not None and args.bc_epochs > 0 and not resume_checkpoint:
        print(f"Behaviour cloning on {len(demos['state'])} oracle records "
              f"(ask weight {args.bc_ask_weight:.2f})", flush=True)
        bc_history = pretrain_behaviour_cloning(online, demos, args.bc_epochs,
                                                args.bc_batch_size, args.bc_lr,
                                                args.bc_ask_weight)
    target = policy_class(); target.load_state_dict(online.state_dict())
    optimizer = torch.optim.Adam(online.parameters(), lr=args.lr)
    replay = ScenarioReplay(args.replay_size) if args.scenario_mix == 'mixed' else deque(maxlen=args.replay_size)
    rng = random.Random(args.seed)
    start_episode, epsilon, returns = 0, args.initial_epsilon, []
    if resume_checkpoint:
        start_episode, epsilon, returns = restore_training(
            resume_checkpoint, online, target, optimizer, replay, rng
        )
        print(f"Resuming {resume_path}: completed={start_episode} epsilon={epsilon:.4f} "
              f"replay={len(replay)}; training {args.episodes} additional episodes")
    output = Path(args.output) if args.output else request_checkpoint_dir / "best.pt"
    if output.name == "latest.pt":
        parser.error("--output must not be named latest.pt; that name is reserved for resumable state")
    output.parent.mkdir(parents=True, exist_ok=True)
    run_root = experiment_results_dir(args.experiment, "RequestDQN") / "experiments"
    run_dir = create_run_dir(run_root, "request_dqn")
    config = {key: getattr(args, key) for key in (
        "experiment", "seed", "planning_horizon", "communication_cost",
        "redundant_request_penalty", "batch_size", "replay_size", "lr", "td_target"
    )}
    config['init_checkpoint'] = initialized_from
    config.update(mlpx_checkpoint=str(mlpx_checkpoint), mlpx_sha256=mlpx_hash)
    config.update({key: getattr(args, key) for key in ('scenario_mix', 'ordinary_seed', 'challenge_seed',
                  'request_state_encoding', 'safety_guard', 'lane_change_guard', 'max_target_speed',
                  'min_clearance', 'min_ttc', 'comfort_ttc', 'batched_planner')})
    config['replay_sampling'] = 'equal_ordinary_intent' if args.scenario_mix == 'mixed' else 'uniform'
    if args.oracle_demos:
        config.update({key: getattr(args, key) for key in (
            'oracle_demos', 'bc_epochs', 'bc_batch_size', 'bc_lr', 'bc_ask_weight', 'demo_batch_size',
            'demo_margin', 'demo_margin_weight', 'initial_epsilon')})
        config['oracle_demo_records'] = int(len(demos['state']))
        config['bc_history'] = bc_history
    save_run_metadata(run_dir, {**config, "resume_from": str(resume_path) if resume_path else None,
                               "start_episode": start_episode, "additional_episodes": args.episodes,
                               "request_action_encoding": "spatial_v1"})
    log_path = Path(args.log_path) if args.log_path else run_dir / "request_dqn_training.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing training log: {log_path}")
    log_file = log_path.open("x", newline="")
    log = csv.DictWriter(log_file, fieldnames=[
        "episode", "return", "mean_loss", "epsilon", "requests", "crashed", "truncated",
        "scenario", "hidden_mode", "environment_seed", "steps", "guard_overrides",
        "decision_changing_requests", "redundant_requests", "selectivity_penalty"
    ])
    log.writeheader()
    end_episode = start_episode + args.episodes
    environment = None

    def save_progress(completed):
        snapshot = training_snapshot(online, target, optimizer, replay, rng,
                                     epsilon, completed, returns, config)
        atomic_save(snapshot, run_dir / "latest.pt")
        atomic_save(snapshot, output.with_name("latest.pt"))

    save_progress(start_episode)

    try:
        for episode in range(start_episode, end_episode):
            source, mode = 'ordinary', None
            environment_seed = (args.ordinary_seed if args.ordinary_seed is not None else args.seed) + episode
            if args.scenario_mix == 'mixed' and episode % 2 == 0:
                source = 'intent'
                challenge_index = episode // 2
                mode = 'stay' if challenge_index % 2 == 0 else 'merge'
                environment_seed = args.challenge_seed + challenge_index // 2
                environment = intent_episode(environment_seed, mode)
                observation = environment.observation_type.observe()
            else:
                environment = CustomEnv()
                observation, _ = environment.reset(seed=environment_seed)
            planner = MCPPlanner(world_model, action_size=environment.action_space.n,
                                 planning_horizon=args.planning_horizon)
            total = 0.0
            episode_losses = []
            requests = 0
            decision_changing_requests = 0
            redundant_requests = 0
            selectivity_penalty = 0.0
            overrides = 0
            controller_state = {} if args.comfort_ttc is not None else None
            terminated = truncated = False
            for timestep in range(environment.config["duration"]):
                visible = visible_neighbors(environment, 60.0, 10.0)
                state = request_state(environment, visible, args.request_state_encoding)
                action = select_action(online, state, valid_actions(visible), epsilon, rng)
                message = requested_message(environment, visible, action)
                requests += int(action != ASK_NOBODY)
                if getattr(world_model, 'STATE_SIZE', 55) == 57:
                    observation = world_model.observe(environment)
                driving_action, guard = choose_driving(planner, environment, observation, message,
                    visible=visible,
                    safety_guard=args.safety_guard, min_clearance=args.min_clearance, min_ttc=args.min_ttc,
                    lane_change_guard=args.lane_change_guard, max_target_speed=args.max_target_speed,
                    comfort_ttc=args.comfort_ttc, batched=args.batched_planner,
                    controller_state=controller_state, commit_state=False)
                teacher_penalty = 0.0
                if action != ASK_NOBODY and args.redundant_request_penalty > 0:
                    blind_action, _ = choose_driving(
                        planner, environment, observation, None,
                        visible=visible, safety_guard=args.safety_guard,
                        min_clearance=args.min_clearance, min_ttc=args.min_ttc,
                        lane_change_guard=args.lane_change_guard,
                        max_target_speed=args.max_target_speed,
                        comfort_ttc=args.comfort_ttc,
                        batched=args.batched_planner,
                        controller_state=controller_state,
                        commit_state=False,
                    )
                    teacher_penalty = redundant_request_penalty(
                        action, driving_action, blind_action,
                        args.redundant_request_penalty,
                    )
                    if teacher_penalty:
                        redundant_requests += 1
                    else:
                        decision_changing_requests += 1
                    selectivity_penalty += teacher_penalty
                update_controller_state(controller_state, driving_action)
                overrides += int(bool(guard and guard['overridden']))
                next_observation, reward, terminated, truncated, _ = environment.step(driving_action)
                next_visible = visible_neighbors(environment, 60.0, 10.0)
                next_state = request_state(environment, next_visible, args.request_state_encoding)
                transition_reward = (
                    float(reward)
                    - (args.communication_cost if action != ASK_NOBODY else 0.0)
                    + teacher_penalty
                )
                total += transition_reward
                transition = (state, action, transition_reward, next_state,
                              terminated or truncated, valid_actions(next_visible))
                replay.append((*transition, source) if args.scenario_mix == 'mixed' else transition)
                loss = optimize(online, target, optimizer, replay, args.batch_size, 0.99,
                                demos=demos, demo_batch_size=args.demo_batch_size,
                                demo_margin=args.demo_margin,
                                demo_margin_weight=args.demo_margin_weight,
                                td_target=args.td_target)
                if loss is not None:
                    episode_losses.append(loss)
                observation = next_observation
                if terminated or truncated:
                    break
            if episode % 50 == 0:
                target.load_state_dict(online.state_dict())
            epsilon = max(0.05, epsilon * 0.995)
            returns.append(total)
            log.writerow({
                "episode": episode + 1,
                "return": total,
                "mean_loss": float(np.mean(episode_losses)) if episode_losses else "",
                "epsilon": epsilon,
                "requests": requests,
                "crashed": int(terminated),
                "truncated": int(truncated),
                "scenario": source, "hidden_mode": mode, "environment_seed": environment_seed,
                "steps": timestep + 1, "guard_overrides": overrides,
                "decision_changing_requests": decision_changing_requests,
                "redundant_requests": redundant_requests,
                "selectivity_penalty": selectivity_penalty,
            })
            log_file.flush()
            if (episode + 1) % 10 == 0:
                print(f"Episode {episode + 1}/{end_episode} | return={np.mean(returns[-10:]):.3f} "
                      f"| loss={np.mean(episode_losses) if episode_losses else float('nan'):.4f} "
                      f"| epsilon={epsilon:.3f}")
            environment.close()
            environment = None
            if (episode + 1) % args.save_every == 0 or episode + 1 == end_episode:
                save_progress(episode + 1)
    except KeyboardInterrupt:
        print(f"Interrupted. Resume the last completed save with --resume {output.with_name('latest.pt')}")
        return
    finally:
        if environment is not None:
            environment.close()
        log_file.close()

    checkpoint = {"model_state_dict": online.state_dict(), "episodes": end_episode,
                  "planning_horizon": args.planning_horizon, "communication_cost": args.communication_cost,
                  "masked_next_request_targets": True, "mlpx_checkpoint": str(mlpx_checkpoint),
                  "request_action_encoding": "spatial_v1", "td_target": args.td_target,
                  "policy_type": online.POLICY_TYPE,
                  "initialized_from": initialized_from}
    checkpoint.update(training_config=config, request_state_encoding=args.request_state_encoding)
    torch.save(checkpoint, output)
    torch.save(checkpoint, run_dir / "best.pt")
    print(f"Saved request DQN to {output}")
    print(f"Saved training log to {log_path}")


if __name__ == "__main__":
    main()
