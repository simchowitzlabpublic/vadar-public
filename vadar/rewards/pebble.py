"""Online (PEBBLE-style) reward-model updates and the learned-reward hooks of the online loop.

During online RL the env reward is replaced by the learned reward:
    Dense:        r_t = r(s_t)
    Pref-Subgoal: r_t = sum_g w(d_{t+1})[g] r(s_t, e_g) + bonus * sum_g d_{t+1}[g]
with s_t the pre-action observation and d_{t+1} = info['subgoal_dones'] after the step.

Every `online_num_interact` env steps:
  1. new preference pairs are sampled from the store of finished online
     episodes (labelled from the env's success flag, or from subgoal_dones);
  2. they join a FIFO online buffer (cap `online_max_feedback`; the offline
     pairs are never evicted);
  3. the reward model is fine-tuned on offline + online pairs;
  4. every replay-buffer transition's `rewards` is recomputed with the new model.

Ported from OGPO-pbrl-subgoal `ogpo/runners/online_pbrl_runner.py` and the
reward override in `ogpo/runners/pbrl_runner.py::build_pbrl_runner_hooks`.
Dropped: frozen-V scoring of online episodes and the V-vs-reward diagnostic
plot (both only used by reward_src=q_val); online episode "returns" are the
summed env rewards, which only matter for label_mode=success_return.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from vadar.rewards.preference_data import (
    concat_preference_buffers,
    concat_stores,
    empty_episode_store,
    empty_pref_buffer,
    generate_preference_pairs,
    generate_preference_pairs_subgoal,
    num_pairs,
    single_episode_entry,
    truncate_fifo,
)
from vadar.rewards.preference_reward import (
    PREF_BUFFER_CKPT_NAME,
    REWARD_MODEL_CKPT_NAME,
    RewardPredictor,
    load_pickle,
    sample_disagreement_pairs,
    save_pickle_atomic,
    save_reward_model,
    train_reward_model,
)


class LearnedReward:
    """The learned reward that replaces the env reward (shared by the env loop and relabelling).

    `holder` is a mutable dict {'params': ...}; online updates swap holder['params']
    and every later call uses the new model.
    """

    def __init__(self, holder: Dict[str, Any], apply_fn, num_subgoals: int = 0, aggregator=None):
        self.holder = holder
        self.apply_fn = apply_fn
        self.num_subgoals = int(num_subgoals)
        self.aggregator = aggregator
        self.predictor = RewardPredictor(apply_fn, num_subgoals=self.num_subgoals)
        if self.num_subgoals > 0:
            assert aggregator is not None, "Pref-Subgoal needs a SubgoalRewardAggregator"

    @property
    def subgoal_mode(self) -> bool:
        return self.num_subgoals > 0

    def step_reward(self, obs: np.ndarray, info: Dict) -> float:
        """Reward of one env transition from the pre-action obs and the step's info."""
        obs = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        if not self.subgoal_mode:
            return float(self.predictor(self.holder['params'], obs).reshape(-1)[0])
        if 'subgoal_dones' not in info:
            raise KeyError("Pref-Subgoal needs info['subgoal_dones'] on every env step; "
                           "build the training env with subgoal tracking")
        dones = np.asarray(info['subgoal_dones'], dtype=np.float32).reshape(1, -1)
        r_sub = self.predictor(self.holder['params'], obs)                  # (G, 1)
        return float(self.aggregator(r_sub, dones)[0])

    def batch_rewards(self, obs: np.ndarray, subgoal_dones: Optional[np.ndarray] = None) -> np.ndarray:
        """Rewards of many transitions (row count padded to a fixed bucket). Returns (B,)."""
        out = self.predictor.padded(self.holder['params'], obs)
        if not self.subgoal_mode:
            return out.reshape(-1)
        assert subgoal_dones is not None, "Pref-Subgoal relabelling needs the stored subgoal_dones"
        return self.aggregator(out, subgoal_dones)


@dataclass
class OnlinePBRLState:
    """Mutable state shared by the env loop (episode hook) and the periodic reward update."""

    reward: LearnedReward
    tx: Any
    opt_state: Any
    rm_cfg: Dict[str, Any]
    pref_buffer_offline: Dict[str, np.ndarray]
    store_offline: Dict[str, np.ndarray]           # offline episode store the offline pairs index
    np_rng: np.random.Generator
    pref_buffer_online: Dict[str, np.ndarray] = field(default_factory=dict)
    online_episode_store: Dict[str, np.ndarray] = field(default_factory=dict)
    online_pairs_seen: int = 0
    online_updates: int = 0
    last_metrics: Dict[str, float] = field(default_factory=dict)

    @property
    def num_subgoals(self) -> int:
        return self.reward.num_subgoals

    @property
    def t_max(self) -> int:
        # Online episodes pad to the offline store's T_max (set to max(offline, env max length)).
        return int(self.store_offline['obs'].shape[1])

    def payload(self) -> Dict[str, Any]:
        return {
            'pref_buffer_offline': self.pref_buffer_offline,
            'store_offline': self.store_offline,
            'pref_buffer_online': self.pref_buffer_online,
            'online_episode_store': self.online_episode_store,
            'online_pairs_seen': int(self.online_pairs_seen),
            'online_updates': int(self.online_updates),
        }


def check_pref_payload(payload: Optional[Dict], num_subgoals: int) -> Optional[Dict]:
    """A saved pref_buffer.pkl is usable only if it matches the run's mode (subgoal_idx iff Pref-Subgoal)."""
    if payload is None:
        return None
    has_sub = 'subgoal_idx' in payload.get('pref_buffer_offline', {})
    if has_sub != (num_subgoals > 0):
        print(f"[PBRL] {PREF_BUFFER_CKPT_NAME} does not match this baseline "
              f"(subgoal pairs: {has_sub}); online reward updates disabled.")
        return None
    return payload


def append_online_trajectory(state: OnlinePBRLState, traj: List[Dict[str, Any]], is_success: float,
                             env_rewards: Optional[List[float]] = None) -> None:
    """Add a finished online episode (list of transitions) to the online episode store.

    `env_rewards` are the env's own rewards of the episode; they only feed the
    episode return used by label_mode=success_return (the reference used the
    learned rewards here, which is what you get when env_rewards is None).
    """
    if not traj:
        return
    obs = np.stack([np.asarray(t['observations'], dtype=np.float32).reshape(-1) for t in traj])
    if env_rewards is None:
        env_rewards = [t['rewards'] for t in traj]
    env_rewards = np.asarray(env_rewards, dtype=np.float32).reshape(-1)
    assert env_rewards.shape[0] == obs.shape[0]
    dones = None
    if state.num_subgoals > 0:
        if any('subgoal_dones' not in t for t in traj):
            raise KeyError("Pref-Subgoal: online transitions lack 'subgoal_dones'")
        dones = np.stack([np.asarray(t['subgoal_dones'], dtype=np.float32).reshape(-1) for t in traj])
        assert dones.shape == (obs.shape[0], state.num_subgoals)
    if not state.online_episode_store:
        state.online_episode_store = empty_episode_store(state.t_max, obs.shape[1], state.num_subgoals)
    entry = single_episode_entry(obs, env_rewards, bool(is_success), state.t_max, subgoal_dones=dones)
    state.online_episode_store = concat_stores(state.online_episode_store, entry)
    if not state.pref_buffer_online:
        state.pref_buffer_online = empty_pref_buffer(subgoal=state.num_subgoals > 0)


def online_reward_update(state: OnlinePBRLState) -> Optional[Dict[str, float]]:
    """Sample new online pairs, fine-tune the reward model on offline + online pairs. None if skipped."""
    cfg = state.rm_cfg
    store = state.online_episode_store
    feedback_batch = int(cfg['online_feedback_batch'])
    if not store or store['obs'].shape[0] < 2 * feedback_batch:
        return None  # not enough fresh episodes yet

    G = state.num_subgoals
    if cfg['online_sampling'] == 'disagreement':
        new_pairs = sample_disagreement_pairs(
            store, state.reward.holder['params'], state.reward.apply_fn, feedback_batch,
            int(cfg['disagreement_oversample']), state.np_rng, label_mode=cfg['label_mode'],
            num_subgoals=G, subgoal_label_mode=cfg['subgoal_label_mode'])
    elif G > 0:
        new_pairs = generate_preference_pairs_subgoal(store, max(1, feedback_batch // G), state.np_rng,
                                                      subgoal_label_mode=cfg['subgoal_label_mode'])
    else:
        new_pairs = generate_preference_pairs(store, feedback_batch, state.np_rng, all_pairs=False,
                                              label_mode=cfg['label_mode'])
    if num_pairs(new_pairs) == 0:
        print("[PBRL] online update: no non-tie pairs this round; skipping.")
        return None

    state.pref_buffer_online = truncate_fifo(concat_preference_buffers(state.pref_buffer_online, new_pairs),
                                             int(cfg['online_max_feedback']))
    state.online_pairs_seen += num_pairs(new_pairs)

    # Online pairs index the online store; shift them past the offline episodes.
    n_off = int(state.store_offline['lengths'].shape[0])
    online = {k: v for k, v in state.pref_buffer_online.items()}
    online['ep_idx_1'] = online['ep_idx_1'].astype(np.int32) + n_off
    online['ep_idx_2'] = online['ep_idx_2'].astype(np.int32) + n_off
    combined = concat_preference_buffers(state.pref_buffer_offline, online)
    combined_store = concat_stores(state.store_offline, store)

    epochs, batch_size = int(cfg['online_train_epochs']), int(cfg['online_large_batch'])
    params, opt_state, history = train_reward_model(
        state.reward.holder['params'], state.reward.apply_fn, state.tx, state.opt_state, combined,
        combined_store, epochs=epochs, batch_size=batch_size, rng=state.np_rng, num_subgoals=G)
    state.reward.holder['params'] = params
    state.opt_state = opt_state
    state.online_updates += 1

    last = history[-1] if history else {}
    state.last_metrics = {
        'online_loss': float(last.get('train_loss', float('nan'))),
        'online_acc': float(last.get('train_accuracy', float('nan'))),
        'online_pairs_seen': float(state.online_pairs_seen),
        'online_pref_buffer_size': float(num_pairs(state.pref_buffer_offline) + num_pairs(state.pref_buffer_online)),
        'online_updates': float(state.online_updates),
        'online_store_size': float(store['obs'].shape[0]),
        'online_grad_steps': float(epochs * ((num_pairs(combined) + batch_size - 1) // batch_size)),
    }
    return dict(state.last_metrics)


def relabel_replay_buffer(replay_buffer, reward: LearnedReward, chunk: int = 65536) -> int:
    """Overwrite replay_buffer['rewards'][:size] with the current learned reward. Returns the count.

    Pref-Subgoal recomputes each transition's weights from its stored subgoal_dones.
    Only `rewards` changes (stored mc_returns, if any, are not recomputed).
    """
    n = int(getattr(replay_buffer, 'size', 0))
    if n == 0:
        return 0
    obs = replay_buffer._dict['observations']
    rewards = replay_buffer._dict['rewards']
    if rewards.shape[0] != obs.shape[0]:
        raise RuntimeError(f"Unexpected replay buffer layout: obs {obs.shape}, rewards {rewards.shape}")
    dones = None
    if reward.subgoal_mode:
        if 'subgoal_dones' not in replay_buffer._dict:
            raise KeyError("Pref-Subgoal relabelling needs replay_buffer['subgoal_dones']")
        dones = replay_buffer._dict['subgoal_dones']
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        r = reward.batch_rewards(obs[s:e], None if dones is None else np.asarray(dones[s:e], dtype=np.float32))
        rewards[s:e] = np.asarray(r, dtype=rewards.dtype).reshape(-1)[: e - s]
    return n


def make_online_callbacks(state: OnlinePBRLState, save_dirs: List[str], logger=None):
    """(on_episode_end(traj, is_success, env_rewards), on_interval(log_step, replay_buffer) -> metrics)."""

    def on_episode_end(traj, is_success, env_rewards=None):
        append_online_trajectory(state, traj, is_success, env_rewards)

    def on_interval(log_step, replay_buffer):
        metrics = online_reward_update(state)
        # Save on every tick (also without an update) so the rolling checkpoint is fresh.
        for d in save_dirs:
            try:
                save_reward_model(f"{d}/{REWARD_MODEL_CKPT_NAME}", state.reward.holder['params'], state.rm_cfg,
                                  goal_dim=state.num_subgoals)
            except Exception as exc:
                print(f"[PBRL] reward checkpoint save warning: {exc}")
        if metrics is None:
            return {}
        metrics['relabel_count'] = float(relabel_replay_buffer(replay_buffer, state.reward,
                                                               chunk=int(state.rm_cfg['online_relabel_chunk'])))
        for d in save_dirs:
            try:
                save_pickle_atomic(f"{d}/{PREF_BUFFER_CKPT_NAME}", state.payload())
            except Exception as exc:
                print(f"[PBRL] pref buffer save warning: {exc}")
        return metrics

    return on_episode_end, on_interval


def load_pref_payload(path: str) -> Optional[Dict]:
    try:
        return load_pickle(path)
    except Exception as exc:
        print(f"[PBRL] Could not load {path} ({exc}); online reward updates disabled.")
        return None
