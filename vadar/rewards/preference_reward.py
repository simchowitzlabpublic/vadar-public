"""Preference reward model: network, Bradley-Terry training, pretraining, checkpoints.

Shared by both preference baselines:
  - Dense:        r(s), one model, whole-episode success preferences.
  - Pref-Subgoal: r(s, g), one shared model on [s, one_hot(g)], per-subgoal preferences.

Ported from OGPO-pbrl-baseline / OGPO-pbrl-subgoal (`ogpo/networks/reward_net.py`,
`ogpo/agents/modules/pbrl_helper.py`, `ogpo/runners/pbrl_runner.py`). The
V*-based targets (reward_src=q_val), robust V smoothing, dense-reward targets
and rollout plots of the reference are not ported: every reference run used
label_mode=success_only, under which those targets never affect a label.
"""

import os
import pickle
from typing import Any, Callable, Dict, Optional

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tqdm

from vadar.networks.critics import ensemblize
from vadar.rewards.preference_data import (
    build_episode_store,
    concat_preference_buffers,
    discounted_mc_returns,
    generate_preference_pairs,
    generate_preference_pairs_subgoal,
    natural_max_episode_length,
    num_pairs,
    split_episode_store,
)

REWARD_MODEL_CKPT_NAME = 'reward_model.pkl'
PREF_BUFFER_CKPT_NAME = 'pref_buffer.pkl'

# Relabel / diagnostic calls pass chunks of varying length; every distinct
# shape is a new XLA compile (and, on the reference runs, ~30 MB/h of device
# memory outside the preallocated pool). Pad row counts up to a multiple of this.
_PAD_BUCKET = 4096


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

class RewardMLP(nn.Module):
    """n_layers x (Dense, leaky_relu), then Dense(1) and tanh: rewards lie in [-1, 1]."""

    hidden_dim: int = 512
    n_layers: int = 4

    @nn.compact
    def __call__(self, x):
        for _ in range(self.n_layers):
            x = nn.Dense(self.hidden_dim)(x)
            x = nn.leaky_relu(x)
        x = nn.Dense(1)(x)
        return nn.tanh(x)


class RewardEnsemble(nn.Module):
    """Ensemble of RewardMLPs.

    goal_dim=0: r(s) on observations (Dense).
    goal_dim=G: r(s, g) on [observations, one_hot(g)] (Pref-Subgoal); one network
    serves every subgoal.
    Returns (ensemble, batch, 1).
    """

    ensemble_size: int = 3
    hidden_dim: int = 512
    n_layers: int = 4
    goal_dim: int = 0

    def setup(self):
        self.net = ensemblize(RewardMLP, self.ensemble_size)(hidden_dim=self.hidden_dim, n_layers=self.n_layers)

    def __call__(self, obs, goal=None):
        if self.goal_dim > 0:
            assert goal is not None, "RewardEnsemble(goal_dim>0) needs a goal input"
            x = jnp.concatenate([obs, goal], axis=-1)
        else:
            assert goal is None, "RewardEnsemble(goal_dim=0) takes no goal input"
            x = obs
        return self.net(x)


def make_reward_model(rm_cfg: Dict, goal_dim: int) -> RewardEnsemble:
    return RewardEnsemble(ensemble_size=int(rm_cfg['ensemble_size']), hidden_dim=int(rm_cfg['hidden_dim']),
                          n_layers=int(rm_cfg['n_layers']), goal_dim=int(goal_dim))


def init_reward_params(model: RewardEnsemble, example_obs, seed: int):
    key = jax.random.PRNGKey(int(seed) + 61000)
    obs = jnp.asarray(np.asarray(example_obs, dtype=np.float32).reshape(1, -1))
    if model.goal_dim > 0:
        return model.init(key, obs, jnp.zeros((1, model.goal_dim), dtype=jnp.float32))['params']
    return model.init(key, obs)['params']


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def _predict_ensemble_rewards(params, apply_fn, obs, goal=None):
    obs = jnp.asarray(obs)
    if obs.ndim == 1:
        obs = obs[None, :]
    if goal is None:
        return apply_fn({'params': params}, obs).squeeze(-1)  # (E, B)
    goal = jnp.asarray(goal)
    if goal.ndim == 1:
        goal = jnp.broadcast_to(goal[None, :], (obs.shape[0], goal.shape[0]))
    return apply_fn({'params': params}, obs, goal).squeeze(-1)  # (E, B)


def predict_reward(params, apply_fn, obs, goal=None):
    """Ensemble-mean reward, (B,)."""
    return _predict_ensemble_rewards(params, apply_fn, obs, goal=goal).mean(axis=0)


def predict_reward_subgoals(params, apply_fn, obs, num_subgoals: int):
    """Ensemble-mean r(s, e_g) for every subgoal in one forward pass, (G, B)."""
    obs = jnp.asarray(obs)
    if obs.ndim == 1:
        obs = obs[None, :]
    B = obs.shape[0]
    obs_tiled = jnp.tile(obs, (num_subgoals, 1))                              # (G*B, D)
    goals = jnp.repeat(jnp.eye(num_subgoals, dtype=obs.dtype), B, axis=0)    # (G*B, G)
    r = _predict_ensemble_rewards(params, apply_fn, obs_tiled, goal=goals)    # (E, G*B)
    return r.mean(axis=0).reshape(num_subgoals, B)


class RewardPredictor:
    """Jitted reward-model queries that read the *current* params on every call.

    params are a jit argument (not a closure constant), so online fine-tuning
    takes effect immediately without retracing.
    num_subgoals=0: __call__ returns (B,) r(s); num_subgoals=G: (G, B) r(s, e_g).
    """

    def __init__(self, apply_fn: Callable, num_subgoals: int = 0):
        self.apply_fn = apply_fn
        self.num_subgoals = int(num_subgoals)
        if self.num_subgoals > 0:
            self._fn = jax.jit(lambda p, o: predict_reward_subgoals(p, apply_fn, o, self.num_subgoals))
        else:
            self._fn = jax.jit(lambda p, o: predict_reward(p, apply_fn, o))

    def __call__(self, params, obs) -> np.ndarray:
        return np.asarray(self._fn(params, jnp.asarray(np.asarray(obs, dtype=np.float32))))

    def padded(self, params, obs, bucket: int = _PAD_BUCKET) -> np.ndarray:
        """Same as __call__ with the batch padded to a multiple of `bucket` (bounded #shapes)."""
        obs = np.asarray(obs, dtype=np.float32)
        if obs.ndim == 1:
            obs = obs[None, :]
        n = obs.shape[0]
        n_pad = ((n + bucket - 1) // bucket) * bucket
        if n_pad != n:
            obs = np.concatenate([obs, np.zeros((n_pad - n, obs.shape[1]), dtype=np.float32)], axis=0)
        out = self(params, obs)
        return out[..., :n]


# ---------------------------------------------------------------------------
# Bradley-Terry training on episode pairs
# ---------------------------------------------------------------------------

def gather_pair_batch(ep_store, ep_idx_1, ep_idx_2, labels, subgoal_idx=None, num_subgoals: int = 0):
    """Batch of padded episode pairs; always padded to the store's T_max so jit traces once."""
    obs_all, lengths_all = ep_store['obs'], ep_store['lengths']
    t_max = obs_all.shape[1]
    arange = np.arange(t_max, dtype=np.int32)[None, :]
    batch = {
        'seg1_obs': jnp.asarray(obs_all[ep_idx_1]),
        'seg1_mask': jnp.asarray((arange < lengths_all[ep_idx_1][:, None]).astype(np.float32)),
        'seg2_obs': jnp.asarray(obs_all[ep_idx_2]),
        'seg2_mask': jnp.asarray((arange < lengths_all[ep_idx_2][:, None]).astype(np.float32)),
        'labels': jnp.asarray(np.asarray(labels, dtype=np.float32)),
    }
    if subgoal_idx is not None:
        assert num_subgoals > 0
        batch['goal'] = jnp.asarray(np.eye(num_subgoals, dtype=np.float32)[np.asarray(subgoal_idx, dtype=np.int32)])
    return batch


def _trajectory_return_hat(params, apply_fn, seg_obs, seg_mask, goal=None):
    """Predicted episode return per ensemble member, padding masked out. Returns (E, B)."""
    b, t, _ = seg_obs.shape
    flat_obs = seg_obs.reshape((b * t, -1))
    if goal is None:
        r_hat = apply_fn({'params': params}, flat_obs).squeeze(-1)
    else:
        r_hat = apply_fn({'params': params}, flat_obs, jnp.repeat(goal, t, axis=0)).squeeze(-1)
    r_hat = r_hat.reshape((r_hat.shape[0], b, t))
    return (r_hat * seg_mask[None, :, :]).sum(axis=-1)


def preference_loss(params, apply_fn, batch):
    """Bradley-Terry cross-entropy on summed predicted rewards, averaged over ensemble members."""
    goal = batch.get('goal')
    r1 = _trajectory_return_hat(params, apply_fn, batch['seg1_obs'], batch['seg1_mask'], goal=goal)
    r2 = _trajectory_return_hat(params, apply_fn, batch['seg2_obs'], batch['seg2_mask'], goal=goal)
    logits = jnp.stack([r1, r2], axis=-1)                       # (E, B, 2)
    labels = batch['labels'].astype(jnp.float32)
    targets = jnp.stack([1.0 - labels, labels], axis=-1)        # (B, 2)
    loss = optax.softmax_cross_entropy(logits, targets[None]).mean()
    accuracy = (jnp.argmax(logits, axis=-1) == jnp.round(labels).astype(jnp.int32)[None, :]).mean()
    return loss, {'loss': loss, 'accuracy': accuracy}


def predict_pair_disagreement(params, apply_fn, batch):
    """Std over ensemble members of (R_hat(tau_2) - R_hat(tau_1)), (B,)."""
    goal = batch.get('goal')
    r1 = _trajectory_return_hat(params, apply_fn, batch['seg1_obs'], batch['seg1_mask'], goal=goal)
    r2 = _trajectory_return_hat(params, apply_fn, batch['seg2_obs'], batch['seg2_mask'], goal=goal)
    return (r2 - r1).std(axis=0)


def sample_disagreement_pairs(ep_store, params, apply_fn, n_pairs: int, oversample: int,
                              rng: np.random.Generator, label_mode: str = 'success_only',
                              num_subgoals: int = 0, subgoal_label_mode: str = 'success_time'):
    """Oversample candidate pairs and keep the n_pairs with the highest ensemble disagreement."""
    n_candidates = max(n_pairs, n_pairs * max(1, int(oversample)))
    if num_subgoals > 0:
        candidates = generate_preference_pairs_subgoal(ep_store, max(1, n_candidates // num_subgoals), rng,
                                                       subgoal_label_mode=subgoal_label_mode)
    else:
        candidates = generate_preference_pairs(ep_store, n_candidates, rng, label_mode=label_mode)
    n_candidates = num_pairs(candidates)
    if n_candidates <= n_pairs:
        return candidates
    parts = []
    for start in range(0, n_candidates, 256):
        end = min(start + 256, n_candidates)
        batch = gather_pair_batch(ep_store, candidates['ep_idx_1'][start:end], candidates['ep_idx_2'][start:end],
                                  np.zeros(end - start, dtype=np.float32),
                                  subgoal_idx=candidates['subgoal_idx'][start:end] if num_subgoals > 0 else None,
                                  num_subgoals=num_subgoals)
        parts.append(np.asarray(predict_pair_disagreement(params, apply_fn, batch)))
    top = np.argsort(np.concatenate(parts))[-n_pairs:]
    return {k: v[top] for k, v in candidates.items()}


def train_reward_model(params, apply_fn, tx, opt_state, pref_buffer, ep_store, epochs: int, batch_size: int,
                       rng: np.random.Generator, val_pref_buffer=None, val_ep_store=None, num_subgoals: int = 0):
    """Train on a fixed buffer of episode-pair indices. Returns (params, opt_state, per-epoch history)."""
    n = num_pairs(pref_buffer)
    if n == 0:
        return params, opt_state, []
    subgoal_mode = 'subgoal_idx' in pref_buffer
    if subgoal_mode:
        assert num_subgoals > 0, "pref_buffer has subgoal_idx but num_subgoals was not given"

    @jax.jit
    def train_step(p, s, batch):
        (_, metrics), grads = jax.value_and_grad(lambda q: preference_loss(q, apply_fn, batch), has_aux=True)(p)
        updates, s = tx.update(grads, s, p)
        return optax.apply_updates(p, updates), s, metrics

    @jax.jit
    def eval_step(p, batch):
        return preference_loss(p, apply_fn, batch)[1]

    def _batch(buf, store, idx):
        return gather_pair_batch(store, buf['ep_idx_1'][idx], buf['ep_idx_2'][idx], buf['labels'][idx],
                                 subgoal_idx=buf['subgoal_idx'][idx] if 'subgoal_idx' in buf else None,
                                 num_subgoals=num_subgoals)

    n_val = num_pairs(val_pref_buffer)
    val_store = val_ep_store if val_ep_store is not None else ep_store
    history = []
    pbar = tqdm.tqdm(range(epochs), desc='reward model', leave=False)
    for _ in pbar:
        perm = rng.permutation(n)
        losses, accs = [], []
        for start in range(0, n, batch_size):
            params, opt_state, m = train_step(params, opt_state, _batch(pref_buffer, ep_store, perm[start:start + batch_size]))
            losses.append(float(m['loss']))
            accs.append(float(m['accuracy']))
        epoch = {'train_loss': float(np.mean(losses)), 'train_accuracy': float(np.mean(accs))}
        if n_val > 0:
            vl, va = [], []
            for start in range(0, n_val, batch_size):
                m = eval_step(params, _batch(val_pref_buffer, val_store, np.arange(start, min(start + batch_size, n_val))))
                vl.append(float(m['loss']))
                va.append(float(m['accuracy']))
            epoch['val_loss'], epoch['val_accuracy'] = float(np.mean(vl)), float(np.mean(va))
        history.append(epoch)
        pbar.set_postfix({k: f"{v:.3f}" for k, v in epoch.items()})
    return params, opt_state, history


def per_subgoal_accuracy(params, apply_fn, pref_buffer, ep_store, num_subgoals: int, batch_size: int = 128):
    """Bradley-Terry accuracy of r(s, g) on the pairs of each subgoal (nan if none)."""
    accs = []
    sub = np.asarray(pref_buffer['subgoal_idx'], dtype=np.int32)
    for g in range(num_subgoals):
        sel = np.nonzero(sub == g)[0]
        if sel.size == 0:
            accs.append(float('nan'))
            continue
        correct = 0.0
        for start in range(0, sel.size, batch_size):
            idx = sel[start:start + batch_size]
            batch = gather_pair_batch(ep_store, pref_buffer['ep_idx_1'][idx], pref_buffer['ep_idx_2'][idx],
                                      pref_buffer['labels'][idx], subgoal_idx=sub[idx], num_subgoals=num_subgoals)
            correct += float(preference_loss(params, apply_fn, batch)[1]['accuracy']) * idx.size
        accs.append(correct / sel.size)
    return accs


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------

def save_reward_model(path: str, params, rm_cfg: Dict, goal_dim: int) -> None:
    """Atomically pickle params + architecture (goal_dim > 0 marks an r(s, g) model)."""
    payload = {
        'reward_params_state_dict': flax.serialization.to_state_dict(params),
        'ensemble_size': int(rm_cfg['ensemble_size']),
        'hidden_dim': int(rm_cfg['hidden_dim']),
        'n_layers': int(rm_cfg['n_layers']),
        'goal_dim': int(goal_dim),
    }
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f:
        pickle.dump(payload, f)
    os.replace(tmp, path)


def load_reward_model(path: str, example_obs, seed: int, expected_goal_dim: Optional[int] = None):
    """Inverse of save_reward_model. Returns (params, apply_fn, goal_dim)."""
    with open(path, 'rb') as f:
        payload = pickle.load(f)
    goal_dim = int(payload.get('goal_dim', 0))
    if expected_goal_dim is not None and goal_dim != int(expected_goal_dim):
        raise ValueError(f"Reward model {path} has goal_dim={goal_dim}, this run needs {expected_goal_dim} "
                         "(0 = Dense, G = Pref-Subgoal)")
    model = RewardEnsemble(ensemble_size=int(payload['ensemble_size']), hidden_dim=int(payload['hidden_dim']),
                           n_layers=int(payload['n_layers']), goal_dim=goal_dim)
    params = flax.serialization.from_state_dict(init_reward_params(model, example_obs, seed),
                                                payload['reward_params_state_dict'])
    print(f"[PBRL] Loaded reward model from {path} (goal_dim={goal_dim})")
    return params, model.apply, goal_dim


def save_pickle_atomic(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def load_pickle(path: str) -> Any:
    with open(path, 'rb') as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Offline pretraining
# ---------------------------------------------------------------------------

def _log_history(logger, history, start_step):
    step = int(start_step)
    for m in history:
        step += 1
        if logger is not None:
            logger.log(dict(m), 'reward_model', step=step)
    return step


def pretrain_reward_model(arrays: Dict[str, np.ndarray], rm_cfg: Dict, seed: int, discount: float,
                          num_subgoals: int = 0, env_max_episode_length: Optional[int] = None,
                          logger=None, start_step: int = 0) -> Dict[str, Any]:
    """Pretrain the reward model on preferences between the episodes in `arrays`.

    Args:
        arrays: flat transitions with observations, next_observations, rewards,
            terminals, is_success (+ subgoal_dones (N, G) when num_subgoals > 0);
            episodes are split at next_obs/obs discontinuities.
        rm_cfg: resolved reward_model config.
        num_subgoals: 0 for Dense, G for Pref-Subgoal.
        env_max_episode_length: with online updates, the episode store is padded
            to max(offline max length, this) so online episodes concatenate.

    Returns:
        dict with params, apply_fn, tx, opt_state, pref_buffer_offline,
        store_offline (the train split), history, metrics.
    """
    np_rng = np.random.default_rng(seed + 61001)
    subgoal_mode = num_subgoals > 0
    obs = np.asarray(arrays['observations'], dtype=np.float32)
    next_obs = np.asarray(arrays['next_observations'], dtype=np.float32)

    subgoal_dones = None
    if subgoal_mode:
        subgoal_dones = np.asarray(arrays['subgoal_dones'], dtype=np.float32)
        assert subgoal_dones.shape == (obs.shape[0], num_subgoals), (
            f"subgoal_dones {subgoal_dones.shape} != (N={obs.shape[0]}, G={num_subgoals})")
        if float(subgoal_dones.sum()) == 0.0:
            raise AssertionError("subgoal_dones are all zero: the dataset relabelling did not run")
        reward_targets = np.zeros((obs.shape[0],), dtype=np.float32)  # unused by subgoal labels
    elif rm_cfg['label_mode'] == 'success_return':
        rm_discount = float(rm_cfg['discount']) if rm_cfg.get('discount') is not None else float(discount)
        reward_targets = discounted_mc_returns(arrays['rewards'], arrays['terminals'], rm_discount)
    else:
        reward_targets = np.asarray(arrays['rewards'], dtype=np.float32)  # unused by success_only labels

    t_max_override = None
    if rm_cfg['online_enabled'] and env_max_episode_length is not None:
        t_max_override = max(natural_max_episode_length(obs, next_obs), int(env_max_episode_length))

    ep_store = build_episode_store(obs, next_obs, reward_targets, np.asarray(arrays['is_success']).astype(bool),
                                   t_max_override=t_max_override, subgoal_dones_per_step=subgoal_dones)
    n_eps = int(ep_store['lengths'].shape[0])
    print(f"[PBRL] Episode store: E={n_eps} | T_max={ep_store['obs'].shape[1]} | "
          f"len mean/min/max={ep_store['lengths'].mean():.1f}/{ep_store['lengths'].min()}/{ep_store['lengths'].max()} | "
          f"successful={int(ep_store['is_success'].sum())}/{n_eps} | obs {ep_store['obs'].nbytes / 1e6:.1f} MB")

    all_pairs = bool(rm_cfg['all_pairs'])
    feed_type = int(rm_cfg['feed_type'])
    if all_pairs:
        n_uniform, n_disagreement = n_eps * (n_eps - 1) // 2, 0
    else:
        total = max(1, int(round(n_eps * float(rm_cfg['reward_dataset_scale']))))
        n_uniform = total if feed_type == 0 else total // 2
        n_disagreement = total - n_uniform

    train_store, val_store = split_episode_store(ep_store, float(rm_cfg['val_fraction']), np_rng)
    n_val_pairs = int(round(n_uniform * float(rm_cfg['val_fraction']))) if (val_store is not None and not all_pairs) else 0
    n_train_pairs = 0 if all_pairs else max(1, n_uniform - n_val_pairs)

    def _pairs(store, n):
        if subgoal_mode:
            return generate_preference_pairs_subgoal(store, max(1, n // num_subgoals), np_rng, all_pairs=all_pairs,
                                                     subgoal_label_mode=rm_cfg['subgoal_label_mode'])
        return generate_preference_pairs(store, n, np_rng, all_pairs=all_pairs, label_mode=rm_cfg['label_mode'])

    if subgoal_mode:
        print(f"[PBRL] per-subgoal achievers (train store): "
              f"{train_store['subgoal_success'].sum(axis=0).tolist()} / {train_store['lengths'].shape[0]}")
    train_buffer = _pairs(train_store, n_train_pairs)
    val_buffer = _pairs(val_store, n_val_pairs) if val_store is not None else None
    if num_pairs(train_buffer) <= int(rm_cfg['min_offline_pairs']):
        raise AssertionError(
            f"[PBRL] only {num_pairs(train_buffer)} non-tie offline pairs (need > {rm_cfg['min_offline_pairs']}). "
            "With success labels this means too few failures/successes: add rollouts "
            "(reward_model.rollout_dataset_path) or lower reward_model.min_offline_pairs.")

    model = make_reward_model(rm_cfg, goal_dim=num_subgoals)
    params = init_reward_params(model, obs[:1], seed)
    tx = optax.adam(float(rm_cfg['lr']))
    opt_state = tx.init(params)

    params, opt_state, history = train_reward_model(
        params, model.apply, tx, opt_state, train_buffer, train_store, epochs=int(rm_cfg['train_epochs']),
        batch_size=int(rm_cfg['train_batch_size']), rng=np_rng, val_pref_buffer=val_buffer,
        val_ep_store=val_store, num_subgoals=num_subgoals)
    step = _log_history(logger, history, start_step)

    if feed_type == 1 and n_disagreement > 0:
        extra = sample_disagreement_pairs(train_store, params, model.apply, n_disagreement,
                                          int(rm_cfg['disagreement_oversample']), np_rng,
                                          label_mode=rm_cfg['label_mode'], num_subgoals=num_subgoals,
                                          subgoal_label_mode=rm_cfg['subgoal_label_mode'])
        train_buffer = concat_preference_buffers(train_buffer, extra)
        params, opt_state, h2 = train_reward_model(
            params, model.apply, tx, opt_state, train_buffer, train_store, epochs=int(rm_cfg['train_epochs']),
            batch_size=int(rm_cfg['train_batch_size']), rng=np_rng, val_pref_buffer=val_buffer,
            val_ep_store=val_store, num_subgoals=num_subgoals)
        history = history + h2
        step = _log_history(logger, h2, step)

    metrics = dict(history[-1]) if history else {}
    if subgoal_mode:
        accs = per_subgoal_accuracy(params, model.apply, train_buffer, train_store, num_subgoals,
                                    batch_size=int(rm_cfg['train_batch_size']))
        print(f"[PBRL] per-subgoal train BT accuracy: {[f'{a:.3f}' for a in accs]}")
        metrics.update({f'subgoal_{g}_train_accuracy': float(a) for g, a in enumerate(accs)})
        if logger is not None:
            logger.log({k: v for k, v in metrics.items() if k.startswith('subgoal_')}, 'reward_model',
                       step=max(step, int(start_step) + 1))
    print(f"[PBRL] Reward model pretraining done | pairs={num_pairs(train_buffer)} | {metrics}")
    return {
        'params': params,
        'apply_fn': model.apply,
        'tx': tx,
        'opt_state': opt_state,
        'pref_buffer_offline': train_buffer,
        'store_offline': train_store,
        'history': history,
        'metrics': metrics,
        'log_step': step,
    }
