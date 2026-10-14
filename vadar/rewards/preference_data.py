"""Episode stores, oracle preference labels and preference-pair generation (numpy only).

Preference pairs are whole episodes of variable length. Episodes are stored once
in an episode store (zero-padded to a global T_max, with per-episode lengths,
returns and success flags); a preference buffer holds only episode indices plus
labels. Every training batch is gathered from the store and padded to the same
T_max so the jitted train step traces once.

Ported from OGPO-pbrl-subgoal `ogpo/agents/modules/pbrl_helper.py` (a superset of
OGPO-pbrl-baseline's). Renamed for VaDaR: `subtask_status` -> `subgoal_dones`.
Segment-level preferences are not ported (the reference rejected
`preference_unit != episode` at config time).

Episode store keys:
    obs                    (E, T_max, D) float32, zero-padded
    lengths                (E,)          int32
    returns                (E,)          float32  sum of the per-step return targets
    is_success             (E,)          bool     any step flagged successful
    per_step_returns       (E, T_max)    float32
    per_step_success       (E, T_max)    bool
  with subgoal dones (Pref-Subgoal):
    per_step_subgoal_dones (E, T_max, G) float32  cumulative completion after each step
    subgoal_success        (E, G)        bool     subgoal completed at some step

Preference buffer keys: ep_idx_1, ep_idx_2 (int32), labels (float32; 1.0 means
episode 2 is preferred), and subgoal_idx (int32) in Pref-Subgoal.
"""

import glob
import os
from typing import Dict, Optional, Tuple

import numpy as np

SUCCESS_KEY = 'is_success'
ROLLOUT_KEYS = ('observations', 'next_observations', 'actions', 'rewards', 'terminals')


# ---------------------------------------------------------------------------
# Episode segmentation and stores
# ---------------------------------------------------------------------------

def episode_boundaries(observations: np.ndarray, next_observations: np.ndarray,
                       tol: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    """Per-episode (starts, ends_inclusive) of concatenated demos.

    `terminals` marks success events (not demo ends) for robomimic datasets, so
    boundaries are detected where next_observations[t-1] != observations[t].
    """
    observations = np.asarray(observations)
    next_observations = np.asarray(next_observations)
    assert observations.shape == next_observations.shape, (
        f"obs/next_obs shape mismatch: {observations.shape} vs {next_observations.shape}")
    n = observations.shape[0]
    if n == 0:
        raise ValueError("observations is empty")
    if n == 1:
        return np.array([0], dtype=np.int32), np.array([0], dtype=np.int32)
    diff_axes = tuple(range(1, observations.ndim))
    diff = np.max(np.abs(next_observations[:-1].astype(np.float32) - observations[1:].astype(np.float32)),
                  axis=diff_axes)
    boundary_starts = np.nonzero(diff > tol)[0] + 1
    starts = np.concatenate([[0], boundary_starts]).astype(np.int32)
    ends = np.concatenate([boundary_starts - 1, [n - 1]]).astype(np.int32)
    return starts, ends


def natural_max_episode_length(observations: np.ndarray, next_observations: np.ndarray) -> int:
    starts, ends = episode_boundaries(observations, next_observations)
    return int((ends - starts + 1).max())


def build_episode_store(observations: np.ndarray, next_observations: np.ndarray,
                        reward_targets: np.ndarray, success_per_step: np.ndarray,
                        t_max_override: Optional[int] = None,
                        subgoal_dones_per_step: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """Split a flat transition dataset into a padded per-episode store (schema in the module docstring)."""
    observations = np.asarray(observations, dtype=np.float32)
    next_observations = np.asarray(next_observations, dtype=np.float32)
    reward_targets = np.asarray(reward_targets, dtype=np.float32).reshape(-1)
    success_per_step = np.asarray(success_per_step).astype(bool).reshape(-1)
    assert observations.ndim == 2, f"expected (N, D) observations, got {observations.shape}"
    n = observations.shape[0]
    assert reward_targets.shape[0] == n and success_per_step.shape[0] == n
    if subgoal_dones_per_step is not None:
        subgoal_dones_per_step = np.asarray(subgoal_dones_per_step, dtype=np.float32)
        assert subgoal_dones_per_step.ndim == 2 and subgoal_dones_per_step.shape[0] == n, (
            f"expected (N, G) subgoal dones, got {subgoal_dones_per_step.shape}")

    starts, ends = episode_boundaries(observations, next_observations)
    n_eps = starts.shape[0]
    lengths = (ends - starts + 1).astype(np.int32)
    t_max = int(lengths.max())
    if t_max_override is not None:
        if int(t_max_override) < t_max:
            raise ValueError(f"t_max_override={t_max_override} < natural max episode length {t_max}")
        t_max = int(t_max_override)

    store = {
        'obs': np.zeros((n_eps, t_max, observations.shape[1]), dtype=np.float32),
        'lengths': lengths,
        'returns': np.zeros((n_eps,), dtype=np.float32),
        'is_success': np.zeros((n_eps,), dtype=bool),
        'per_step_returns': np.zeros((n_eps, t_max), dtype=np.float32),
        'per_step_success': np.zeros((n_eps, t_max), dtype=bool),
    }
    if subgoal_dones_per_step is not None:
        G = subgoal_dones_per_step.shape[1]
        store['per_step_subgoal_dones'] = np.zeros((n_eps, t_max, G), dtype=np.float32)
        store['subgoal_success'] = np.zeros((n_eps, G), dtype=bool)
    for i in range(n_eps):
        s, e = int(starts[i]), int(ends[i])
        L = e - s + 1
        store['obs'][i, :L] = observations[s:e + 1]
        store['per_step_returns'][i, :L] = reward_targets[s:e + 1]
        store['per_step_success'][i, :L] = success_per_step[s:e + 1]
        store['returns'][i] = reward_targets[s:e + 1].sum()
        store['is_success'][i] = bool(success_per_step[s:e + 1].any())
        if subgoal_dones_per_step is not None:
            store['per_step_subgoal_dones'][i, :L] = subgoal_dones_per_step[s:e + 1]
            store['subgoal_success'][i] = subgoal_dones_per_step[s:e + 1].any(axis=0)
    assert np.isfinite(store['returns']).all(), "Episode returns must be finite"
    return store


def empty_episode_store(t_max: int, obs_dim: int, num_subgoals: int = 0) -> Dict[str, np.ndarray]:
    """An episode store with zero episodes (same key schema as build_episode_store)."""
    store = {
        'obs': np.zeros((0, t_max, obs_dim), dtype=np.float32),
        'lengths': np.zeros((0,), dtype=np.int32),
        'returns': np.zeros((0,), dtype=np.float32),
        'is_success': np.zeros((0,), dtype=bool),
        'per_step_returns': np.zeros((0, t_max), dtype=np.float32),
        'per_step_success': np.zeros((0, t_max), dtype=bool),
    }
    if num_subgoals > 0:
        store['per_step_subgoal_dones'] = np.zeros((0, t_max, num_subgoals), dtype=np.float32)
        store['subgoal_success'] = np.zeros((0, num_subgoals), dtype=bool)
    return store


def single_episode_entry(obs: np.ndarray, per_step_returns: np.ndarray, is_success: bool, t_max: int,
                         subgoal_dones: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """Wrap one online trajectory as a 1-episode store padded to `t_max`.

    Only the last step is flagged successful (the env terminates on success).
    """
    L = obs.shape[0]
    if L > t_max:  # time-limited envs never exceed t_max; guard anyway
        L = t_max
        obs, per_step_returns = obs[:t_max], per_step_returns[:t_max]
        if subgoal_dones is not None:
            subgoal_dones = subgoal_dones[:t_max]
    entry = {
        'obs': np.zeros((1, t_max, obs.shape[1]), dtype=np.float32),
        'lengths': np.array([L], dtype=np.int32),
        'returns': np.array([float(np.sum(per_step_returns[:L]))], dtype=np.float32),
        'is_success': np.array([bool(is_success)], dtype=bool),
        'per_step_returns': np.zeros((1, t_max), dtype=np.float32),
        'per_step_success': np.zeros((1, t_max), dtype=bool),
    }
    entry['obs'][0, :L] = obs[:L]
    entry['per_step_returns'][0, :L] = per_step_returns[:L]
    if bool(is_success) and L > 0:
        entry['per_step_success'][0, L - 1] = True
    if subgoal_dones is not None:
        G = subgoal_dones.shape[1]
        entry['per_step_subgoal_dones'] = np.zeros((1, t_max, G), dtype=np.float32)
        entry['per_step_subgoal_dones'][0, :L] = subgoal_dones[:L]
        entry['subgoal_success'] = (subgoal_dones[:L] > 0.5).any(axis=0)[None, :]
    return entry


def concat_stores(a: Dict[str, np.ndarray], b: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Concatenate two stores with the same key schema along the episode axis."""
    if not a or a['obs'].shape[0] == 0:
        return {k: v.copy() for k, v in b.items()}
    if not b or b['obs'].shape[0] == 0:
        return {k: v.copy() for k, v in a.items()}
    return {k: np.concatenate([a[k], b[k]], axis=0) for k in a.keys()}


def split_episode_store(ep_store: Dict[str, np.ndarray], val_fraction: float,
                        rng: np.random.Generator):
    """Episode-level (train_store, val_store) split; val_store is None when not splitting."""
    n = int(ep_store['lengths'].shape[0])
    if n < 2 or val_fraction <= 0.0:
        return ep_store, None
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in [0, 1), got {val_fraction}")
    n_val = max(1, min(n - 1, int(round(n * val_fraction))))
    perm = rng.permutation(n)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    return ({k: v[train_idx] for k, v in ep_store.items()},
            {k: v[val_idx] for k, v in ep_store.items()})


# ---------------------------------------------------------------------------
# Oracle labels
# ---------------------------------------------------------------------------

def compute_oracle_labels_from_returns(ret1, ret2) -> np.ndarray:
    """1.0 when episode 2 has the higher return, 0.0 when episode 1, 0.5 on ties."""
    ret1 = np.asarray(ret1, dtype=np.float32).reshape(-1)
    ret2 = np.asarray(ret2, dtype=np.float32).reshape(-1)
    return np.where(ret1 < ret2, 1.0, np.where(ret1 > ret2, 0.0, 0.5)).astype(np.float32)


def compute_oracle_labels_with_success(ret1, ret2, success1, success2) -> np.ndarray:
    """Success strictly beats failure; returns break ties between equal-success episodes."""
    s1 = np.asarray(success1).astype(bool).reshape(-1)
    s2 = np.asarray(success2).astype(bool).reshape(-1)
    return_labels = compute_oracle_labels_from_returns(ret1, ret2)
    return np.where(s1 & ~s2, 0.0, np.where(~s1 & s2, 1.0, return_labels)).astype(np.float32)


def compute_oracle_labels_success_only(success1, success2) -> Tuple[np.ndarray, np.ndarray]:
    """(labels, keep_mask): the successful episode wins; same-success pairs must be dropped."""
    s1 = np.asarray(success1).astype(bool).reshape(-1)
    s2 = np.asarray(success2).astype(bool).reshape(-1)
    keep_mask = s1 != s2
    labels = np.where(s2 & ~s1, 1.0, 0.0).astype(np.float32)
    return labels, keep_mask


def empty_pref_buffer(subgoal: bool = False) -> Dict[str, np.ndarray]:
    buf = {
        'ep_idx_1': np.zeros((0,), dtype=np.int32),
        'ep_idx_2': np.zeros((0,), dtype=np.int32),
        'labels': np.zeros((0,), dtype=np.float32),
    }
    if subgoal:
        buf['subgoal_idx'] = np.zeros((0,), dtype=np.int32)
    return buf


def concat_preference_buffers(first: Dict[str, np.ndarray], second: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    if not first:
        return second
    if not second:
        return first
    return {k: np.concatenate([first[k], second[k]], axis=0) for k in first.keys()}


def num_pairs(buf: Optional[Dict[str, np.ndarray]]) -> int:
    return 0 if not buf else int(buf['labels'].shape[0])


# ---------------------------------------------------------------------------
# Pair generation
# ---------------------------------------------------------------------------

def generate_preference_pairs(ep_store: Dict[str, np.ndarray], n_pairs: int, rng: np.random.Generator,
                              all_pairs: bool = False, label_mode: str = 'success_only',
                              success_flags: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """Enumerate every unordered episode pair, label it, drop ties, then subsample.

    label_mode:
        success_return  success beats failure, return tie-break; a tie is an equal-success,
                        equal-return pair.
        success_only    labels from success flags only; a tie is an equal-success pair.
    With all_pairs=True the whole non-tie pool is returned and n_pairs is ignored.
    """
    if label_mode not in ('success_return', 'success_only'):
        raise ValueError(f"label_mode must be 'success_return' or 'success_only', got {label_mode!r}")
    n_eps = int(ep_store['lengths'].shape[0])
    if n_eps < 2:
        raise ValueError(f"generate_preference_pairs needs at least 2 episodes, got {n_eps}")
    is_success = ep_store['is_success'] if success_flags is None else success_flags
    is_success = np.asarray(is_success).astype(bool)
    assert is_success.shape[0] == n_eps

    i_full, j_full = np.triu_indices(n_eps, k=1)
    s1, s2 = is_success[i_full], is_success[j_full]
    if label_mode == 'success_only':
        labels_full, keep = compute_oracle_labels_success_only(s1, s2)
    else:
        returns = np.asarray(ep_store['returns'], dtype=np.float32)
        labels_full = compute_oracle_labels_with_success(returns[i_full], returns[j_full], s1, s2)
        keep = labels_full != 0.5

    i_pool = i_full[keep].astype(np.int32)
    j_pool = j_full[keep].astype(np.int32)
    labels_pool = labels_full[keep].astype(np.float32)
    pool, total = int(labels_pool.shape[0]), int(labels_full.shape[0])
    print(f"[PBRL] {label_mode}: non-tie pool {pool} (dropped ties {total - pool}/{total})")
    if pool == 0:
        return empty_pref_buffer()
    if all_pairs:
        return {'ep_idx_1': i_pool, 'ep_idx_2': j_pool, 'labels': labels_pool}
    n = int(n_pairs)
    if pool >= n:
        sel = rng.choice(pool, size=n, replace=False)
    else:
        print(f"[PBRL] {label_mode}: pool {pool} < n_pairs {n}; sampling with replacement.")
        sel = rng.integers(0, pool, size=n)
    return {'ep_idx_1': i_pool[sel], 'ep_idx_2': j_pool[sel], 'labels': labels_pool[sel]}


def compute_first_achievement_steps(ep_store: Dict[str, np.ndarray]) -> np.ndarray:
    """(E, G) first step at which each subgoal is done; inf where it never is."""
    dones = np.asarray(ep_store['per_step_subgoal_dones'])
    lengths = np.asarray(ep_store['lengths'], dtype=np.int64)
    n_units, _, G = dones.shape
    first = np.full((n_units, G), np.inf, dtype=np.float32)
    for e in range(n_units):
        done = dones[e, :int(lengths[e])] > 0.5
        any_done = done.any(axis=0)
        first[e, any_done] = done.argmax(axis=0)[any_done]
    return first


def generate_preference_pairs_subgoal(ep_store: Dict[str, np.ndarray], n_pairs_per_subgoal: int,
                                      rng: np.random.Generator, all_pairs: bool = False,
                                      subgoal_label_mode: str = 'success_time') -> Dict[str, np.ndarray]:
    """Per-subgoal preference pairs tagged with `subgoal_idx`, concatenated over subgoals.

    subgoal_label_mode:
        success_only  an episode completing subgoal i beats one that does not; other pairs dropped.
        success_time  as success_only, and between two completers the earlier completion wins;
                      equal-step and neither-completes pairs dropped.
    The budget n_pairs_per_subgoal applies to each subgoal separately.
    """
    if 'subgoal_success' not in ep_store:
        raise KeyError("generate_preference_pairs_subgoal needs a store built with subgoal dones")
    if subgoal_label_mode not in ('success_only', 'success_time'):
        raise ValueError(f"subgoal_label_mode must be 'success_only' or 'success_time', got {subgoal_label_mode!r}")
    subgoal_success = np.asarray(ep_store['subgoal_success']).astype(bool)
    n_units, G = subgoal_success.shape
    first_steps = compute_first_achievement_steps(ep_store) if subgoal_label_mode == 'success_time' else None

    buffers, counts = [], []
    for g in range(G):
        if subgoal_label_mode == 'success_time':
            i_full, j_full = np.triu_indices(n_units, k=1)
            f1, f2 = first_steps[i_full, g], first_steps[j_full, g]
            keep = f1 != f2                       # equal steps (incl. both never) are ties
            labels_full = (f2 < f1).astype(np.float32)  # 1.0: episode 2 reached g earlier
            i_pool, j_pool = i_full[keep].astype(np.int32), j_full[keep].astype(np.int32)
            labels_pool = labels_full[keep]
            pool = int(labels_pool.shape[0])
            if pool == 0:
                pairs = empty_pref_buffer()
            elif all_pairs or pool <= n_pairs_per_subgoal:
                pairs = {'ep_idx_1': i_pool, 'ep_idx_2': j_pool, 'labels': labels_pool}
            else:
                sel = rng.choice(pool, size=int(n_pairs_per_subgoal), replace=False)
                pairs = {'ep_idx_1': i_pool[sel], 'ep_idx_2': j_pool[sel], 'labels': labels_pool[sel]}
        else:
            pairs = generate_preference_pairs(ep_store, n_pairs=n_pairs_per_subgoal, rng=rng,
                                              all_pairs=all_pairs, label_mode='success_only',
                                              success_flags=subgoal_success[:, g])
        n_g = num_pairs(pairs)
        counts.append(n_g)
        if n_g == 0:
            continue
        pairs['subgoal_idx'] = np.full((n_g,), g, dtype=np.int32)
        buffers.append(pairs)

    print(f"[PBRL] subgoal pair counts: {counts}")
    starved = [g for g, c in enumerate(counts) if c == 0]
    if starved:
        print(f"[PBRL] WARNING: subgoals {starved} contributed 0 pairs (no contrast in the store).")
    if not buffers:
        return empty_pref_buffer(subgoal=True)
    out = buffers[0]
    for b in buffers[1:]:
        out = concat_preference_buffers(out, b)
    return out


def truncate_fifo(buf: Dict[str, np.ndarray], cap: int) -> Dict[str, np.ndarray]:
    """Keep the newest `cap` pairs."""
    if num_pairs(buf) <= cap:
        return buf
    return {k: v[-cap:] for k, v in buf.items()}


# ---------------------------------------------------------------------------
# Return targets (label_mode=success_return only)
# ---------------------------------------------------------------------------

def discounted_mc_returns(rewards: np.ndarray, terminals: np.ndarray, discount: float) -> np.ndarray:
    """Per-step discounted MC returns, restarting at every terminal (reference: mc_return_sparse)."""
    rewards = np.asarray(rewards, dtype=np.float32).reshape(-1)
    terminals = np.asarray(terminals).astype(bool).reshape(-1)
    assert rewards.shape == terminals.shape and rewards.shape[0] > 0
    out = np.zeros_like(rewards)
    running = 0.0
    for t in reversed(range(rewards.shape[0])):
        running = rewards[t] if terminals[t] else rewards[t] + discount * running
        out[t] = running
    return out


# ---------------------------------------------------------------------------
# Extra rollout trajectories for pretraining
# ---------------------------------------------------------------------------

def _rollout_files(path: str):
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, '*.h5')) + glob.glob(os.path.join(path, '*.hdf5'))
                       + glob.glob(os.path.join(path, '*.npz')))
        if not files:
            raise ValueError(f"rollout_dataset_path={path!r} contains no *.h5, *.hdf5 or *.npz files")
        return files
    if os.path.isfile(path):
        return [path]
    raise ValueError(f"rollout_dataset_path={path!r} is neither a file nor a directory")


def load_rollout_dataset(path: str, need_subgoal_dones: bool = False) -> Dict[str, np.ndarray]:
    """Load rollout trajectories saved as flat or robomimic-style ("data/demo_k") HDF5, or npz.

    Success per step comes from an `is_success` array when present, else from
    `terminals` (the collectors mark successful termination there). Returns a
    flat dict with ROLLOUT_KEYS + is_success (+ subgoal_dones).
    """
    keys = ROLLOUT_KEYS + (('subgoal_dones',) if need_subgoal_dones else ())
    parts = {k: [] for k in keys}
    success_parts = []

    def _add(arrays, success):
        missing = [k for k in keys if k not in arrays]
        if missing:
            raise KeyError(f"rollout data is missing {missing}"
                           + (" (Pref-Subgoal needs per-step subgoal_dones in rollouts)" if 'subgoal_dones' in missing else ""))
        for k in keys:
            parts[k].append(np.asarray(arrays[k]))
        s = np.asarray(success if success is not None else arrays['terminals'], dtype=np.float32).reshape(-1)
        assert s.shape[0] == np.asarray(arrays['rewards']).shape[0]
        success_parts.append(s)

    for fp in _rollout_files(path):
        if fp.endswith('.npz'):
            with np.load(fp) as npz:
                arrays = {k: npz[k] for k in npz.files}
            _add(arrays, arrays.get(SUCCESS_KEY))
            print(f"[PBRL] Loaded rollout file {fp} ({arrays['rewards'].shape[0]} transitions)")
            continue
        import h5py
        with h5py.File(fp, 'r') as f:
            if 'data' in f and isinstance(f['data'], h5py.Group):
                demos = sorted(f['data'].keys(), key=lambda s: int(s.rsplit('_', 1)[-1])
                               if s.rsplit('_', 1)[-1].isdigit() else s)
                n_succ = 0
                for demo in demos:
                    grp = f['data'][demo]
                    arrays = {k: grp[k][...] for k in grp.keys() if isinstance(grp[k], h5py.Dataset)}
                    _add(arrays, arrays.get(SUCCESS_KEY))
                    n_succ += int(success_parts[-1].any())
                print(f"[PBRL] Loaded rollout file {fp} ({len(demos)} demos, {n_succ} successful)")
            else:
                arrays = {k: f[k][...] for k in f.keys() if isinstance(f[k], h5py.Dataset)}
                _add(arrays, arrays.get(SUCCESS_KEY))
                print(f"[PBRL] Loaded rollout file {fp} ({arrays['rewards'].shape[0]} transitions)")

    out = {k: np.concatenate(v, axis=0) for k, v in parts.items()}
    out[SUCCESS_KEY] = np.concatenate(success_parts, axis=0)
    return out


def dataset_arrays_for_pbrl(train_dataset, need_subgoal_dones: bool = False) -> Dict[str, np.ndarray]:
    """The fields of the offline dataset the reward model is pretrained on.

    Success per step is `is_success` when the dataset has it, else `terminals`.
    """
    def _has(k):
        try:
            return k in train_dataset
        except TypeError:
            return k in getattr(train_dataset, '_dict', {})

    keys = ROLLOUT_KEYS + (('subgoal_dones',) if need_subgoal_dones else ())
    missing = [k for k in keys if not _has(k)]
    if missing:
        raise KeyError(f"train_dataset lacks {missing}"
                       + ("; build the env/datasets with subgoal tracking for pref_subgoal"
                          if 'subgoal_dones' in missing else ""))
    out = {k: np.asarray(train_dataset[k]) for k in keys}
    out[SUCCESS_KEY] = np.asarray(train_dataset[SUCCESS_KEY] if _has(SUCCESS_KEY) else train_dataset['terminals'],
                                  dtype=np.float32)
    return out


def concat_transition_arrays(a: Dict[str, np.ndarray], b: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {k: np.concatenate([a[k], b[k]], axis=0) for k in a.keys()}


# ---------------------------------------------------------------------------
# Episode-length cap for online episode stores
# ---------------------------------------------------------------------------

def resolve_env_max_episode_length(env_name: str, env=None) -> Optional[int]:
    """Best-effort max episode length of the training env (None if unknown).

    Tries, in order: the env wrapper chain (`max_episode_length`, `_max_episode_steps`,
    `spec.max_episode_steps`), then the per-suite helpers in envs/.
    """
    e = env
    while e is not None:
        for attr in ('max_episode_length', '_max_episode_steps'):
            v = getattr(e, attr, None)
            if isinstance(v, (int, np.integer)) and v > 0:
                return int(v)
        spec = getattr(e, 'spec', None)
        v = getattr(spec, 'max_episode_steps', None) if spec is not None else None
        if isinstance(v, (int, np.integer)) and v > 0:
            return int(v)
        e = getattr(e, 'env', None)

    try:
        if env_name.startswith('robocasa_'):
            from envs.robocasa_utils import MAX_EPISODE_LENGTH, _TASK_MAX_EP_LEN, _get_task_name
            return int(_TASK_MAX_EP_LEN.get(_get_task_name(env_name), MAX_EPISODE_LENGTH))
        if env_name.startswith('libero_'):
            from envs.libero_utils import _get_task_config
            return int(_get_task_config(env_name)['max_episode_length'])
        from envs.robomimic_utils import is_robomimic_env, _get_max_episode_length
        if is_robomimic_env(env_name):
            return int(_get_max_episode_length(env_name))
    except (ImportError, AttributeError, KeyError):
        pass
    return None
