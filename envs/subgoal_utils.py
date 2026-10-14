"""Shared helpers for the VaDaR environment contract (see docs/DESIGN.md).

Used by the per-suite env modules (robomimic, robocasa, libero, ogbench):
`subgoal_cfg` validation, cumulative done ratcheting, goal-vector encoding,
reward modes and the on-disk cache of offline subgoal labels.

Label convention (all suites): `subgoal_dones[t]` is the cumulative completion
vector AFTER executing action t (i.e. at the state of next_observations[t]),
exactly what the online env reports in `info['subgoal_dones']` after
`env.step(actions[t])`. The goal suffix of observations[t] is the status
before the action (subgoal_dones[t-1], zeros at the episode start) and that of
next_observations[t] is subgoal_dones[t].
"""

import hashlib
import json
import os

import numpy as np

# Goal-vector encodings appended to observations.
#   'cumulative': which subgoals are done so far, e.g. (1, 1, 0, 0, 0)
#   'singular':   frontier subgoals, e.g. (0, 0, 1, 0, 0)
GOAL_OBS_MODES = (None, 'cumulative', 'singular')
# HER goal suffixes ('state': object-state goal, 'latent': encoder embedding).
HER_GOAL_OBS_MODES = ('state', 'latent')
REWARD_MODES = ('sparse', 'hgc', 'shaped', 'dense_parity')
HGC_BUMP = 5.0


def parse_subgoal_cfg(env_name, subgoal_cfg):
    """Validate `subgoal_cfg` against envs/subgoals.py and fill defaults (None stays None)."""
    if subgoal_cfg is None:
        return None
    from envs.subgoals import get_subgoal_spec
    spec = get_subgoal_spec(env_name)
    cfg = dict(subgoal_cfg)
    G = cfg.get('num_subgoals')
    G = spec['num_subgoals'] if G is None else int(G)
    if G != spec['num_subgoals']:
        raise ValueError(f"{env_name}: num_subgoals={G} but the spec defines {spec['num_subgoals']} "
                         f"({spec['names']})")
    cfg['num_subgoals'] = G
    dag = cfg['dag'] if 'dag' in cfg else spec['dag']
    cfg['dag'] = None if dag is None else {int(k): [int(p) for p in v] for k, v in dict(dag).items()}
    cfg['reward_mode'] = cfg.get('reward_mode') or 'sparse'
    if cfg['reward_mode'] not in REWARD_MODES:
        raise ValueError(f"Unknown reward_mode {cfg['reward_mode']!r}; choose from {REWARD_MODES}")
    cfg.setdefault('goal_obs', None)
    if cfg['goal_obs'] not in GOAL_OBS_MODES + HER_GOAL_OBS_MODES:
        raise ValueError(f"Unknown goal_obs {cfg['goal_obs']!r}; "
                         f"choose from {GOAL_OBS_MODES + HER_GOAL_OBS_MODES}")
    if cfg['goal_obs'] in HER_GOAL_OBS_MODES and cfg['reward_mode'] != 'sparse':
        raise ValueError("HER goal_obs modes define their own goal-space reward; use reward_mode='sparse'")
    cfg.setdefault('her_pos_threshold', 0.05)
    cfg.setdefault('her_encoder_path', None)
    cfg.setdefault('her_bc_goal', 'final')
    cfg['names'] = list(spec['names'])
    return cfg


def prerequisites(num_subgoals, dag=None, mode='sequential'):
    """Per-subgoal prerequisite lists used by `ratchet`.

    mode='sticky': no prerequisites (robomimic); mode='sequential': all earlier
    subgoals, or the DAG's parents when `dag` is given.
    """
    if mode == 'sticky':
        return [[] for _ in range(num_subgoals)]
    if dag is not None:
        return [list(dag.get(i, [])) for i in range(num_subgoals)]
    return [list(range(i)) for i in range(num_subgoals)]


def ratchet(goal_done, raw_goals, prereqs):
    """Sticky ratchet: subgoal i becomes (and stays) done once its check holds and its
    prerequisites are done. Updated in index order, so a prerequisite completed on
    this step already counts (as in the fork)."""
    goal_done = list(goal_done)
    for i, ok in enumerate(raw_goals[:len(goal_done)]):
        if ok and all(goal_done[p] for p in prereqs[i]):
            goal_done[i] = True
    return goal_done


def cumulative_to_singular(cumulative):
    """Cumulative done flags -> frontier flags: 1 at incomplete goals whose index
    predecessor is done (or that are first). (Index-chain rule, as in the fork.)"""
    arr = np.asarray(cumulative)
    incomplete = arr == 0
    pred_done = np.concatenate(
        [np.ones((*arr.shape[:-1], 1), dtype=arr.dtype), arr[..., :-1]],
        axis=-1,
    ).astype(bool)
    return np.where(incomplete & pred_done, np.int8(1), np.int8(0))


def encode_goal_vector(cumulative, mode):
    """Encode (..., G) binary done flags as the 'cumulative' or 'singular' goal vector (int8)."""
    if mode == 'cumulative':
        return np.asarray(cumulative, dtype=np.int8)
    if mode == 'singular':
        return cumulative_to_singular(cumulative)
    raise ValueError(f'Unknown goal_obs mode: {mode!r}. Choose from {GOAL_OBS_MODES}.')


def pre_action_status(post_labels, episode_starts):
    """Status before each action: post_labels shifted by one within each episode, zeros at starts."""
    post_labels = np.asarray(post_labels)
    pre = np.zeros_like(post_labels)
    pre[1:] = post_labels[:-1]
    pre[np.asarray(episode_starts, dtype=bool)] = 0
    return pre


def goal_suffixes(post_labels, episode_starts, mode):
    """(obs_goal, next_obs_goal) float32 arrays for the dataset (see module docstring)."""
    pre = pre_action_status(post_labels, episode_starts)
    return (encode_goal_vector(pre, mode).astype(np.float32),
            encode_goal_vector(post_labels, mode).astype(np.float32))


def get_shaped_reward_fn():
    """VaDaR-Shaped reward (imported lazily: vadar_helper pulls in jax)."""
    from vadar.agents.modules.vadar_helper import shaped_reward
    return shaped_reward


def hgc_reward(base_reward, prev_dones, dones):
    """HGC reward: base reward + HGC_BUMP if a subgoal newly completed on this step.

    `prev_dones=None` means no bump can fire (first step of an episode in the
    fork's RoboCasa wrapper)."""
    bump = prev_dones is not None and list(prev_dones) != list(dones)
    return float(base_reward) + (HGC_BUMP if bump else 0.0)


def relabel_rewards(reward_mode, base_rewards, post_labels, episode_starts, first_step_bump=True):
    """Offline rewards matching the online wrapper for reward_mode 'hgc' / 'shaped'.

    hgc: base_rewards + HGC_BUMP where the post-action dones changed w.r.t. the previous
    step (zeros before the first step; no bump at episode starts if not first_step_bump).
    shaped: shaped_reward(post_labels). Other modes return base_rewards unchanged.
    """
    base_rewards = np.asarray(base_rewards, dtype=np.float32)
    post_labels = np.asarray(post_labels, dtype=np.float32)
    if reward_mode == 'hgc':
        pre = pre_action_status(post_labels, episode_starts)
        bump = np.any(post_labels != pre, axis=-1)
        if not first_step_bump:
            bump &= ~np.asarray(episode_starts, dtype=bool)
        return (base_rewards + HGC_BUMP * bump).astype(np.float32)
    if reward_mode == 'shaped':
        return np.asarray(get_shaped_reward_fn()(post_labels), dtype=np.float32)
    return base_rewards


def dag_tag(dag):
    """Short stable tag of a DAG for cache file names ('' for None)."""
    if dag is None:
        return ''
    blob = json.dumps({str(k): list(v) for k, v in sorted(dag.items())}, sort_keys=True)
    return '_dag' + hashlib.md5(blob.encode()).hexdigest()[:8]


def label_cache_path(dataset_path, num_subgoals, dag=None, tag=''):
    """Cache file for post-action subgoal labels, next to the dataset.

    `tag` distinguishes relabels that depend on more than the dataset, e.g. the
    RoboCasa scene (anchor demo) the stored states are replayed into.
    """
    tag = f"_{tag}" if tag else ''
    return f"{dataset_path}.subgoal_dones_post_G{num_subgoals}{dag_tag(dag)}{tag}.npy"


def load_or_compute_labels(cache_path, num_steps, num_subgoals, compute_fn):
    """Load (num_steps, G) labels from `cache_path`, else compute_fn() and try to cache them."""
    if os.path.exists(cache_path):
        cached = np.load(cache_path)
        if cached.shape == (num_steps, num_subgoals):
            print(f"[subgoals] loaded post-action labels from {cache_path}")
            return cached.astype(np.float32)
        print(f"[subgoals] cache {cache_path} has shape {cached.shape}, "
              f"expected {(num_steps, num_subgoals)}; recomputing")
    labels = np.asarray(compute_fn(), dtype=np.float32)
    assert labels.shape == (num_steps, num_subgoals), (labels.shape, (num_steps, num_subgoals))
    try:
        tmp = cache_path + f'.tmp{os.getpid()}.npy'
        np.save(tmp, labels)
        os.replace(tmp, cache_path)
        print(f"[subgoals] wrote post-action labels to {cache_path}")
    except OSError as e:
        print(f"[subgoals] could not cache labels at {cache_path}: {e}")
    return labels
