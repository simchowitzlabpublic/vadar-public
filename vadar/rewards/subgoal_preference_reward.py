"""Pref-Subgoal: aggregate per-subgoal preference rewards with a VaDaR weighting.

The reward model r(s, g) is conditioned on a one-hot subgoal e_g. The scalar
reward of a transition (s_t, a_t, s_{t+1}) is

    r_t = sum_g w(d_{t+1})[g] * r(s_t, e_g)  +  subgoal_done_bonus * sum_g d_{t+1}[g]

where d_{t+1} = info['subgoal_dones'] after the step (the same post-action
labels the replay buffer stores) and w is a VaDaR policy-extraction weighting,
next-future by default. This reward replaces the env reward and trains the
scalar critic of the sparse backbone.

`subgoal_weights_np` is a numpy copy of `vadar.agents.modules.vadar_helper.subgoal_weights`
(sequential and DAG cases). It runs once per env step, so it avoids JAX. It
also equals the fork's `compute_gate_weights_np` under the name mapping in
`_FORK_WEIGHTING_NAMES` (checked in tests/test_pbrl_rewards.py).
"""

from typing import Dict, Optional

import numpy as np

# Same table as vadar_helper.WEIGHTING_ALIASES (kept jax-free here), plus the
# OGPO-pbrl-subgoal fork's gating names.
_WEIGHTING_ALIASES = {
    'nf': 'next_future', 'next_future': 'next_future',
    'cf': 'current_future', 'current_future': 'current_future',
    'gm': 'current_myopic', 'cm': 'current_myopic', 'current_myopic': 'current_myopic',
    'nm': 'next_myopic', 'next_myopic': 'next_myopic',
    'last': 'last', 'lastq': 'last', 'last_q': 'last',
}
_FORK_WEIGHTING_NAMES = {
    'grandparent_future': 'next_future',
    'goal_future': 'current_future',
    'goal_myopic': 'current_myopic',
    'last_only': 'last',
}


def canonical_weighting(name: str) -> str:
    key = str(name).lower()
    if key in _FORK_WEIGHTING_NAMES:
        return _FORK_WEIGHTING_NAMES[key]
    if key not in _WEIGHTING_ALIASES:
        raise ValueError(f"Unknown subgoal weighting {name!r} for Pref-Subgoal. "
                         f"Options: {sorted(set(_WEIGHTING_ALIASES.values()))}")
    return _WEIGHTING_ALIASES[key]


def normalize_dag(dag) -> Optional[Dict[int, list]]:
    if dag is None:
        return None
    return {int(k): [int(p) for p in v] for k, v in dict(dag).items()}


def resolve_subgoal_dag(cfg_dag, env_name: str):
    """`subgoal_dag` config value -> DAG dict or None. "spec" takes the DAG of envs.subgoals."""
    if cfg_dag is None:
        return None
    if isinstance(cfg_dag, str):
        if cfg_dag.lower() != 'spec':
            raise ValueError(f"reward_model.subgoal_dag must be null, 'spec' or a dict, got {cfg_dag!r}")
        from envs.subgoals import get_subgoal_spec
        return normalize_dag(get_subgoal_spec(env_name)['dag'])
    return normalize_dag(cfg_dag)


def subgoal_weights_np(dones: np.ndarray, weighting: str = 'next_future', dag=None) -> np.ndarray:
    """numpy version of vadar_helper.subgoal_weights. dones (B, G) -> weights (G, B)."""
    weighting = canonical_weighting(weighting)
    d = np.asarray(dones, dtype=np.float32).T  # (G, B)
    G, B = d.shape
    if weighting == 'last':
        w = np.zeros((G, B), dtype=np.float32)
        w[-1] = 1.0
        return w
    dag = normalize_dag(dag)
    if dag is not None:
        return _dag_weights_np(d, weighting, dag)
    return _sequential_weights_np(d, weighting)


def _sequential_weights_np(d, weighting):
    G, B = d.shape
    current_idx = np.argmin(d, axis=0)                               # first incomplete subgoal
    current = np.eye(G, dtype=np.float32)[current_idx].T             # (G, B)
    parent_done = np.concatenate([np.ones((1, B), dtype=np.float32), d[:-1]], axis=0)
    if weighting == 'current_myopic':
        return current * (1.0 - d)
    if weighting == 'current_future':
        w = 1.0 - d
        w[-1] = 1.0
        return w
    if weighting == 'next_future':
        w = 1.0 - parent_done
        w[-1] = 1.0
        return w
    if weighting == 'next_myopic':
        all_done = d.min(axis=0) > 0
        next_idx = np.where(all_done, G - 1, np.minimum(current_idx + 1, G - 1))
        return np.eye(G, dtype=np.float32)[next_idx].T
    raise ValueError(f"Unsupported sequential weighting: {weighting}")


def _dag_weights_np(d, weighting, dag):
    G, B = d.shape
    prereqs_done = np.stack([
        d[np.asarray(dag[g], dtype=np.int64)].prod(axis=0) if dag.get(g) else np.ones(B, dtype=np.float32)
        for g in range(G)
    ])
    current = prereqs_done * (1.0 - d)
    if weighting == 'current_myopic':
        w = current.copy()
        w[-1] = 1.0
        return w
    if weighting == 'current_future':
        w = 1.0 - d
        w[-1] = 1.0
        return w
    if weighting == 'next_future':
        w = 1.0 - prereqs_done
        w[-1] = 1.0
        return w
    if weighting == 'next_myopic':
        has_child = np.zeros(G, dtype=bool)
        for g, parents in dag.items():
            for p in parents:
                has_child[p] = True
        children_of_current = np.stack([
            current[np.asarray(dag[g], dtype=np.int64)].max(axis=0) if dag.get(g) else np.zeros(B, dtype=np.float32)
            for g in range(G)
        ])
        terminal = (~has_child).astype(np.float32)[:, None]
        w = np.maximum(children_of_current, current * terminal)
        last_only = np.zeros((G, B), dtype=np.float32)
        last_only[-1] = 1.0
        all_done = d.min(axis=0) > 0
        return np.where(all_done[None], last_only, w)
    raise ValueError(f"Unsupported DAG weighting: {weighting}")


def aggregate_subgoal_rewards(subgoal_rewards: np.ndarray, subgoal_dones: np.ndarray,
                              weighting: str = 'next_future', done_bonus: float = 0.0,
                              dag=None) -> np.ndarray:
    """Scalarize per-subgoal rewards.

    Args:
        subgoal_rewards: (G, B) ensemble-mean r(s, e_g).
        subgoal_dones: (B, G) completion indicators after the step.

    Returns:
        (B,) sum_g w[g] r_g + done_bonus * sum_g d_g.
    """
    r = np.asarray(subgoal_rewards, dtype=np.float32)
    d = np.asarray(subgoal_dones, dtype=np.float32)
    assert r.shape == (d.shape[1], d.shape[0]), f"subgoal_rewards {r.shape} vs dones {d.shape}: expected (G, B), (B, G)"
    total = (subgoal_weights_np(d, weighting, dag) * r).sum(axis=0)
    if done_bonus != 0.0:
        total = total + float(done_bonus) * d.sum(axis=1)
    return total


class SubgoalRewardAggregator:
    """Callable (r_sub (G, B), dones (B, G)) -> (B,) with the run's weighting, bonus and DAG."""

    def __init__(self, num_subgoals: int, weighting: str = 'next_future', done_bonus: float = 0.0, dag=None):
        self.num_subgoals = int(num_subgoals)
        self.weighting = canonical_weighting(weighting)
        self.done_bonus = float(done_bonus)
        self.dag = normalize_dag(dag)

    def __call__(self, subgoal_rewards, subgoal_dones):
        return aggregate_subgoal_rewards(subgoal_rewards, subgoal_dones, weighting=self.weighting,
                                         done_bonus=self.done_bonus, dag=self.dag)

    def __repr__(self):
        return (f"SubgoalRewardAggregator(G={self.num_subgoals}, weighting={self.weighting}, "
                f"done_bonus={self.done_bonus}, dag={self.dag})")
