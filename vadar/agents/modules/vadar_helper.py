"""Value Disaggregation (VaDaR): subgoal weightings and all-subgoal TD targets.

Notation follows the paper. G subgoals, completion indicators d in {0,1}^G
(cumulative: once a subgoal is done it stays done), current subgoal
g_curr = first incomplete subgoal, next subgoal g_next = min(g_curr + 1, g_last).

Policy extraction scalarizes the per-subgoal critic with a weighting w(d):
    Q_agg(s, a) = sum_g w[g] * Q(s, a, g)

Weightings (paper Table 1, plus the ablations):
    next_future     (NF, default)  w[g] = 1[g >= g_next]
    current_future  (CF)           w[g] = 1[g >= g_curr]
    current_myopic  (GM)           w[g] = 1[g == g_curr]
    next_myopic     (NM)           w[g] = 1[g == g_next]
    last            (VaDaR Last-Q) w[g] = 1[g == g_last]
    constant                       w = config subgoal_weights (multi-objective)

With a subgoal DAG, "current" is the set of incomplete subgoals whose
prerequisites are all done and "next" is the set of their children
(paper Appendix, DAG extension).
"""

from typing import Dict, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np

WEIGHTING_ALIASES = {
    'nf': 'next_future',
    'next_future': 'next_future',
    'cf': 'current_future',
    'current_future': 'current_future',
    'gm': 'current_myopic',
    'cm': 'current_myopic',
    'current_myopic': 'current_myopic',
    'nm': 'next_myopic',
    'next_myopic': 'next_myopic',
    'last': 'last',
    'lastq': 'last',
    'last_q': 'last',
    'constant': 'constant',
}


def canonical_weighting(name: str) -> str:
    """Map a weighting name or alias (NF, CF, GM, NM, lastq, ...) to its canonical name."""
    key = str(name).lower()
    if key not in WEIGHTING_ALIASES:
        raise ValueError(f"Unknown subgoal weighting {name!r}. Options: {sorted(set(WEIGHTING_ALIASES.values()))}")
    return WEIGHTING_ALIASES[key]


def normalize_dag(dag) -> Optional[Dict[int, list]]:
    """Return the subgoal DAG as {subgoal: [prerequisites]} with int keys (or None)."""
    if dag is None:
        return None
    return {int(k): [int(p) for p in v] for k, v in dict(dag).items()}


def subgoal_weights(dones, weighting='next_future', dag=None, constant_weights=None):
    """Policy-extraction weights w(d).

    Args:
        dones: (B, G) subgoal completion indicators.
        weighting: one of the weightings in the module docstring (aliases allowed).
        dag: optional {subgoal: [prerequisites]}. None means a sequential chain.
        constant_weights: (G,) weights for the 'constant' weighting.

    Returns:
        (G, B) weights.
    """
    weighting = canonical_weighting(weighting)
    d = jnp.asarray(dones, dtype=jnp.float32).T  # (G, B)
    G, B = d.shape

    if weighting == 'last':
        return jnp.zeros((G, B)).at[-1].set(1.0)
    if weighting == 'constant':
        assert constant_weights is not None, "the 'constant' weighting needs subgoal_weights in the config"
        w = jnp.asarray(constant_weights, dtype=jnp.float32)
        assert w.shape == (G,), f"subgoal_weights must have length {G}, got {w.shape}"
        return jnp.broadcast_to(w[:, None], (G, B))

    dag = normalize_dag(dag)
    if dag is not None:
        return _dag_weights(d, weighting, dag)
    return _sequential_weights(d, weighting)


def _sequential_weights(d, weighting):
    """Weights for a chain g_1 -> g_2 -> ... -> g_G. `d` is (G, B)."""
    G, B = d.shape
    current_idx = jnp.argmin(d, axis=0)                          # first incomplete subgoal
    current = jax.nn.one_hot(current_idx, G).T                   # (G, B)
    parent_done = jnp.concatenate([jnp.ones((1, B)), d[:-1]], axis=0)  # d[g-1], with d[-1] = 1

    if weighting == 'current_myopic':
        return current * (1.0 - d)
    if weighting == 'current_future':
        return (1.0 - d).at[-1].set(1.0)
    if weighting == 'next_future':
        # 1 - d[g-1] is 1 exactly for subgoals after g_curr; the final subgoal always counts.
        return (1.0 - parent_done).at[-1].set(1.0)
    if weighting == 'next_myopic':
        all_done = (d.min(axis=0) > 0)
        next_idx = jnp.where(all_done, G - 1, jnp.minimum(current_idx + 1, G - 1))
        return jax.nn.one_hot(next_idx, G).T
    raise ValueError(f"Unsupported sequential weighting: {weighting}")


def _dag_weights(d, weighting, dag):
    """Weights for a subgoal DAG {subgoal: [prerequisites]}. `d` is (G, B)."""
    G, B = d.shape
    prereqs_done = jnp.stack([
        d[jnp.asarray(dag.get(g, []), dtype=jnp.int32)].prod(axis=0) if dag.get(g) else jnp.ones(B)
        for g in range(G)
    ])                                                           # (G, B)
    current = prereqs_done * (1.0 - d)                           # available, incomplete subgoals

    if weighting == 'current_myopic':
        return current.at[-1].set(1.0)
    if weighting == 'current_future':
        return (1.0 - d).at[-1].set(1.0)
    if weighting == 'next_future':
        return (1.0 - prereqs_done).at[-1].set(1.0)
    if weighting == 'next_myopic':
        has_child = np.zeros(G, dtype=bool)
        for g, parents in dag.items():
            for p in parents:
                has_child[p] = True
        children_of_current = jnp.stack([
            current[jnp.asarray(dag.get(g, []), dtype=jnp.int32)].max(axis=0) if dag.get(g) else jnp.zeros(B)
            for g in range(G)
        ])
        terminal = jnp.asarray(~has_child, dtype=jnp.float32)[:, None]
        w = jnp.maximum(children_of_current, current * terminal)
        all_done = (d.min(axis=0) > 0)
        return jnp.where(all_done[None], jnp.zeros((G, B)).at[-1].set(1.0), w)
    raise ValueError(f"Unsupported DAG weighting: {weighting}")


def aggregate_subgoal_values(q, weights):
    """Q_agg = sum_g w[g] Q(., ., g).

    Args:
        q: (G, ..., B) per-subgoal values, e.g. (G, M, B) or (G, B).
        weights: (G, B).

    Returns:
        (..., B).
    """
    w = weights.reshape(weights.shape[:1] + (1,) * (q.ndim - 2) + weights.shape[1:])
    return (q * w).sum(axis=0)


def subgoal_discount_factors(config, num_subgoals):
    """Per-subgoal discount gamma_g (G,), or the scalar discount if none are given."""
    discounts = config.get('subgoal_discounts')
    if discounts is None:
        return config['discount']
    discounts = jnp.asarray(discounts, dtype=jnp.float32)
    assert discounts.shape == (num_subgoals,), (
        f"subgoal_discounts has length {discounts.shape[0]}, expected {num_subgoals}")
    return discounts


def compute_vadar_td_target(
    dones,
    next_q,
    masks,
    discount,
    horizon_length,
    td_target='gated',
    r_done=5.0,
    r_step=-1.0,
    clip_value=15.0,
    head_types: Optional[Sequence[str]] = None,
    cost_scale=5.0,
):
    """All-subgoal TD targets, one per subgoal.

    The per-subgoal reward is r[g] = r_done if subgoal g is done at the end of
    the chunk, else r_step.

    td_target='gated' (the default used for the paper's experiments):
        y[g] = min(r[g] + gamma_g^H * mask * (d[g] * clip_value + (1 - d[g]) * Q'[g]), clip_value)
        and y[g] = clip_value at a terminal step where g is done.
    td_target='clipped' (Eq. 3 of the paper):
        y[g] = min(r[g] + gamma_g^H * mask * Q'[g], clip_value)

    Heads listed as 'cost' in `head_types` (multi-objective safety heads) use
    y[g] = -cost_scale * c[g] + gamma_g^H * mask * Q'[g], where c[g] in [0, 1]
    is stored in the subgoal-done slot g; they are never clipped.

    Args:
        dones: (B, G) completion indicators at the last step of the chunk.
        next_q: (G, B) ensemble-aggregated next values.
        masks: (B,) non-terminal masks at the last step of the chunk.
        discount: scalar or (G,) per-subgoal discounts.

    Returns:
        (G, B) targets.
    """
    d = jnp.asarray(dones, dtype=jnp.float32).T  # (G, B)
    disc = jnp.asarray(discount, dtype=jnp.float32) ** horizon_length
    if disc.ndim > 0:
        disc = disc[:, None]
    rewards = jnp.where(d > 0, r_done, r_step)

    if td_target == 'clipped':
        target = jnp.minimum(rewards + disc * masks * next_q, clip_value)
    elif td_target == 'gated':
        bootstrap = d * clip_value + (1.0 - d) * next_q
        target = jnp.minimum(rewards + disc * masks * bootstrap, clip_value)
        terminal_and_done = ((1.0 - masks) > 0)[None, :] & (d > 0)
        target = jnp.where(terminal_and_done, clip_value, target)
    else:
        raise ValueError(f"Unknown td_target {td_target!r}; expected 'gated' or 'clipped'")

    if head_types is not None and 'cost' in head_types:
        is_cost = jnp.asarray([t == 'cost' for t in head_types])[:, None]
        cost_target = -cost_scale * d + disc * masks * next_q
        target = jnp.where(is_cost, cost_target, target)
    return target


def shaped_reward(dones, r_done=5.0, r_step=-1.0):
    """VaDaR-Shaped reward: the per-subgoal sparse rewards summed into one scalar.

    Args:
        dones: (..., G) completion indicators after the transition.

    Returns:
        (...,) scalar reward sum_g r[g].
    """
    d = np.asarray(dones, dtype=np.float32)
    return np.where(d > 0, r_done, r_step).sum(axis=-1)
