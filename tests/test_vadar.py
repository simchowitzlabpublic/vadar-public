"""CPU unit tests for VaDaR: weightings, targets, critics and agent updates.

Run with `pytest tests/test_vadar.py` (no simulators needed).
"""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vadar.agents.modules.vadar_helper import (
    subgoal_weights, compute_vadar_td_target, aggregate_subgoal_values, shaped_reward,
)

TRANSPORT_DAG = {0: [], 1: [0], 2: [1], 3: [], 4: [3], 5: [2, 4], 6: [5]}


# ---------------------------------------------------------------------------
# Weightings
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dones,expected", [
    # current subgoal is index 1 (subgoal 0 done)
    ([1, 0, 0, 0], {'next_future': [0, 0, 1, 1], 'current_future': [0, 1, 1, 1],
                    'current_myopic': [0, 1, 0, 0], 'next_myopic': [0, 0, 1, 0], 'last': [0, 0, 0, 1]}),
    # nothing done
    ([0, 0, 0, 0], {'next_future': [0, 1, 1, 1], 'current_future': [1, 1, 1, 1],
                    'current_myopic': [1, 0, 0, 0], 'next_myopic': [0, 1, 0, 0], 'last': [0, 0, 0, 1]}),
    # current subgoal is the last one
    ([1, 1, 1, 0], {'next_future': [0, 0, 0, 1], 'current_future': [0, 0, 0, 1],
                    'current_myopic': [0, 0, 0, 1], 'next_myopic': [0, 0, 0, 1], 'last': [0, 0, 0, 1]}),
])
def test_sequential_weightings(dones, expected):
    d = np.asarray([dones], dtype=np.float32)
    for name, w in expected.items():
        np.testing.assert_allclose(np.asarray(subgoal_weights(d, name))[:, 0], w, err_msg=name)


def test_weighting_aliases():
    d = np.random.default_rng(0).integers(0, 2, (16, 5)).astype(np.float32)
    for alias, name in [('NF', 'next_future'), ('CF', 'current_future'), ('GM', 'current_myopic'), ('lastq', 'last')]:
        np.testing.assert_allclose(subgoal_weights(d, alias), subgoal_weights(d, name))


def test_dag_next_future_transport():
    # lid removed and trash grasped: current = {payload_grasped, trash_in_bin}
    d = np.asarray([[1, 0, 0, 1, 0, 0, 0]], dtype=np.float32)
    w = np.asarray(subgoal_weights(d, 'next_future', TRANSPORT_DAG))[:, 0]
    np.testing.assert_allclose(w, [0, 0, 1, 0, 0, 1, 1])
    w = np.asarray(subgoal_weights(d, 'next_myopic', TRANSPORT_DAG))[:, 0]
    np.testing.assert_allclose(w, [0, 0, 1, 0, 0, 1, 0])


def test_aggregate_subgoal_values():
    q = jnp.arange(2 * 3 * 4, dtype=jnp.float32).reshape(2, 3, 4)  # (G, M, B)
    w = jnp.asarray([[1, 0, 1, 0], [1, 1, 0, 0]], dtype=jnp.float32)  # (G, B)
    out = aggregate_subgoal_values(q, w)
    np.testing.assert_allclose(out, (q * w[:, None, :]).sum(0))


# ---------------------------------------------------------------------------
# TD targets
# ---------------------------------------------------------------------------

def test_gated_target_done_subgoal_is_clip_value():
    dones = np.asarray([[1, 0]], dtype=np.float32)
    next_q = jnp.asarray([[3.0], [-4.0]])
    y = compute_vadar_td_target(dones, next_q, masks=jnp.ones(1), discount=0.99, horizon_length=4)
    np.testing.assert_allclose(y[0, 0], 15.0)
    np.testing.assert_allclose(y[1, 0], -1.0 + 0.99 ** 4 * -4.0, rtol=1e-6)


def test_clipped_target():
    dones = np.asarray([[1, 0]], dtype=np.float32)
    next_q = jnp.asarray([[14.0], [-4.0]])
    y = compute_vadar_td_target(dones, next_q, masks=jnp.ones(1), discount=0.99, horizon_length=1, td_target='clipped')
    np.testing.assert_allclose(y[:, 0], [15.0, -1.0 + 0.99 * -4.0], rtol=1e-6)


def test_cost_head_target():
    dones = np.asarray([[1.0, 0.4]], dtype=np.float32)  # task done, cost 0.4
    next_q = jnp.asarray([[2.0], [-3.0]])
    y = compute_vadar_td_target(dones, next_q, masks=jnp.ones(1), discount=0.99, horizon_length=4,
                                head_types=['subgoal', 'cost'])
    np.testing.assert_allclose(y[1, 0], -5.0 * 0.4 + 0.99 ** 4 * -3.0, rtol=1e-6)


def test_shaped_reward():
    np.testing.assert_allclose(shaped_reward(np.asarray([[1, 0, 0], [1, 1, 1]])), [3.0, 15.0])


# ---------------------------------------------------------------------------
# Critics
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind", ['goal_conditioned', 'vector', 'separate'])
def test_vadar_critic_shapes(kind):
    from vadar.networks.vadar_critics import VADAR_CRITICS
    critic = VADAR_CRITICS[kind](hidden_dims=(32, 32), num_ensembles=3, num_subgoals=4)
    obs, act = jnp.zeros((5, 7)), jnp.zeros((5, 2))
    params = critic.init(jax.random.PRNGKey(0), obs, actions=act)
    assert critic.apply(params, obs, actions=act).shape == (4, 3, 5)


def test_miq_vadar_critic_shapes():
    from vadar.networks.critics import ValueMIPEnsemble
    from vadar.agents.modules.q_helper import sample_mip_q_ensemble_values, compute_mip_q_ensemble_predictions
    critic = ValueMIPEnsemble(hidden_dims=(32, 32), num_ensembles=3, num_subgoals=4)
    obs, act = jnp.zeros((5, 7)), jnp.zeros((5, 2))
    params = critic.init(jax.random.PRNGKey(0), obs, actions=act, rng=jax.random.PRNGKey(1))
    fn = lambda o, actions, scalar_input, time, is_encoded=False, return_logits=False: critic.apply(
        params, o, actions=actions, scalar_input=scalar_input, time=time)
    q = sample_mip_q_ensemble_values(fn, obs, act, jnp.zeros((5, 4)), 0.9)
    assert q.shape == (4, 3, 5)
    q0, q1 = compute_mip_q_ensemble_predictions(fn, obs, act, jnp.zeros((4, 5)), jax.random.PRNGKey(2))
    assert q0.shape == q1.shape == (4, 3, 5)
