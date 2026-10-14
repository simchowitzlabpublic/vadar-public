"""CPU smoke tests: build each agent from its real config and run one update.

Uses tiny networks and random batches (no simulator). Checks that the sparse
agents and their VaDaR twins run the BC, critic-only and joint updates, and
that their rollout methods return actions of the right shape.
"""

import jax
import numpy as np
import pytest

from vadar.utils.config_loader import load_config, config_to_agent_dict

OBS_DIM, ACT_DIM, H, B, G = 11, 3, 4, 8, 4


def _agent_config(algo, overrides):
    config = load_config(algo, {
        'agent.actor_hidden_dims': [32, 32],
        'agent.value_hidden_dims': [32, 32],
        'agent.one_step_hidden_dims': [32, 32],
        'agent.num_qs': 3,
        'agent.batch_size': B,
        'agent.ppo_batch_size': B,
        'agent.group_num_samples': 4,
        'agent.best_of_n': 3,
        'agent.horizon_length': H,
        'agent.flow_steps': 3,
        **overrides,
    })
    agent_config = config_to_agent_dict(config)
    agent_config['q_min'], agent_config['q_max'] = -100.0, 15.0
    if algo.endswith('_vadar'):
        agent_config.update(num_subgoals=G, subgoal_discounts=[0.9, 0.99, 0.995, 0.999])
    return agent_config


def _batch(rng, utd=1):
    batch = dict(
        observations=rng.normal(size=(utd, B, OBS_DIM)).astype(np.float32),
        next_observations=rng.normal(size=(utd, B, OBS_DIM)).astype(np.float32),
        actions=rng.uniform(-1, 1, size=(utd, B, H, ACT_DIM)).astype(np.float32),
        rewards=-np.ones((utd, B, H), np.float32),
        masks=np.ones((utd, B, H), np.float32),
        terminals=np.zeros((utd, B, H), np.float32),
        valid=np.ones((utd, B, H), np.float32),
        mc_returns=-np.ones((utd, B), np.float32),
        subgoal_dones=np.sort(rng.integers(0, 2, size=(utd, B, H, G)), axis=-1)[..., ::-1].astype(np.float32),
    )
    return batch


def _agent_class(algo):
    from vadar.agents.ogpo import OGPOAgent
    from vadar.agents.ogpo_vadar import OGPOVadarAgent
    return OGPOVadarAgent if algo.endswith('_vadar') else OGPOAgent


def _make(algo, overrides):
    rng = np.random.default_rng(0)
    cfg = _agent_config(algo, overrides)
    agent = _agent_class(algo).create(0, np.zeros(OBS_DIM, np.float32), np.zeros(ACT_DIM, np.float32), cfg,
                                cfg.get('adv_clip_min'))
    return agent, rng


@pytest.mark.parametrize("algo,overrides", [
    ('ogpo', {}),
    ('ogpo_vadar', {}),
    ('ogpo_vadar', {'agent.vadar_critic': 'vector', 'agent.weighting': 'CF'}),
    ('ogpo_vadar', {'agent.vadar_critic': 'separate', 'agent.adv_strategy': 'conservative'}),
    ('ogpo_vadar', {'agent.use_mip_q': True, 'agent.mip_q_ensemble': True}),
    ('ogpo_vadar', {'agent.td_target': 'clipped', 'agent.q_agg': 'subsample'}),
    ('fql', {}),
    ('fql_vadar', {}),
])
def test_ogpo_family_updates(algo, overrides):
    agent, rng = _make(algo, overrides)
    batch = _batch(rng)
    flat = jax.tree.map(lambda x: x[0], batch)
    agent, info = agent.bc_update(flat)
    agent = agent.reset_optimizers_with_lr()
    agent, info = agent.batch_q_warmup_update(batch)
    assert np.isfinite(float(info['critic/td_loss']))
    agent, info = agent.batch_update(batch, batch, True)
    assert all(np.all(np.isfinite(np.asarray(v))) for v in info.values())

    obs = rng.normal(size=(2, OBS_DIM)).astype(np.float32)
    key = jax.random.PRNGKey(1)
    assert agent.sample_actions(observations=obs, rng=key).shape == (2, H * ACT_DIM)
    assert agent.sample_actions_BON(observations=obs, rng=key).shape == (2, H * ACT_DIM)
    if agent.one_step_network is not None:
        assert agent.sample_actions_one_step_BON(observations=obs, rng=key).shape == (2, H * ACT_DIM)
    assert agent.compute_flow_actions_corrected(obs, rng=key).shape == (2, H * ACT_DIM)


@pytest.mark.parametrize("algo,overrides", [
    ('sac', {}),
    ('sac_vadar', {}),
    ('sac_vadar', {'agent.vadar_critic': 'vector', 'agent.weighting': 'last'}),
    ('sac_vadar', {'agent.use_mip_q': True, 'agent.mip_q_ensemble': True}),
])
def test_sac_family_updates(algo, overrides):
    from vadar.agents.sac import SACAgent
    from vadar.agents.sac_vadar import SACVadarAgent
    rng = np.random.default_rng(0)
    cfg = _agent_config(algo, overrides)
    cls = SACVadarAgent if algo.endswith('_vadar') else SACAgent
    agent = cls.create(0, np.zeros(OBS_DIM, np.float32), np.zeros(ACT_DIM, np.float32), cfg)
    batch = _batch(rng)
    agent, info = agent.bc_update(jax.tree.map(lambda x: x[0], batch))
    agent, info = agent.batch_q_warmup_update(batch)
    agent, info = agent.batch_update(batch, batch, False)
    assert all(np.all(np.isfinite(np.asarray(v))) for v in info.values())
    obs = rng.normal(size=(2, OBS_DIM)).astype(np.float32)
    assert agent.sample_actions(observations=obs, rng=jax.random.PRNGKey(1)).shape == (2, H * ACT_DIM)
    assert agent.compute_flow_actions(obs).shape == (2, H * ACT_DIM)


def test_sac_vadar_multiobjective_heads():
    """PushT-Obs: task head + cost head with constant weights (1, lambda)."""
    from vadar.agents.sac_vadar import SACVadarAgent
    rng = np.random.default_rng(0)
    cfg = _agent_config('sac_vadar', {'agent.vadar_critic': 'vector', 'agent.weighting': 'constant'})
    cfg.update(num_subgoals=2, subgoal_weights=[1.0, 5.0], subgoal_head_types=['subgoal', 'cost'],
               subgoal_discounts=None)
    agent = SACVadarAgent.create(0, np.zeros(OBS_DIM, np.float32), np.zeros(ACT_DIM, np.float32), cfg)
    batch = _batch(rng)
    batch['subgoal_dones'] = rng.uniform(size=(1, B, H, 2)).astype(np.float32)
    agent, info = agent.batch_update(batch, batch, False)
    assert all(np.all(np.isfinite(np.asarray(v))) for v in info.values())
