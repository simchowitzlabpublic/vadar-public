"""Tests for the preference-reward baselines (vadar/rewards, vadar/runners/online_pbrl_runner.py).

numpy-only tests run anywhere; the JAX tests are skipped when jax is missing.
Run:  python -m pytest tests/test_pbrl_rewards.py
"""
import os
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from vadar.rewards.preference_data import (  # noqa: E402
    build_episode_store,
    compute_first_achievement_steps,
    concat_stores,
    empty_episode_store,
    episode_boundaries,
    generate_preference_pairs,
    generate_preference_pairs_subgoal,
    single_episode_entry,
    split_episode_store,
    truncate_fifo,
)
from vadar.rewards.subgoal_preference_reward import (  # noqa: E402
    aggregate_subgoal_rewards,
    canonical_weighting,
    subgoal_weights_np,
)


def _make_dataset(rng, n_eps=40, G=3, obs_dim=6):
    """Concatenated episodes with cumulative subgoal dones; success = last subgoal done."""
    obs, nobs, dones = [], [], []
    for _ in range(n_eps):
        L = int(rng.integers(10, 16))
        o = (rng.normal(size=(1, obs_dim)) * 20 + np.cumsum(rng.normal(size=(L, obs_dim)), 0) * 0.01).astype(np.float32)
        d = np.zeros((L, G), np.float32)
        for g in range(int(rng.integers(0, G + 1))):
            d[int(rng.integers(1, L)):, g] = 1
        d = np.minimum.accumulate(d, axis=1)  # sequential: g done => g-1 done
        obs.append(o)
        nobs.append(np.concatenate([o[1:], o[-1:] + 50]))
        dones.append(d)
    d = np.concatenate(dones)
    return dict(observations=np.concatenate(obs), next_observations=np.concatenate(nobs),
                rewards=np.where(d[:, -1] > 0, 0.0, -1.0).astype(np.float32),
                terminals=d[:, -1].copy(), is_success=d[:, -1].copy(), subgoal_dones=d), n_eps


# ---------------------------------------------------------------------------
# numpy
# ---------------------------------------------------------------------------

def test_episode_store_and_boundaries():
    rng = np.random.default_rng(0)
    ds, n_eps = _make_dataset(rng)
    starts, ends = episode_boundaries(ds['observations'], ds['next_observations'])
    assert starts.shape[0] == n_eps
    store = build_episode_store(ds['observations'], ds['next_observations'], ds['rewards'], ds['is_success'],
                                t_max_override=40, subgoal_dones_per_step=ds['subgoal_dones'])
    assert store['obs'].shape[:2] == (n_eps, 40)
    for i in range(n_eps):
        L = int(store['lengths'][i])
        assert (store['subgoal_success'][i] == (store['per_step_subgoal_dones'][i, :L] > 0).any(0)).all()
        assert store['obs'][i, L:].sum() == 0
    tr, va = split_episode_store(store, 0.1, np.random.default_rng(0))
    assert tr['lengths'].shape[0] + va['lengths'].shape[0] == n_eps


def test_success_only_labels_drop_ties():
    store = {'lengths': np.ones(4, np.int32), 'returns': np.array([0, 5, 1, 2], np.float32),
             'is_success': np.array([True, False, True, False])}
    buf = generate_preference_pairs(store, 0, np.random.default_rng(0), all_pairs=True, label_mode='success_only')
    succ = store['is_success']
    assert len(buf['labels']) == 4  # 2 successes x 2 failures
    for i, j, y in zip(buf['ep_idx_1'], buf['ep_idx_2'], buf['labels']):
        assert succ[i] != succ[j]
        assert y == (1.0 if succ[j] else 0.0)
    buf = generate_preference_pairs(store, 0, np.random.default_rng(0), all_pairs=True, label_mode='success_return')
    for i, j, y in zip(buf['ep_idx_1'], buf['ep_idx_2'], buf['labels']):
        if succ[i] == succ[j]:
            assert y == float(store['returns'][j] > store['returns'][i])


def test_subgoal_success_time_labels():
    rng = np.random.default_rng(1)
    ds, _ = _make_dataset(rng)
    store = build_episode_store(ds['observations'], ds['next_observations'], ds['rewards'], ds['is_success'],
                                subgoal_dones_per_step=ds['subgoal_dones'])
    first = compute_first_achievement_steps(store)
    buf = generate_preference_pairs_subgoal(store, 0, rng, all_pairs=True, subgoal_label_mode='success_time')
    assert len(buf['labels']) > 0
    for i, j, g, y in zip(buf['ep_idx_1'], buf['ep_idx_2'], buf['subgoal_idx'], buf['labels']):
        assert first[i, g] != first[j, g]
        assert y == float(first[j, g] < first[i, g])  # earlier (or only) completion wins
    buf = generate_preference_pairs_subgoal(store, 0, rng, all_pairs=True, subgoal_label_mode='success_only')
    for i, j, g, y in zip(buf['ep_idx_1'], buf['ep_idx_2'], buf['subgoal_idx'], buf['labels']):
        si, sj = store['subgoal_success'][i, g], store['subgoal_success'][j, g]
        assert si != sj and y == float(sj)


def test_online_entry_concats_and_fifo():
    rng = np.random.default_rng(2)
    off = empty_episode_store(20, 4, num_subgoals=2)
    e = single_episode_entry(rng.normal(size=(7, 4)).astype(np.float32), np.zeros(7), True, 20,
                             subgoal_dones=np.tile([1.0, 0.0], (7, 1)).astype(np.float32))
    s = concat_stores(concat_stores(off, e), e)
    assert s['obs'].shape == (2, 20, 4) and s['per_step_success'][0, 6] and not s['per_step_success'][0, 5]
    assert s['subgoal_success'][0].tolist() == [True, False]
    buf = {'labels': np.arange(10.0), 'ep_idx_1': np.arange(10), 'ep_idx_2': np.arange(10)}
    assert truncate_fifo(buf, 4)['labels'].tolist() == [6.0, 7.0, 8.0, 9.0]


def _fork_compute_gate_weights_np(subtask_status, gating_mode, dag=None):
    """OGPO-pbrl-subgoal pbrl_helper.compute_gate_weights_np (reference copy)."""
    done_t = np.asarray(subtask_status, dtype=np.float32).T
    S, B = done_t.shape
    if gating_mode == 'last_only':
        gate = np.zeros_like(done_t)
        gate[-1, :] = 1.0
        return gate
    if dag is not None:
        apd = np.stack([np.prod(np.stack([done_t[p] for p in dag.get(i, [])]), axis=0) if dag.get(i, [])
                        else np.ones(B, np.float32) for i in range(S)])
        gate = {'goal_myopic': apd * (1.0 - done_t), 'goal_future': 1.0 - done_t,
                'grandparent_future': 1.0 - apd}[gating_mode]
        gate[-1, :] = 1.0
        return gate
    active = np.eye(S, dtype=np.float32)[np.argmin(done_t, axis=0)].T
    parent_done = np.concatenate([np.ones((1, B), np.float32), done_t[:-1]], axis=0)
    if gating_mode == 'goal_myopic':
        return active * (1.0 - done_t)
    gate = 1.0 - done_t if gating_mode == 'goal_future' else 1.0 - parent_done
    gate[-1, :] = 1.0
    return gate


@pytest.mark.parametrize('fork_name', ['goal_myopic', 'goal_future', 'grandparent_future', 'last_only'])
def test_weights_match_fork_gates(fork_name):
    rng = np.random.default_rng(3)
    for G in (3, 5, 7):
        raw = rng.integers(0, 2, size=(16, G)).astype(np.float32)
        mono = np.sort(raw, axis=1)[:, ::-1].copy()
        for dones in (raw, mono):
            assert np.allclose(subgoal_weights_np(dones, fork_name), _fork_compute_gate_weights_np(dones, fork_name))
            if fork_name != 'last_only':
                dag = {g: list(range(g)) for g in range(G)}
                assert np.allclose(subgoal_weights_np(dones, fork_name, dag=dag),
                                   _fork_compute_gate_weights_np(dones, fork_name, dag=dag))


def test_aggregate_next_future_and_bonus():
    assert canonical_weighting('NF') == canonical_weighting('grandparent_future') == 'next_future'
    r = np.array([[1.0], [2.0], [4.0], [8.0]], np.float32)       # (G=4, B=1)
    d = np.array([[1, 0, 0, 0]], np.float32)                      # current = 1, next = 2
    assert aggregate_subgoal_rewards(r, d, 'next_future')[0] == 4.0 + 8.0
    assert aggregate_subgoal_rewards(r, d, 'current_future')[0] == 2.0 + 4.0 + 8.0
    assert aggregate_subgoal_rewards(r, d, 'current_myopic')[0] == 2.0
    assert aggregate_subgoal_rewards(r, d, 'next_myopic')[0] == 4.0
    assert aggregate_subgoal_rewards(r, d, 'next_future', done_bonus=2.0)[0] == 12.0 + 2.0


def test_config_resolution():
    pytest.importorskip('yaml')
    from vadar.rewards.config import resolve_reward_model_config
    cfg = resolve_reward_model_config({'reward_model': {'baseline': 'pref_subgoal'}}, 'robocasa_PreSoakPan')
    assert cfg['weighting'] == 'next_future' and cfg['subgoal_done_bonus'] == 0.0
    assert cfg['train_batch_size'] == 16 and cfg['subgoal_label_mode'] == 'success_time'
    cfg = resolve_reward_model_config({'reward_model': {'baseline': 'dense', 'train_batch_size': 8}},
                                      'tool_hang-ph-low_dim')
    assert cfg['label_mode'] == 'success_only' and cfg['train_batch_size'] == 8
    assert cfg['rollout_dataset_path'].endswith('toolhang_rollouts.h5')


# ---------------------------------------------------------------------------
# JAX
# ---------------------------------------------------------------------------

def test_weights_match_vadar_helper():
    pytest.importorskip('jax')
    from vadar.agents.modules.vadar_helper import subgoal_weights
    rng = np.random.default_rng(4)
    for G in (3, 5):
        dones = np.sort(rng.integers(0, 2, size=(32, G)).astype(np.float32), axis=1)[:, ::-1].copy()
        dag = {g: ([] if g == 0 else [g - 1]) for g in range(G)}
        for w in ('next_future', 'current_future', 'current_myopic', 'next_myopic', 'last'):
            for dg in (None, dag):
                assert np.allclose(subgoal_weights_np(dones, w, dg), np.asarray(subgoal_weights(dones, w, dg))), (w, dg)


def _rm_cfg(**kw):
    from vadar.rewards.config import default_reward_model_config, validate_reward_model_config
    cfg = default_reward_model_config()
    cfg.update(dict(ensemble_size=2, hidden_dim=32, n_layers=2, train_epochs=3, train_batch_size=64,
                    min_offline_pairs=10, online_feedback_batch=8, online_large_batch=64))
    cfg.update(kw)
    return validate_reward_model_config(cfg)


@pytest.mark.parametrize('G', [0, 3])
def test_pretrain_online_update_relabel(G):
    pytest.importorskip('jax')
    from vadar.rewards.pebble import (LearnedReward, OnlinePBRLState, append_online_trajectory,
                                      online_reward_update, relabel_replay_buffer)
    from vadar.rewards.preference_reward import load_reward_model, pretrain_reward_model, save_reward_model
    from vadar.rewards.subgoal_preference_reward import SubgoalRewardAggregator

    rng = np.random.default_rng(5)
    ds, _ = _make_dataset(rng, n_eps=60)
    cfg = _rm_cfg(baseline='pref_subgoal' if G else 'dense')
    out = pretrain_reward_model(ds, cfg, seed=0, discount=0.99, num_subgoals=G, env_max_episode_length=20)
    assert out['store_offline']['obs'].shape[1] == 20

    with tempfile.TemporaryDirectory() as d:
        save_reward_model(f'{d}/rm.pkl', out['params'], cfg, goal_dim=G)
        params, apply_fn, gd = load_reward_model(f'{d}/rm.pkl', ds['observations'][:1], seed=0, expected_goal_dim=G)
        assert gd == G

    agg = SubgoalRewardAggregator(G, 'next_future') if G else None
    reward = LearnedReward({'params': out['params']}, out['apply_fn'], num_subgoals=G, aggregator=agg)
    state = OnlinePBRLState(reward=reward, tx=out['tx'], opt_state=out['opt_state'], rm_cfg=cfg,
                            pref_buffer_offline=out['pref_buffer_offline'], store_offline=out['store_offline'],
                            np_rng=np.random.default_rng(0))
    for _ in range(20):
        L = int(rng.integers(5, 12))
        dn = np.zeros((L, max(G, 1)), np.float32)
        if G:
            dn[L // 2:, 0] = 1
        traj = [dict(observations=rng.normal(size=6).astype(np.float32), rewards=0.0,
                     **({'subgoal_dones': dn[t]} if G else {})) for t in range(L)]
        info = {'subgoal_dones': dn[0]} if G else {}
        assert np.isfinite(reward.step_reward(traj[0]['observations'], info))
        append_online_trajectory(state, traj, float(rng.integers(0, 2)))
    metrics = online_reward_update(state)
    assert metrics is not None and state.online_updates == 1

    class _RB:
        size = 50
        _dict = {'observations': rng.normal(size=(64, 6)).astype(np.float32),
                 'rewards': np.full((64,), 7.0, np.float32),
                 'actions': np.ones((64, 2), np.float32)}
        if G:
            _dict['subgoal_dones'] = np.zeros((64, G), np.float32)
    rb = _RB()
    actions_before = rb._dict['actions'].copy()
    assert relabel_replay_buffer(rb, reward, chunk=16) == 50
    assert np.all(rb._dict['rewards'][50:] == 7.0) and not np.any(rb._dict['rewards'][:50] == 7.0)
    assert np.array_equal(rb._dict['actions'], actions_before)
    expect = reward.batch_rewards(rb._dict['observations'][:50],
                                  rb._dict['subgoal_dones'][:50] if G else None)
    assert np.allclose(rb._dict['rewards'][:50], expect, atol=1e-5)
