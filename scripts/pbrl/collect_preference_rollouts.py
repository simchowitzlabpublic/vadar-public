"""Collect policy rollouts to pretrain the preference reward model (Dense baseline).

The paper's Dense baseline pretrains its reward model on the offline demos plus
100 rollouts from each of three checkpoints of a run (pi_early, pi_mid,
pi_late), so the preferences see a range of failures. This script writes one
flat HDF5 per checkpoint:

    observations, next_observations  (T, obs_dim) float32
    actions                          (T, action_dim) float32
    rewards                          (T,) float32   env reward
    terminals                        (T,) bool      env termination (= success for the paper envs)
    is_success                       (T,) float32   info['success'] at each step
    subgoal_dones                    (T, G) float32 only with --subgoals (needs env subgoal tracking)

Point `reward_model.rollout_dataset_path` at one file or the output directory.

Ported from OGPO-pbrl-baseline scripts/rollout/collect_robocasa_rollouts_parallel.py
(vectorized envs, tolerant checkpoint restore, atomic partial saves); added
is_success / subgoal_dones and agent dispatch for the VaDaR repo.

Example:
    python scripts/pbrl/collect_preference_rollouts.py \
        --run_dir /data/user_data/$USER/ogpo/<project>/<group>/<env>/<run> \
        --ckpt_steps 300000 1300000 3000000 --tiers early mid late \
        --env_name robocasa_PreSoakPan --num_rollouts 100 --out_dir preference_data/presoakpan
"""

import argparse
import datetime
import inspect
import json
import os
import pickle
import sys
import time

import flax
import h5py
import jax
import numpy as np
import tqdm

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from envs.env_utils import make_env_and_datasets  # noqa: E402
from vadar.utils.config_loader import config_to_agent_dict  # noqa: E402
from vadar.utils.main_helper import get_q_bounds, make_vectorized_env, process_train_dataset  # noqa: E402

# Algorithms whose create() takes (seed, obs, act, config, adv_clip_min, ex_images=...).
_FLOW_ALGOS = {'ogpo', 'fql', 'ogpo_vadar', 'fql_vadar', 'reinflow_calql'}


def _agent_class(algo: str):
    try:
        from vadar.main import AGENTS
        if algo in AGENTS:
            return AGENTS[algo]
    except Exception as exc:  # main.py may import optional baselines
        print(f"[collect] could not import vadar.main.AGENTS ({exc}); using the built-in table")
    if algo in ('ogpo', 'fql'):
        from vadar.agents.ogpo import OGPOAgent
        return OGPOAgent
    raise ValueError(f"No agent class for algo {algo!r}")


def _tolerant_restore(agent, ckpt_path):
    """Restore params_<step>.pkl, keeping live entries the checkpoint lacks."""
    with open(ckpt_path, 'rb') as f:
        saved = pickle.load(f)['agent']
    live = flax.serialization.to_state_dict(agent)

    def _merge(live_d, saved_d):
        if not isinstance(live_d, dict) or not isinstance(saved_d, dict):
            return saved_d if saved_d is not None else live_d
        out = dict(live_d)
        for k, v in saved_d.items():
            out[k] = _merge(out[k], v) if k in out else v
        return out

    print(f"Restored {ckpt_path}")
    return flax.serialization.from_state_dict(agent, _merge(live, saved))


def _supply_rng(f, rng):
    def wrapped(*args, **kwargs):
        nonlocal rng
        rng, key = jax.random.split(rng)
        return f(*args, rng=key, **kwargs)
    return wrapped


def _info_array(infos, key, num_envs, width=None):
    """(num_envs,) or (num_envs, width) array of `key` from vector-env infos (zeros if absent)."""
    shape = (num_envs,) if width is None else (num_envs, width)
    out = np.zeros(shape, dtype=np.float32)
    if isinstance(infos, dict) and key in infos:
        v = np.asarray(infos[key], dtype=np.float32)
        out[: v.shape[0]] = v.reshape((v.shape[0],) + shape[1:])
    elif isinstance(infos, (list, tuple)):
        for i, info in enumerate(infos):
            if isinstance(info, dict) and key in info:
                out[i] = np.asarray(info[key], dtype=np.float32)
    return out


def build_agent_and_envs(run_dir, env_name, seed, num_envs, algo_override=None, subgoals=False):
    with open(os.path.join(run_dir, 'flags.json')) as f:
        flags = json.load(f)
    config = {'env': dict(flags.get('env', {})), 'agent': dict(flags.get('agent', {}))}
    config['env']['name'] = env_name
    agent_config = config_to_agent_dict(config)
    algo = algo_override or str(agent_config.get('agent_name', 'ogpo')).lower()
    rew_fn = config['env'].get('rew_fn', 'sparse')
    post_success_steps = int(config['env'].get('post_success_steps', 0) or 0)

    subgoal_cfg = None
    if subgoals:
        from envs.subgoals import get_subgoal_spec
        spec = get_subgoal_spec(env_name)
        subgoal_cfg = dict(num_subgoals=spec['num_subgoals'], dag=spec['dag'], reward_mode='sparse')

    def _call(fn, *args, **kwargs):
        if subgoal_cfg is not None:
            if 'subgoal_cfg' not in inspect.signature(fn).parameters:
                raise RuntimeError(f"{fn.__name__} has no subgoal_cfg argument yet; cannot record subgoal_dones")
            kwargs['subgoal_cfg'] = subgoal_cfg
        return fn(*args, **kwargs)

    _, _, train_dataset, _ = _call(make_env_and_datasets, env_name, rew_fn=rew_fn,
                                   post_success_steps=post_success_steps)
    train_dataset = process_train_dataset(train_dataset, env_name, numpy_rng=np.random.default_rng(seed),
                                          discount=agent_config['discount'])
    example = train_dataset.sample(())
    agent_config['q_min'], agent_config['q_max'] = get_q_bounds(env_name, agent_config['discount'])

    vec_envs = _call(make_vectorized_env, env_name=env_name, num_envs=num_envs, seed=seed, rew_fn=rew_fn,
                     post_success_steps=post_success_steps)

    cls = _agent_class(algo)
    if algo in _FLOW_ALGOS:
        agent = cls.create(seed, example['observations'], example['actions'], agent_config,
                           agent_config['adv_clip_min'], ex_images=example.get('images'))
    else:
        agent = cls.create(seed, example['observations'], example['actions'], agent_config)
    G = subgoal_cfg['num_subgoals'] if subgoal_cfg else 0
    return agent, vec_envs, agent_config, G


def _save_h5(path, episodes, attrs):
    flat = {k: np.concatenate([ep[k] for ep in episodes], axis=0) for k in episodes[0]}
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    tmp = path + '.tmp'
    with h5py.File(tmp, 'w') as f:
        for k, v in flat.items():
            f.create_dataset(k, data=v, compression='gzip', compression_opts=4)
        for k, v in attrs.items():
            f.attrs[k] = v
    os.replace(tmp, path)


def rollout_checkpoint(agent, vec_envs, ckpt_path, num_rollouts, action_dim, num_envs, action_chunking,
                       base_seed, deterministic, out_path, ckpt_step, save_every, G):
    agent = _tolerant_restore(agent, ckpt_path)
    if deterministic and hasattr(agent, 'compute_flow_actions'):
        actor_fn = agent.compute_flow_actions
    else:
        actor_fn = _supply_rng(agent.sample_actions, jax.random.PRNGKey(base_seed))

    episodes, done_count, batch_idx = [], 0, 0
    pbar = tqdm.tqdm(total=num_rollouts, desc=f"step{ckpt_step}", dynamic_ncols=True)
    while done_count < num_rollouts:
        raw_obs, _ = vec_envs.reset(seed=base_seed + 1000 * (batch_idx + 1))
        obs = np.asarray(raw_obs, dtype=np.float32)
        buf = [{k: [] for k in ('observations', 'next_observations', 'actions', 'rewards', 'terminals',
                                'is_success') + (('subgoal_dones',) if G else ())} for _ in range(num_envs)]
        queues = [[] for _ in range(num_envs)]
        active = np.ones(num_envs, dtype=bool)
        while active.any():
            need = [i for i in range(num_envs) if active[i] and not queues[i]]
            if need:
                o = obs[need]
                if len(need) < num_envs:  # pad to a fixed batch to avoid re-jitting
                    o = np.concatenate([o, np.zeros((num_envs - len(need),) + o.shape[1:], o.dtype)])
                chunks = np.asarray(actor_fn(observations=o)).reshape(num_envs, -1, action_dim)[:len(need)]
                for k, i in enumerate(need):
                    queues[i].extend(chunks[k] if action_chunking else chunks[k][:1])
            acts = np.zeros((num_envs, action_dim), dtype=np.float32)
            for i in range(num_envs):
                if active[i]:
                    acts[i] = queues[i].pop(0)
            acts = np.clip(acts, -1.0, 1.0)
            next_obs, rew, term, trunc, infos = vec_envs.step(acts)
            next_obs = np.asarray(next_obs, dtype=np.float32)
            succ = _info_array(infos, 'success', num_envs)
            sub = _info_array(infos, 'subgoal_dones', num_envs, G) if G else None
            for i in range(num_envs):
                if not active[i]:
                    continue
                b = buf[i]
                b['observations'].append(obs[i].copy())
                b['next_observations'].append(next_obs[i].copy())
                b['actions'].append(acts[i].copy())
                b['rewards'].append(np.float32(rew[i]))
                b['terminals'].append(bool(term[i]))
                b['is_success'].append(np.float32(succ[i]))
                if G:
                    b['subgoal_dones'].append(sub[i].copy())
                if term[i] or trunc[i]:
                    active[i] = False
            obs = next_obs

        n_commit = min(num_envs, num_rollouts - done_count)
        for i in range(n_commit):
            episodes.append({k: np.asarray(v) for k, v in buf[i].items()})
        done_count += n_commit
        pbar.update(n_commit)
        batch_idx += 1
        if done_count >= num_rollouts or batch_idx % save_every == 0:
            n_succ = sum(int(ep['is_success'].any()) for ep in episodes)
            _save_h5(out_path, episodes, dict(ckpt_step=ckpt_step, ckpt_path=ckpt_path,
                                              success_rate=n_succ / len(episodes), num_rollouts=len(episodes),
                                              num_rollouts_target=num_rollouts,
                                              partial=bool(len(episodes) < num_rollouts)))
    pbar.close()
    n_succ = sum(int(ep['is_success'].any()) for ep in episodes)
    return {'success_rate': n_succ / len(episodes), 'mean_length': float(np.mean([len(ep['rewards']) for ep in episodes])),
            'num_rollouts': len(episodes)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run_dir', required=True, help='run dir holding flags.json and params_<step>.pkl')
    p.add_argument('--ckpt_steps', type=int, nargs='+', required=True)
    p.add_argument('--tiers', nargs='+', default=None, help='names per checkpoint, e.g. early mid late')
    p.add_argument('--env_name', required=True)
    p.add_argument('--algo', default=None, help="override flags.json agent.agent_name")
    p.add_argument('--num_envs', type=int, default=32)
    p.add_argument('--num_rollouts', type=int, default=100)
    p.add_argument('--seed', type=int, default=12345)
    p.add_argument('--out_dir', required=True)
    p.add_argument('--deterministic', action='store_true', help='ODE actions (default: stochastic sample_actions)')
    p.add_argument('--save_every', type=int, default=1, help='atomic partial save every N env batches')
    p.add_argument('--subgoals', action='store_true', help='also record info[subgoal_dones] (Pref-Subgoal rollouts)')
    args = p.parse_args()

    tiers = args.tiers or [f'step{s}' for s in args.ckpt_steps]
    if len(tiers) != len(args.ckpt_steps):
        raise ValueError('--tiers must match --ckpt_steps')
    ckpts = []
    for step, tier in zip(args.ckpt_steps, tiers):
        path = os.path.join(args.run_dir, f'params_{step}.pkl')
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        ckpts.append((step, tier, path))

    agent, vec_envs, agent_config, G = build_agent_and_envs(args.run_dir, args.env_name, args.seed, args.num_envs,
                                                            algo_override=args.algo, subgoals=args.subgoals)
    action_dim = vec_envs.single_action_space.shape[0]
    action_chunking = bool(agent_config.get('action_chunking', True))
    stamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
    t0, summary = time.time(), {}
    for step, tier, path in ckpts:
        out_path = os.path.join(args.out_dir, f'{args.env_name}-{tier}-{stamp}.h5')
        summary[tier] = rollout_checkpoint(agent, vec_envs, path, args.num_rollouts, action_dim, args.num_envs,
                                           action_chunking, args.seed, args.deterministic, out_path, step,
                                           args.save_every, G)
        print(f">>> {tier} (step {step}): {summary[tier]} -> {out_path}")
    print(json.dumps(summary, indent=2))
    print(f"Wall time {time.time() - t0:.0f}s")
    try:
        vec_envs.close()
    except Exception:
        pass


if __name__ == '__main__':
    main()
