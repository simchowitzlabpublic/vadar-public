"""One-call setup of the preference-reward baselines for main.py.

    from vadar.rewards.config import pbrl_baseline
    from vadar.rewards.pbrl_setup import prepare_pbrl
    if pbrl_baseline(config):
        pbrl = prepare_pbrl(config, ...)          # before BC, as in the reference
        ...
        run_online_pbrl(**runner_kwargs, **pbrl.runner_kwargs())

`prepare_pbrl` gets a reward model in one of three ways (same order as the reference main.py):
  1. resume:     the rolling checkpoint dir already holds reward_model.pkl (+ pref_buffer.pkl);
  2. pretrained: reward_model.pretrained_checkpoint_path (optionally frozen);
  3. pretrain:   Bradley-Terry on the offline dataset (+ rollout_dataset_path trajectories).
and builds the online PEBBLE state when online updates are enabled.
"""

import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import numpy as np
import optax

from vadar.rewards.config import resolve_reward_model_config
from vadar.rewards.pebble import (
    LearnedReward,
    OnlinePBRLState,
    check_pref_payload,
    load_pref_payload,
    make_online_callbacks,
)
from vadar.rewards.preference_data import (
    concat_transition_arrays,
    dataset_arrays_for_pbrl,
    load_rollout_dataset,
    resolve_env_max_episode_length,
)
from vadar.rewards.preference_reward import (
    PREF_BUFFER_CKPT_NAME,
    REWARD_MODEL_CKPT_NAME,
    load_reward_model,
    pretrain_reward_model,
    save_pickle_atomic,
    save_reward_model,
)
from vadar.rewards.subgoal_preference_reward import SubgoalRewardAggregator, resolve_subgoal_dag


@dataclass
class PBRLContext:
    rm_cfg: Dict[str, Any]
    reward: LearnedReward
    state: Optional[OnlinePBRLState]
    on_episode_end: Optional[Callable]
    on_interval: Optional[Callable]
    interval: Optional[int]
    pretrain_metrics: Dict[str, float]

    def runner_kwargs(self) -> Dict[str, Any]:
        """Extra kwargs of vadar.runners.online_pbrl_runner.run_online_pbrl."""
        return dict(
            pbrl_reward=self.reward,
            pbrl_on_episode_end=self.on_episode_end,
            pbrl_on_interval=self.on_interval,
            pbrl_interval=self.interval,
        )


def _num_subgoals(train_dataset, env_name: str) -> int:
    try:
        G = int(np.asarray(train_dataset['subgoal_dones']).shape[-1])
    except (KeyError, TypeError) as exc:
        raise KeyError("pref_subgoal needs train_dataset['subgoal_dones'] (N, G); build the env/datasets "
                       "with subgoal_cfg (see docs/DESIGN.md)") from exc
    try:
        from envs.subgoals import get_subgoal_spec
        G_spec = get_subgoal_spec(env_name)['num_subgoals']
        assert G == G_spec, f"dataset has G={G} subgoal_dones but envs.subgoals says {G_spec} for {env_name}"
    except ValueError:
        pass
    return G


def prepare_pbrl(config, env_name: str, env, train_dataset, seed: int, discount: float,
                 checkpoint_dir: str, save_dir: str, logger=None, log_step: int = 0,
                 is_resumed: bool = False, log_enabled: bool = True) -> PBRLContext:
    """Resolve the config, obtain the reward model, and build the online hooks.

    Args:
        config: full run config with reward_model.baseline set.
        env: the training env (only used to infer the max episode length).
        train_dataset: offline dataset (after process_train_dataset); state observations.
        discount: agent discount (MC returns for label_mode=success_return).
        checkpoint_dir: rolling checkpoint dir of the run (get_checkpoint_dir(seed, env, algo)).
        save_dir: run artifact dir from setup_experiment_logging.
        logger: LoggingHelper with a 'reward_model' prefix.
    """
    rm_cfg = resolve_reward_model_config(config, env_name)
    baseline = rm_cfg['baseline']
    G = _num_subgoals(train_dataset, env_name) if baseline == 'pref_subgoal' else 0
    aggregator = None
    if G > 0:
        aggregator = SubgoalRewardAggregator(G, rm_cfg['weighting'], rm_cfg['subgoal_done_bonus'],
                                             dag=resolve_subgoal_dag(rm_cfg['subgoal_dag'], env_name))
    print(f"[PBRL] baseline={baseline} | G={G} | {aggregator if aggregator else 'r(s)'}")

    example_obs = np.asarray(train_dataset['observations'][:1], dtype=np.float32)
    rolling_rm = os.path.join(checkpoint_dir, REWARD_MODEL_CKPT_NAME)
    rolling_pref = os.path.join(checkpoint_dir, PREF_BUFFER_CKPT_NAME)
    save_dirs = [checkpoint_dir] + ([save_dir] if log_enabled else [])

    params = apply_fn = None
    pretrain_out: Dict[str, Any] = {}
    payload = None

    if is_resumed and os.path.exists(rolling_rm):
        print("[PBRL] Resuming reward model from the rolling checkpoint (pretraining skipped)")
        params, apply_fn, _ = load_reward_model(rolling_rm, example_obs, seed, expected_goal_dim=G)
        if os.path.exists(rolling_pref):
            payload = check_pref_payload(load_pref_payload(rolling_pref), G)
    elif rm_cfg['pretrained_checkpoint_path'] is not None:
        src = rm_cfg['pretrained_checkpoint_path']
        print(f"[PBRL] Loading reward model from {src} (frozen={rm_cfg['freeze_pretrained']})")
        params, apply_fn, _ = load_reward_model(os.path.join(src, REWARD_MODEL_CKPT_NAME), example_obs, seed,
                                                expected_goal_dim=G)
        if not rm_cfg['freeze_pretrained']:
            src_pref = os.path.join(src, PREF_BUFFER_CKPT_NAME)
            if os.path.exists(src_pref):
                payload = check_pref_payload(load_pref_payload(src_pref), G)
            else:
                print(f"[PBRL] No {PREF_BUFFER_CKPT_NAME} in {src}; online reward updates disabled.")
    else:
        arrays = dataset_arrays_for_pbrl(train_dataset, need_subgoal_dones=G > 0)
        if rm_cfg['rollout_dataset_path'] is not None:
            rollouts = load_rollout_dataset(rm_cfg['rollout_dataset_path'], need_subgoal_dones=G > 0)
            arrays = concat_transition_arrays(arrays, {k: rollouts[k] for k in arrays})
            print(f"[PBRL] Added {rollouts['rewards'].shape[0]} rollout transitions for reward pretraining")
        env_t_max = rm_cfg['env_max_episode_length'] or resolve_env_max_episode_length(env_name, env)
        if rm_cfg['online_enabled'] and env_t_max is None:
            print("[PBRL] WARNING: unknown env max episode length; online episodes longer than the "
                  "longest offline episode will be truncated. Set reward_model.env_max_episode_length.")
        pretrain_out = pretrain_reward_model(arrays, rm_cfg, seed=seed, discount=discount, num_subgoals=G,
                                             env_max_episode_length=env_t_max, logger=logger, start_step=log_step)
        params, apply_fn = pretrain_out['params'], pretrain_out['apply_fn']

    for d in save_dirs:
        save_reward_model(os.path.join(d, REWARD_MODEL_CKPT_NAME), params, rm_cfg, goal_dim=G)
    if pretrain_out:
        for d in save_dirs:  # lets a later run resume / reuse the offline pairs
            save_pickle_atomic(os.path.join(d, PREF_BUFFER_CKPT_NAME), {
                'pref_buffer_offline': pretrain_out['pref_buffer_offline'],
                'store_offline': pretrain_out['store_offline'],
                'pref_buffer_online': {}, 'online_episode_store': {},
                'online_pairs_seen': 0, 'online_updates': 0,
            })

    reward = LearnedReward({'params': params}, apply_fn, num_subgoals=G, aggregator=aggregator)

    state = None
    if rm_cfg['online_enabled'] and (pretrain_out or payload is not None):
        if pretrain_out:
            tx, opt_state = pretrain_out['tx'], pretrain_out['opt_state']
            src = {'pref_buffer_offline': pretrain_out['pref_buffer_offline'],
                   'store_offline': pretrain_out['store_offline']}
        else:
            tx = optax.adam(float(rm_cfg['lr']))
            opt_state = tx.init(params)
            src = payload
        state = OnlinePBRLState(
            reward=reward, tx=tx, opt_state=opt_state, rm_cfg=rm_cfg,
            pref_buffer_offline=src['pref_buffer_offline'], store_offline=src['store_offline'],
            np_rng=np.random.default_rng(seed + 71001),
            pref_buffer_online=src.get('pref_buffer_online') or {},
            online_episode_store=src.get('online_episode_store') or {},
            online_pairs_seen=int(src.get('online_pairs_seen', 0)),
            online_updates=int(src.get('online_updates', 0)),
        )
        print(f"[PBRL] Online reward updates every {rm_cfg['online_num_interact']} env steps "
              f"(feedback_batch={rm_cfg['online_feedback_batch']}, max_feedback={rm_cfg['online_max_feedback']})")
    elif rm_cfg['online_enabled']:
        print("[PBRL] online_enabled=true but no offline preference buffer is available; online updates disabled.")

    on_episode_end = on_interval = interval = None
    if state is not None:
        on_episode_end, on_interval = make_online_callbacks(state, save_dirs, logger=logger)
        interval = int(rm_cfg['online_num_interact'])

    return PBRLContext(rm_cfg=rm_cfg, reward=reward, state=state, on_episode_end=on_episode_end,
                       on_interval=on_interval, interval=interval,
                       pretrain_metrics=dict(pretrain_out.get('metrics', {})))
