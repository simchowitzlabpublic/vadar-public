"""Online RL with a learned preference reward (the "Dense" and "Pref-Subgoal" baselines).

`run_online_pbrl` is `vadar.runners.online_rl_runner.run_online_rl` (and
`_q_warmup_collect_pbrl` is its `_q_warmup_collect`) copied verbatim except for
the blocks marked `# --- PBRL begin/end ---`:

1. After every env.step the stored reward is `pbrl_reward.step_reward(ob, info)`
   instead of the env reward: r(ob) for Dense, the weighted per-subgoal reward
   from info['subgoal_dones'] for Pref-Subgoal. The env reward is kept only to
   be handed to the episode hook.
2. When an episode ends (before its transitions enter the replay buffer) it is
   passed to `pbrl_on_episode_end(transitions, is_success, env_rewards)`, which
   adds it to the online preference store.
3. Every `pbrl_interval` loop steps, right after that step's agent update,
   `pbrl_on_interval(log_step, replay_buffer)` fine-tunes the reward model on
   offline + new online preference pairs and relabels the whole replay buffer.
   It logs under the 'reward_model' prefix.
4. The mean learned reward is logged as online_agent/learned_reward_mean.
5. State-only: image observations and frozen encoders are rejected.

Ported from the PBRL hooks of OGPO-pbrl-baseline/OGPO-pbrl-subgoal
`ogpo/runners/online_rl_runner.py` (reward_override_fn, on_episode_end,
online_pbrl_callback). The fork read `info['reward_info']['subtask_status']`
into a `subtask_status` replay field; this uses the upstream `subgoal_dones`.
"""

import gc
from typing import Any, Callable, Dict, Optional, Tuple

import jax
import numpy as np
import tqdm

from vadar.agents.modules.flax_utils import save_agent
from vadar.runners.online_rl_runner import (
    ALGORITHMS_DUAL_EVAL,
    ALGORITHMS_PURE_ONLINE,
    ALGORITHMS_WITH_BATCH_BC,
    ALGORITHMS_WITH_PHASES,
    _actor_supported_kwargs,
    _add_subgoal_dones,
    _build_eval_flags,
    _filter_actor_kwargs,
    _q_warmup_train,
    _run_evaluation,
    _sample_mixed_batch,
    _sample_online_batch,
    _sample_success_batch,
    _split_obs,
    _state_for_encoder,
    _wrap_actor_for_frozen_encoder,
)
from vadar.utils.datasets import Dataset, ReplayBuffer
from vadar.utils.main_helper import LoggingHelper, save_rolling_checkpoint
from vadar.utils.mem_monitor import log_memory
from vadar.utils.sharding import shard_batch


def _q_warmup_collect_pbrl(
    agent, env, replay_buffer, *,
    q_warmup_steps, action_dim, has_images, discount, seed,
    use_success_buffer, mc_regression, encode_fn, track_subgoals=False,
    pbrl_reward=None, pbrl_on_episode_end=None,  # PBRL
):
    """Phase A of phase-split q_warmup: pure env collection with base actor (no BoN).

    PBRL: identical to online_rl_runner._q_warmup_collect except that the stored reward
    is the learned reward and finished episodes are passed to `pbrl_on_episode_end`.

    Runs exactly `q_warmup_steps` env steps. Transitions from the final incomplete
    episode are discarded so the buffer only holds fully-bootstrapped trajectories.
    """
    _use_full_state = agent.config.get('use_state', 'proprio') == 'full'
    online_rng = jax.random.PRNGKey(seed + 40000 + 1)
    initial_buffer_size = replay_buffer.size
    episode_counter = 0
    raw_ob, _ = env.reset(seed=seed + 10000)
    ob, ob_images, ob_full_state = _split_obs(raw_ob)
    if encode_fn is not None and ob_images is not None:
        ob = np.asarray(encode_fn(_state_for_encoder(ob, ob_full_state, _use_full_state)[None], ob_images[None])[0])
        ob_images = None
    _qwarmup_actor_supported = _actor_supported_kwargs(agent.sample_actions)
    trajectory_buffer = []
    traj_env_rewards = []  # PBRL
    action_queue = []

    for _ in tqdm.tqdm(range(q_warmup_steps), dynamic_ncols=True, desc="Q-warmup collect"):
        online_rng, key = jax.random.split(online_rng)
        if len(action_queue) == 0:
            actor_kwargs = dict(observations=ob, rng=key)
            if ob_images is not None:
                actor_kwargs['images'] = ob_images
            if ob_full_state is not None:
                actor_kwargs['full_state'] = ob_full_state
            actor_kwargs = _filter_actor_kwargs(actor_kwargs, _qwarmup_actor_supported)
            action = agent.sample_actions(**actor_kwargs)
            for a in np.array(action).reshape(-1, action_dim):
                action_queue.append(a)
        action = action_queue.pop(0)

        raw_next_ob, int_reward, terminated, truncated, info = env.step(action)
        # --- PBRL begin ---
        traj_env_rewards.append(float(int_reward))
        int_reward = pbrl_reward.step_reward(ob, info)
        # --- PBRL end ---
        next_ob, next_ob_images, next_ob_full_state = _split_obs(raw_next_ob)
        if encode_fn is not None and next_ob_images is not None:
            next_ob = np.asarray(encode_fn(_state_for_encoder(next_ob, next_ob_full_state, _use_full_state)[None], next_ob_images[None])[0])
            next_ob_images = None
        done = terminated or truncated

        transition = dict(
            observations=ob, actions=action, rewards=int_reward,
            terminals=float(done), masks=1.0 - float(terminated),
            next_observations=next_ob,
        )
        if has_images:
            transition['images'] = ob_images
            transition['next_images'] = next_ob_images
        # Only attach full_states when NOT pre-encoding: the encoded obs
        # already folds full_state in, and the replay buffer schema won't
        # have full_states keys.
        if ob_full_state is not None and encode_fn is None:
            transition['full_states'] = ob_full_state
            transition['next_full_states'] = next_ob_full_state
        _add_subgoal_dones(transition, info, track_subgoals)
        trajectory_buffer.append(transition)

        if done:
            is_success = float(info.get('success', 0.0))
            score = is_success if is_success != 0 else 1e-9
            # --- PBRL begin ---
            if pbrl_on_episode_end is not None:
                pbrl_on_episode_end(trajectory_buffer, is_success, traj_env_rewards)
            traj_env_rewards = []
            # --- PBRL end ---
            if mc_regression:
                G = 0.0
                for t_idx in range(len(trajectory_buffer) - 1, -1, -1):
                    G = float(trajectory_buffer[t_idx]['rewards']) + discount * G
                    trajectory_buffer[t_idx]['mc_returns'] = G
            for t in trajectory_buffer:
                if use_success_buffer:
                    t['is_success'] = is_success
                    t['score'] = score
                replay_buffer.add_transition(t)
            trajectory_buffer = []
            episode_counter += 1
            raw_ob, _ = env.reset(seed=seed + 10000 + episode_counter)
            ob, ob_images, ob_full_state = _split_obs(raw_ob)
            if encode_fn is not None and ob_images is not None:
                ob = np.asarray(encode_fn(_state_for_encoder(ob, ob_full_state, _use_full_state)[None], ob_images[None])[0])
                ob_images = None
            action_queue = []
        else:
            ob = next_ob
            ob_images = next_ob_images
            ob_full_state = next_ob_full_state

    discarded = len(trajectory_buffer)
    collected = replay_buffer.size - initial_buffer_size
    num_success = int(replay_buffer._traj_success_mask[:replay_buffer.size].sum())
    print(f"[q_warmup] collect: +{collected} transitions, {episode_counter} complete episodes, "
          f"{discarded} dangling steps discarded. Success transitions in RB: {num_success}.")



def run_online_pbrl(
    agent: Any,
    env: Any,
    eval_env: Any,
    eval_envs: Optional[Any],
    train_dataset: Dataset,
    config: Dict,
    logger: LoggingHelper,
    save_dir: str,
    algo_name: str,
    online_steps: int,
    start_training: int,
    batch_size: int,
    horizon_length: int,
    discount: float,
    utd_warmup: int = 1,
    utd_online: int = 1,
    utd_q: Optional[int] = None,
    utd_pi: Optional[int] = None,
    utd_v: Optional[int] = None,
    offline_ratio: float = 0.0,
    warmup_offline_ratio: float = 0.0,
    q_warmup_mode: str = 'phase_split',
    bc_refine_steps: int = 0,
    q_warmup_steps: int = 0,
    use_success_buffer: bool = False,
    success_buffer_batch_size: int = 256,
    best_of_n: int = 1,
    eval_interval: int = 100000,
    n_eval_envs: int = 32,
    eval_episodes: int = 2,
    video_episodes: int = 0,
    video_frame_skip: int = 3,
    plot_q_vs_mc: bool = False,
    run_name: str = 'online_rl',
    env_name: str = '',
    log_interval: int = 5000,
    step_data_log_interval: int = 10,
    save_interval: int = 500000,
    preemption_backup_interval: int = -1,
    log_enabled: bool = True,
    sparse: bool = False,
    p_aug: Optional[float] = None,
    buffer_size: int = 2000000,
    seed: int = 0,
    is_resumed: bool = False,
    resumed_online_step: int = 0,
    resumed_replay_buffer: Optional[ReplayBuffer] = None,
    start_step: int = 0,
    wandb_run_id: Optional[str] = None,
    force_jax_sync: bool = True,
    encode_fn: Optional[Any] = None,
    mc_regression: bool = False,
    mesh=None,
    # --- PBRL arguments ---
    pbrl_reward: Any = None,
    pbrl_on_episode_end: Optional[Callable] = None,
    pbrl_on_interval: Optional[Callable] = None,
    pbrl_interval: Optional[int] = None,
) -> Tuple[Any, int, Dict, ReplayBuffer]:
    """run_online_rl with the env reward replaced by a learned preference reward.

    See the module docstring for the exact differences. `pbrl_reward` is a
    vadar.rewards.pebble.LearnedReward; the other pbrl_* arguments come from
    vadar.rewards.pbrl_setup.PBRLContext.runner_kwargs().
    """
    # --- PBRL begin ---
    assert pbrl_reward is not None, 'run_online_pbrl needs pbrl_reward'
    assert encode_fn is None, 'The preference reward model is state-only; frozen-encoder runs are not supported'
    # --- PBRL end ---
    if algo_name in ALGORITHMS_PURE_ONLINE and offline_ratio != 0.0:
        raise ValueError(f"{algo_name} requires offline_ratio=0.0, got {offline_ratio}")

    _use_full_state = agent.config.get('use_state', 'proprio') == 'full'

    example_batch = train_dataset.sample(())
    action_dim = example_batch['actions'].shape[-1]
    has_images = 'images' in example_batch
    # --- PBRL begin ---
    assert not has_images, 'The preference reward model is state-only; image observations are not supported'
    # --- PBRL end ---

    needs_offline_data = offline_ratio > 0.0
    needs_warmup_offline = warmup_offline_ratio > 0.0
    assert q_warmup_mode in ('phase_split', 'interleaved'), q_warmup_mode
    if needs_warmup_offline:
        assert q_warmup_mode == 'interleaved', "warmup_offline_ratio requires q_warmup_mode='interleaved'"
    if not needs_offline_data and not needs_warmup_offline:
        train_dataset = None
        gc.collect()

    # Replay buffer
    if is_resumed and resumed_replay_buffer is not None:
        replay_buffer = resumed_replay_buffer
    else:
        rb_template = dict(
            observations=example_batch['observations'],
            actions=example_batch['actions'],
            rewards=np.array(0.0), terminals=np.array(0.0),
            masks=np.array(1.0), next_observations=example_batch['observations'],
        )
        if has_images:
            rb_template['images'] = example_batch['images']
            rb_template['next_images'] = example_batch['images']
        if 'full_states' in example_batch:
            rb_template['full_states'] = example_batch['full_states']
            rb_template['next_full_states'] = example_batch['full_states']
        if mc_regression:
            rb_template['mc_returns'] = np.array(0.0)
        if 'subgoal_dones' in example_batch:
            rb_template['subgoal_dones'] = np.zeros_like(example_batch['subgoal_dones'], dtype=np.float32)
        replay_buffer = ReplayBuffer.create(
            rb_template, buffer_size, numpy_rng=np.random.default_rng(seed + 30000))

    if p_aug is not None:
        replay_buffer.p_aug = p_aug
    track_subgoals = 'subgoal_dones' in replay_buffer._dict
    # --- PBRL begin ---
    if pbrl_reward.subgoal_mode and not track_subgoals:
        raise KeyError('Pref-Subgoal needs subgoal_dones in the replay buffer '
                       '(the offline dataset must carry subgoal_dones)')
    reward_mean_acc, reward_mean_count = 0.0, 0
    # --- PBRL end ---

    episode_counter = 0
    raw_ob, _ = env.reset(seed=seed + 10000)
    ob, ob_images, ob_full_state = _split_obs(raw_ob)
    if encode_fn is not None and ob_images is not None:
        ob = np.asarray(encode_fn(_state_for_encoder(ob, ob_full_state, _use_full_state)[None], ob_images[None])[0])
        ob_images = None
    trajectory_buffer = []
    traj_env_rewards = []  # PBRL
    action_queue = []

    if not is_resumed and hasattr(agent, 'reset_optimizers_with_lr'):
        agent = agent.reset_optimizers_with_lr()
        print("Reset optimizer with online learning rate")

    # Resolve separate Q/Pi/V UTD ratios (default to utd_online)
    eff_utd_q = utd_q if utd_q is not None else utd_online
    eff_utd_pi = utd_pi if utd_pi is not None else utd_online
    eff_utd_v = utd_v if utd_v is not None else eff_utd_q
    eff_utd_v_extra = max(0, eff_utd_v - eff_utd_q) if hasattr(agent, 'batch_v_only_update') else 0
    if eff_utd_v_extra > 0:
        print(f"[UTD] V extra updates: running {eff_utd_v_extra} V-only updates per env step "
              f"(utd_v={eff_utd_v}, utd_q={eff_utd_q})")
    split_utd = eff_utd_q != eff_utd_pi
    if split_utd:
        has_critic_only = hasattr(agent, 'batch_critic_only_update')
        if eff_utd_pi > eff_utd_q:
            print(f"[UTD] WARNING: utd_pi={eff_utd_pi} > utd_q={eff_utd_q} not supported, "
                  f"using max={eff_utd_pi} for both")
            split_utd = False
        elif not has_critic_only:
            print(f"[UTD] WARNING: {algo_name} has no batch_critic_only_update, "
                  f"using max(utd_q={eff_utd_q}, utd_pi={eff_utd_pi}) for both")
            split_utd = False
        else:
            print(f"[UTD] Split updates: critic={eff_utd_q}, actor={eff_utd_pi} per env step")
    elif eff_utd_q != utd_online:
        print(f"[UTD] critic={eff_utd_q}, actor={eff_utd_pi} per env step")

    # Pre-compute invariants
    needs_batch_bc = algo_name in ALGORITHMS_WITH_BATCH_BC
    has_phases = algo_name in ALGORITHMS_WITH_PHASES
    needs_dual_eval = algo_name in ALGORITHMS_DUAL_EVAL
    should_log_step_data = step_data_log_interval > 0

    log_step = start_step
    eval_step = 0
    update_info = {}
    online_rng = jax.random.PRNGKey(seed + 40000)
    eval_flags = _build_eval_flags(n_eval_envs, eval_episodes, discount, run_name, env_name)

    # === Phase-split q_warmup ===
    # When q_warmup_steps > 0 and not resuming, run a dedicated collect-then-train
    # phase before the main online loop: q_warmup_steps env steps with base actor
    # (no BoN), then q_warmup_steps * utd_warmup critic updates back-to-back.
    # start_training is bypassed for the warmed-up main loop.
    # Phase-split is only safe when bc_refine is not set — otherwise we'd reverse
    # the old order (BC-refine then q_warmup) by running warmup first.
    do_phase_split_q_warmup = (
        q_warmup_mode == 'phase_split' and has_phases and q_warmup_steps > 0 and bc_refine_steps == 0 and not is_resumed
    )
    if do_phase_split_q_warmup:
        print(f"\n=== Phase-split Q-warmup: collect {q_warmup_steps} env steps -> "
              f"{q_warmup_steps * utd_warmup} critic updates ===")
        _q_warmup_collect_pbrl(
            agent=agent, env=env, replay_buffer=replay_buffer,
            q_warmup_steps=q_warmup_steps, action_dim=action_dim,
            has_images=has_images, discount=discount, seed=seed,
            use_success_buffer=use_success_buffer, mc_regression=mc_regression,
            encode_fn=encode_fn, track_subgoals=track_subgoals,
            pbrl_reward=pbrl_reward, pbrl_on_episode_end=pbrl_on_episode_end,  # PBRL
        )
        agent, log_step = _q_warmup_train(
            agent=agent, replay_buffer=replay_buffer, train_dataset=train_dataset,
            logger=logger,
            q_warmup_steps=q_warmup_steps, utd_warmup=utd_warmup,
            batch_size=batch_size, horizon_length=horizon_length, discount=discount,
            use_success_buffer=use_success_buffer,
            success_buffer_batch_size=success_buffer_batch_size,
            needs_offline_data=needs_offline_data, offline_ratio=offline_ratio,
            log_interval=log_interval, log_step=log_step,
            mesh=mesh, force_jax_sync=force_jax_sync,
        )
        # Reset env for the main loop and zero out the warmup budget there.
        raw_ob, _ = env.reset(seed=seed + 10000 + 999999)
        ob, ob_images, ob_full_state = _split_obs(raw_ob)
        if encode_fn is not None and ob_images is not None:
            ob = np.asarray(encode_fn(_state_for_encoder(ob, ob_full_state, _use_full_state)[None], ob_images[None])[0])
            ob_images = None
        trajectory_buffer = []
        traj_env_rewards = []  # PBRL
        action_queue = []
        episode_counter = 0
        q_warmup_steps = 0
        start_training = 0

    q_warmup_end = start_training + bc_refine_steps + q_warmup_steps
    eval_start = q_warmup_end if has_phases else start_training

    # Determine which policy to use for online rollouts
    if config.get('use_one_step_for_rollouts', False):
        if best_of_n > 1 and hasattr(agent, 'sample_actions_one_step_BON'):
            _actor_method_name = 'sample_actions_one_step_BON'
            print(f"Using one-step BON policy (N={best_of_n}) for online rollouts")
        elif hasattr(agent, 'sample_actions_one_step'):
            _actor_method_name = 'sample_actions_one_step'
            print("Using one-step policy for online rollouts")
        else:
            _actor_method_name = 'sample_actions'
    elif best_of_n > 1 and hasattr(agent, 'sample_actions_BON'):
        _actor_method_name = 'sample_actions_BON'
    else:
        _actor_method_name = 'sample_actions'

    # Introspect the rollout actor's kwarg set so we don't pass OGPO-only kwargs
    # (images / full_state / is_encoded) to baselines whose sample_actions only
    # accepts (observations, rng).
    _rollout_actor_supported = _actor_supported_kwargs(getattr(agent, _actor_method_name))

    start_online_step = 1
    if is_resumed and resumed_online_step > 0:
        start_online_step = resumed_online_step + 1
        print(f"Resuming online loop at step {start_online_step}")

    eval_info = {}
    mem_log_interval = max(eval_interval, 20000)  # Log memory at least every eval

    def _maybe_shard(b):
        return shard_batch(b, mesh) if (mesh is not None and b is not None) else b

    log_memory("online_start", step=log_step, logger=logger)

    for i in tqdm.tqdm(range(start_online_step, online_steps + 1), dynamic_ncols=True, desc="Online RL", initial=start_online_step - 1, total=online_steps,):
        log_step += 1
        online_rng, key = jax.random.split(online_rng)

        # Action chunking
        if len(action_queue) == 0:
            actor_kwargs = dict(observations=ob, rng=key)
            if ob_images is not None:
                actor_kwargs['images'] = ob_images
            if ob_full_state is not None:
                actor_kwargs['full_state'] = ob_full_state
            # Drop kwargs the rollout actor doesn't accept (baselines'
            # sample_actions takes only (observations, rng)).
            actor_kwargs = _filter_actor_kwargs(actor_kwargs, _rollout_actor_supported)
            action = getattr(agent, _actor_method_name)(**actor_kwargs)
            for a in np.array(action).reshape(-1, action_dim):
                action_queue.append(a)
        action = action_queue.pop(0)

        # Environment step
        raw_next_ob, int_reward, terminated, truncated, info = env.step(action)
        # --- PBRL begin ---
        traj_env_rewards.append(float(int_reward))
        int_reward = pbrl_reward.step_reward(ob, info)
        reward_mean_acc += int_reward
        reward_mean_count += 1
        # --- PBRL end ---
        next_ob, next_ob_images, next_ob_full_state = _split_obs(raw_next_ob)
        if encode_fn is not None and next_ob_images is not None:
            next_ob = np.asarray(encode_fn(_state_for_encoder(next_ob, next_ob_full_state, _use_full_state)[None], next_ob_images[None])[0])
            next_ob_images = None
        done = terminated or truncated

        env_info = {k: v for k, v in info.items() if k.startswith("distance")}
        if env_info:
            try:
                logger.log(env_info, "env", step=log_step)
            except Exception:
                pass

        transition = dict(
            observations=ob, actions=action, rewards=int_reward,
            terminals=float(done), masks=1.0 - float(terminated),
            next_observations=next_ob,
        )
        if has_images:
            transition['images'] = ob_images
            transition['next_images'] = next_ob_images
        # Only attach full_states when NOT pre-encoding: the encoded obs
        # already folds full_state in, and the replay buffer schema won't
        # have full_states keys.
        if ob_full_state is not None and encode_fn is None:
            transition['full_states'] = ob_full_state
            transition['next_full_states'] = next_ob_full_state
        _add_subgoal_dones(transition, info, track_subgoals)
        trajectory_buffer.append(transition)

        if should_log_step_data and i % step_data_log_interval == 0:
            try:
                logger.log({
                    'phase': 'online', 'reward': float(int_reward), 'done': float(done),
                    'batch_obs_mean': float(np.mean(ob)), 'batch_obs_std': float(np.std(ob)),
                    'batch_action_mean': float(np.mean(action)), 'batch_action_std': float(np.std(action)),
                }, "step_data", step=log_step)
            except Exception:
                pass

        if done:
            is_success = float(info.get('success', 0.0))
            score = is_success if is_success != 0 else 1e-9

            # --- PBRL begin ---
            if pbrl_on_episode_end is not None:
                pbrl_on_episode_end(trajectory_buffer, is_success, traj_env_rewards)
            traj_env_rewards = []
            # --- PBRL end ---

            # Compute MC returns for the trajectory (backward pass)
            if mc_regression:
                G = 0.0
                for t_idx in range(len(trajectory_buffer) - 1, -1, -1):
                    G = float(trajectory_buffer[t_idx]['rewards']) + discount * G
                    trajectory_buffer[t_idx]['mc_returns'] = G

            for t in trajectory_buffer:
                if use_success_buffer:
                    t['is_success'] = is_success
                    t['score'] = score
                replay_buffer.add_transition(t)
            trajectory_buffer = []
            episode_counter += 1
            raw_ob, _ = env.reset(seed=seed + 10000 + episode_counter)
            ob, ob_images, ob_full_state = _split_obs(raw_ob)
            if encode_fn is not None and ob_images is not None:
                ob = np.asarray(encode_fn(_state_for_encoder(ob, ob_full_state, _use_full_state)[None], ob_images[None])[0])
                ob_images = None
            action_queue = []
        else:
            ob = next_ob
            ob_images = next_ob_images
            ob_full_state = next_ob_full_state

        # Training update
        if i < start_training:
            continue

        is_warmup = has_phases and (i < bc_refine_steps or i < q_warmup_end)
        utd_ratio = utd_warmup if is_warmup else max(eff_utd_q, eff_utd_pi)

        if replay_buffer.size < batch_size * utd_ratio * horizon_length:
            continue

        # Sample batch
        if is_warmup and needs_warmup_offline:
            batch = _sample_mixed_batch(replay_buffer, train_dataset, batch_size, utd_ratio,
                                        horizon_length, discount, warmup_offline_ratio)
        elif needs_offline_data:
            batch = _sample_mixed_batch(replay_buffer, train_dataset, batch_size, utd_ratio,
                                        horizon_length, discount, offline_ratio)
        else:
            batch = _sample_online_batch(replay_buffer, batch_size, utd_ratio, horizon_length, discount)

        # Success buffer: sample with utd_ratio (= max(utd_q, utd_pi)) so batch_update can slice per phase
        batch_success = None
        success_flag = False
        if use_success_buffer:
            batch_success, success_flag = _sample_success_batch(replay_buffer, success_buffer_batch_size,
                                                                utd_ratio, horizon_length, discount)

        if has_phases and i < bc_refine_steps:
            if batch_success is None:
                batch_success = batch
            agent, update_info["online_agent"] = agent.batch_bc_refine_update(_maybe_shard(batch_success))
        elif has_phases and i < q_warmup_end:
            agent, update_info["online_agent"] = agent.batch_q_warmup_update(_maybe_shard(batch))
            if batch_success is not None and q_warmup_mode != 'interleaved':
                agent, update_info["online_agent"] = agent.batch_q_warmup_update(_maybe_shard(batch_success))
        else:
            if batch_success is None:
                batch_success = batch
            if split_utd:
                batch_joint = jax.tree.map(lambda x: x[:eff_utd_pi], batch)
                bs_joint = jax.tree.map(lambda x: x[:eff_utd_pi], batch_success)
                agent, update_info["online_agent"] = agent.batch_update(
                    _maybe_shard(batch_joint), _maybe_shard(bs_joint), success_flag)
                batch_extra = jax.tree.map(lambda x: x[eff_utd_pi:eff_utd_q], batch)
                agent, critic_info = agent.batch_critic_only_update(_maybe_shard(batch_extra))
                update_info["online_agent"].update(critic_info)
            else:
                agent, update_info["online_agent"] = agent.batch_update(
                    _maybe_shard(batch), _maybe_shard(batch_success), success_flag)

            if eff_utd_v_extra > 0:
                v_batch = _sample_online_batch(replay_buffer, batch_size, eff_utd_v_extra,
                                               horizon_length, discount)
                agent, v_info = agent.batch_v_only_update(_maybe_shard(v_batch))
                update_info["online_agent"].update(v_info)

        if force_jax_sync:
            jax.block_until_ready(agent)

        # --- PBRL begin ---
        # Online reward-model update + replay-buffer relabel (after this step's agent update).
        if pbrl_on_interval is not None and pbrl_interval and i % pbrl_interval == 0:
            rm_metrics = pbrl_on_interval(log_step, replay_buffer)
            if rm_metrics:
                logger.log(rm_metrics, 'reward_model', step=log_step)
        # --- PBRL end ---

        # Logging
        if i % log_interval == 0 and update_info:
            for key, info_dict in update_info.items():
                logger.log(info_dict, key, step=log_step)
            update_info = {}
        # --- PBRL begin ---
        if i % log_interval == 0 and reward_mean_count > 0:
            logger.log({'learned_reward_mean': reward_mean_acc / reward_mean_count}, 'online_agent', step=log_step)
            reward_mean_acc, reward_mean_count = 0.0, 0
        # --- PBRL end ---

        # Evaluation
        if i >= eval_start and (i == online_steps or (eval_interval > 0 and i % eval_interval == 0)):
            eval_step += 1
            gc.collect()

            ode_actor_fn = getattr(agent, 'compute_flow_actions_corrected', agent.compute_flow_actions)
            if encode_fn is not None:
                _actor_uses_state = agent.config.get('_original_actor_obs', 'state') == 'state'
                ode_actor_fn = _wrap_actor_for_frozen_encoder(
                    ode_actor_fn, encode_fn,
                    actor_uses_state=_actor_uses_state, use_full_state=_use_full_state)
            eval_info = _run_evaluation(
                agent, eval_env, eval_envs, eval_flags, action_dim,
                ode_actor_fn, log_step, eval_episodes, video_episodes,
                video_frame_skip, run_name, plot=(eval_step % 2 == 0) and plot_q_vs_mc,
                encode_fn=encode_fn,
            )
            rb_success_pct = 100.0 * replay_buffer._traj_success_mask[:replay_buffer.size].sum() / max(replay_buffer.size, 1)
            print(f"[EVAL ODE] step={i} | success={float(eval_info.get('success', 0.0)):.4f} | rb_success={rb_success_pct:.1f}% ({replay_buffer.size} total)")
            logger.log(eval_info, "eval", step=log_step)

            if needs_dual_eval:
                sde_actor_fn = getattr(agent, _actor_method_name)
                if encode_fn is not None:
                    sde_actor_fn = _wrap_actor_for_frozen_encoder(
                        sde_actor_fn, encode_fn,
                        actor_uses_state=_actor_uses_state, use_full_state=_use_full_state)
                eval_info_sde = _run_evaluation(
                    agent, eval_env, eval_envs, eval_flags, action_dim,
                    sde_actor_fn, log_step, eval_episodes, 0,
                    video_frame_skip, run_name, plot=False,
                    encode_fn=encode_fn,
                )
                print(f"[EVAL SDE] step={i} | success={float(eval_info_sde.get('success', 0.0)):.4f}")
                try:
                    logger.log(eval_info_sde, "eval_sde", step=log_step)
                except Exception:
                    pass

            # Evaluate one-step policy if enabled
            if hasattr(agent, 'one_step_network') and agent.one_step_network is not None:
                # Use BON version if best_of_n > 1 to match rollout behavior
                one_step_actor_fn = agent.sample_actions_one_step_BON if best_of_n > 1 else agent.sample_actions_one_step
                # Frozen-encoder runs: wrap so the env's raw obs gets re-encoded
                # to match the dim the one-step MLP was init'd with.
                if encode_fn is not None:
                    one_step_actor_fn = _wrap_actor_for_frozen_encoder(
                        one_step_actor_fn, encode_fn,
                        actor_uses_state=_actor_uses_state, use_full_state=_use_full_state)
                eval_info_one_step = _run_evaluation(
                    agent, eval_env, eval_envs, eval_flags, action_dim,
                    one_step_actor_fn, log_step, eval_episodes, 0,
                    video_frame_skip, run_name, plot=False,
                )
                print(f"[EVAL OneStep] step={i} | success={float(eval_info_one_step.get('success', 0.0)):.4f}")
                try:
                    logger.log(eval_info_one_step, "eval_one_step", step=log_step)
                except Exception:
                    pass

        # Memory monitoring (every eval or mem_log_interval steps)
        if i % mem_log_interval == 0:
            gc.collect()
            log_memory("online", step=log_step, logger=logger)

        # Saving
        if log_enabled and save_interval > 0 and i % save_interval == 0:
            save_agent(agent, save_dir, log_step)

        if preemption_backup_interval > 0 and i % preemption_backup_interval == 0:
            print(f"Saving rolling checkpoint at step {log_step}")
            save_rolling_checkpoint(agent, replay_buffer, log_step, i, seed, env_name, wandb_run_id=wandb_run_id, algo_name=algo_name)

    save_agent(agent, save_dir, log_step)
    print(f"Online RL training complete. Final step: {log_step}")
    return agent, log_step, eval_info, replay_buffer
