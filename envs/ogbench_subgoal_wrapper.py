"""Subgoal tracking for OGBench scene task4 (ported from the QIPM fork's
envs/ogbench_dense_wrapper.py, OGBenchDenseRewardWrapper).

Sequential subgoals of scene-*-singletask-task4 (put_in_drawer):
  0. button 0 pressed (unlocks the drawer)
  1. drawer open (drawer_pos < -0.12)
  2. block in drawer
  3. drawer closed (|drawer_pos| < 0.04)
The block target is inside the closed drawer, so the drawer must be opened,
filled and closed again.
"""

import os

import gymnasium
import numpy as np
from gymnasium.spaces import Box

from envs import subgoal_utils as SG


def check_subtask_conditions(base_env, no_of_subtasks=4):
    """Raw (un-ratcheted) subgoal conditions from the current env state."""
    # Subtask 0: button 0 pressed
    button_0_pressed = bool(base_env._cur_button_states[0] == 1)

    # Subtask 1: drawer open (drawer_pos < -0.12, range is [-0.16, 0])
    drawer_pos = float(base_env._data.joint('drawer_slide').qpos[0])
    drawer_open = drawer_pos < -0.12

    # Subtask 2: block in drawer
    block_pos = base_env._data.joint('object_joint_0').qpos[:3]
    block_in_drawer = base_env._is_in_drawer(block_pos)

    # Subtask 3: drawer closed (back to ~0)
    drawer_closed = abs(drawer_pos) < 0.04

    raw_goals = [button_0_pressed, drawer_open, block_in_drawer, drawer_closed]
    return raw_goals[:no_of_subtasks]


class OGBenchSubgoalWrapper(gymnasium.Wrapper):
    """Ratchets the sequential task4 subgoals and implements the VaDaR env contract.

    Every step sets info['subgoal_dones'] (float32 (G,), after the step); the
    episode terminates with info['success'] = 1.0 once all subgoals are done
    (as in the fork). reward_mode: sparse = the env's singletask reward; hgc =
    (all subgoals done) - 1 plus +5 on a step where a subgoal newly completes;
    shaped = shaped_reward(subgoal_dones). goal_obs appends the encoded dones.
    """

    def __init__(self, env, subgoal_cfg):
        super().__init__(env)
        self.subgoal_cfg = subgoal_cfg
        self.num_subgoals = subgoal_cfg['num_subgoals']
        self.goal_obs = subgoal_cfg['goal_obs']
        self.reward_mode = subgoal_cfg['reward_mode']
        if self.goal_obs not in SG.GOAL_OBS_MODES:
            raise ValueError(f"goal_obs={self.goal_obs!r} is not supported for OGBench")
        if self.reward_mode == 'dense_parity':
            raise ValueError("reward_mode='dense_parity' is PushT-only")
        self._prereqs = SG.prerequisites(self.num_subgoals, subgoal_cfg['dag'], 'sequential')
        self._shaped_reward = SG.get_shaped_reward_fn() if self.reward_mode == 'shaped' else None
        self.goal_done = [False] * self.num_subgoals

    @property
    def observation_space(self):
        base_space = self.env.observation_space
        if self.goal_obs is not None:
            low = np.concatenate([base_space.low, np.zeros(self.num_subgoals, dtype=np.float32)])
            high = np.concatenate([base_space.high, np.ones(self.num_subgoals, dtype=np.float32)])
            return Box(low=low, high=high, dtype=np.float32)
        return base_space

    def _build_observation(self, obs):
        if self.goal_obs is not None:
            goal_vec = SG.encode_goal_vector(np.asarray(self.goal_done, dtype=np.int8),
                                             self.goal_obs).astype(np.float32)
            obs = np.concatenate([obs, goal_vec], axis=-1)
        return obs

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.goal_done = [False] * self.num_subgoals
        return self._build_observation(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        prev_goal_done = list(self.goal_done)
        raw = check_subtask_conditions(self.env.unwrapped, self.num_subgoals)
        self.goal_done = SG.ratchet(self.goal_done, raw, self._prereqs)
        info['subgoal_dones'] = np.asarray(self.goal_done, dtype=np.float32)

        if self.reward_mode == 'hgc':
            reward = SG.hgc_reward(float(all(self.goal_done)) - 1.0, prev_goal_done, self.goal_done)
        elif self.reward_mode == 'shaped':
            reward = float(self._shaped_reward(info['subgoal_dones']))

        # Terminate on success: all subgoals done.
        if all(self.goal_done):
            terminated = True
            info['success'] = 1.0

        return self._build_observation(obs), reward, terminated, truncated, info


def compute_subgoal_labels(env, dataset, subgoal_cfg):
    """Post-action (N, G) subgoal labels of a non-compact OGBench dataset.

    Replays the next-state info (next_qpos/next_qvel/next_button_states, i.e. the
    state after each action) with env.unwrapped.set_state and applies the
    wrapper's ratchet, starting from all-false at each trajectory start
    (t == 0 or terminals[t-1] == 1).
    """
    G = subgoal_cfg['num_subgoals']
    prereqs = SG.prerequisites(G, subgoal_cfg['dag'], 'sequential')
    base_env = env.unwrapped
    terminals = dataset['terminals']
    N = len(terminals)
    labels = np.zeros((N, G), dtype=np.float32)
    goal_done = [False] * G
    for t in range(N):
        if t == 0 or terminals[t - 1] > 0.5:
            goal_done = [False] * G
        base_env.set_state(dataset['next_qpos'][t], dataset['next_qvel'][t], dataset['next_button_states'][t])
        goal_done = SG.ratchet(goal_done, check_subtask_conditions(base_env, G), prereqs)
        labels[t] = np.asarray(goal_done, dtype=np.float32)
    return labels


def relabel_subgoal_dones(env, dataset, subgoal_cfg, dataset_path=None):
    """Add `subgoal_dones` to an OGBench dataset dict (in place); with goal_obs, append the
    goal suffix (obs: status before the action, next_obs: after); with reward_mode hgc/shaped,
    relabel rewards to match OGBenchSubgoalWrapper. Labels are cached next to dataset_path."""
    G = subgoal_cfg['num_subgoals']
    N = len(dataset['terminals'])
    compute = lambda: compute_subgoal_labels(env, dataset, subgoal_cfg)  # noqa: E731
    if dataset_path is not None and os.path.exists(dataset_path):
        labels = SG.load_or_compute_labels(
            SG.label_cache_path(dataset_path, G, subgoal_cfg['dag']), N, G, compute)
    else:
        labels = compute()
    terminals = np.asarray(dataset['terminals'])
    episode_starts = np.concatenate([[True], terminals[:-1] > 0.5])

    dataset['subgoal_dones'] = labels
    print(f"[subgoals] OGBench: {N} transitions labelled (final-step sums: {labels.sum(axis=0)})")
    if subgoal_cfg['reward_mode'] == 'hgc':
        base = np.all(labels > 0.5, axis=-1).astype(np.float32) - 1.0
        dataset['rewards'] = SG.relabel_rewards('hgc', base, labels, episode_starts)
    elif subgoal_cfg['reward_mode'] == 'shaped':
        dataset['rewards'] = SG.relabel_rewards('shaped', dataset['rewards'], labels, episode_starts)
    if subgoal_cfg['goal_obs'] is not None:
        obs_goal, next_goal = SG.goal_suffixes(labels, episode_starts, subgoal_cfg['goal_obs'])
        obs, next_obs = dataset['observations'], dataset['next_observations']
        dataset['observations'] = np.concatenate([obs, obs_goal], axis=-1).astype(obs.dtype)
        dataset['next_observations'] = np.concatenate([next_obs, next_goal], axis=-1).astype(next_obs.dtype)
