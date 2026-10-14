import collections
import re
import time

import gymnasium
import numpy as np
import ogbench
from gymnasium.spaces import Box

from vadar.utils.datasets import Dataset
from envs import gymnasium_robotics_utils
# from envs import d3il_utils

GYMNASIUM_ROBOTICS_ADROITHAND_ENVS = [
    'AdroitHandDoor-v1',
    'AdroitHandHammer-v1',
    'AdroitHandPen-v1',
    'AdroitHandRelocate-v1',
]

D3IL_ENVS = [
    'inserting-state',
    'inserting-image',
    'stacking-state',
    'stacking-image',
]

class EpisodeMonitor(gymnasium.Wrapper):
    """Environment wrapper to monitor episode statistics."""

    def __init__(self, env, filter_regexes=None):
        super().__init__(env)
        self._reset_stats()
        self.total_timesteps = 0
        self.filter_regexes = filter_regexes if filter_regexes is not None else []

    def _reset_stats(self):
        self.reward_sum = 0.0
        self.episode_length = 0
        self.start_time = time.time()

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)

        # Remove keys that are not needed for logging.
        for filter_regex in self.filter_regexes:
            for key in list(info.keys()):
                if re.match(filter_regex, key) is not None:
                    del info[key]

        self.reward_sum += reward
        self.episode_length += 1
        self.total_timesteps += 1
        info['total'] = {'timesteps': self.total_timesteps}

        if terminated or truncated:
            info['episode'] = {}
            info['episode']['final_reward'] = reward
            info['episode']['return'] = self.reward_sum
            info['episode']['length'] = self.episode_length
            info['episode']['duration'] = time.time() - self.start_time

            if hasattr(self.unwrapped, 'get_normalized_score'):
                info['episode']['normalized_return'] = (
                    self.unwrapped.get_normalized_score(info['episode']['return']) * 100.0
                )

        return observation, reward, terminated, truncated, info

    def reset(self, *args, **kwargs):
        self._reset_stats()
        return self.env.reset(*args, **kwargs)


class FrameStackWrapper(gymnasium.Wrapper):
    """Environment wrapper to stack observations."""

    def __init__(self, env, num_stack):
        super().__init__(env)

        self.num_stack = num_stack
        self.frames = collections.deque(maxlen=num_stack)

        low = np.concatenate([self.observation_space.low] * num_stack, axis=-1)
        high = np.concatenate([self.observation_space.high] * num_stack, axis=-1)
        self.observation_space = Box(low=low, high=high, dtype=self.observation_space.dtype)

    def get_observation(self):
        assert len(self.frames) == self.num_stack
        return np.concatenate(list(self.frames), axis=-1)

    def reset(self, **kwargs):
        ob, info = self.env.reset(**kwargs)
        for _ in range(self.num_stack):
            self.frames.append(ob)
        if 'goal' in info:
            info['goal'] = np.concatenate([info['goal']] * self.num_stack, axis=-1)
        return self.get_observation(), info

    def step(self, action):
        ob, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(ob)
        return self.get_observation(), reward, terminated, truncated, info


def get_goal_dim(env_name, subgoal_cfg):
    """Width of the goal suffix appended to observations under `subgoal_cfg` (0 if none).

    G for goal_obs 'cumulative'/'singular', the HER object-state goal dim for 'state',
    and the encoder embedding dim for 'latent'.
    """
    goal_obs = subgoal_cfg.get('goal_obs') if subgoal_cfg is not None else None
    if goal_obs is None:
        return 0
    if goal_obs in ('cumulative', 'singular'):
        from envs.subgoals import get_subgoal_spec
        return int(subgoal_cfg.get('num_subgoals') or get_subgoal_spec(env_name)['num_subgoals'])
    if goal_obs == 'state':
        if env_name.startswith('libero_'):
            from envs.libero_utils import get_her_spec
        else:
            from envs.robomimic_utils import get_her_spec
        return int(get_her_spec(env_name)['goal_dim'])
    if goal_obs == 'latent':
        assert subgoal_cfg.get('her_encoder_path'), "'latent' goal_obs requires her_encoder_path"
        return int(np.load(subgoal_cfg['her_encoder_path'])['embed_dim'])
    raise ValueError(f"Unknown goal_obs: {goal_obs!r}")


def make_env_and_datasets(env_name, frame_stack=None, action_clip_eps=1e-5, rew_fn='sparse', post_success_steps=0,
                          subgoal_cfg=None, bc_dataset_path=None, anchor_demo=None):
    """Make offline RL environment and datasets.

    Args:
        env_name: Name of the environment or dataset.
        frame_stack: Number of frames to stack.
        action_clip_eps: Epsilon for action clipping.
        subgoal_cfg: VaDaR subgoal config (docs/DESIGN.md). None = upstream behavior. Otherwise
            env.step reports info['subgoal_dones'] and the training dataset has `subgoal_dones`.

    Returns:
        A tuple of the environment, evaluation environment, training dataset, and validation dataset.
    """

    if 'singletask' in env_name and subgoal_cfg is not None:
        # OGBench with subgoal tracking (scene task4).
        from envs.ogbench_utils import make_ogbench_env_and_datasets
        env, eval_env, train_dataset, val_dataset = make_ogbench_env_and_datasets(
            env_name, compact_dataset=False, subgoal_cfg=subgoal_cfg)
        env = EpisodeMonitor(env, filter_regexes=['.*privileged.*', '.*proprio.*'])
        eval_env = EpisodeMonitor(eval_env, filter_regexes=['.*privileged.*', '.*proprio.*'])
        train_dataset = Dataset.create(**train_dataset)
        val_dataset = Dataset.create(**val_dataset)
    elif 'singletask' in env_name:
        # OGBench.
        env, train_dataset, val_dataset = ogbench.make_env_and_datasets(env_name)
        eval_env = ogbench.make_env_and_datasets(env_name, env_only=True)
        env = EpisodeMonitor(env, filter_regexes=['.*privileged.*', '.*proprio.*'])
        eval_env = EpisodeMonitor(eval_env, filter_regexes=['.*privileged.*', '.*proprio.*'])
        train_dataset = Dataset.create(**train_dataset)
        val_dataset = Dataset.create(**val_dataset)
    elif 'antmaze' in env_name and ('diverse' in env_name or 'play' in env_name or 'umaze' in env_name):
        # D4RL AntMaze.
        from envs import d4rl_utils

        env = d4rl_utils.make_env(env_name)
        eval_env = d4rl_utils.make_env(env_name)
        dataset = d4rl_utils.get_dataset(env, env_name)
        train_dataset, val_dataset = dataset, None
    
    elif env_name in GYMNASIUM_ROBOTICS_ADROITHAND_ENVS:
        import gymnasium_robotics
        gymnasium.register_envs(gymnasium_robotics)
        # Gymnasium Robotics Adroit environments with Minari datasets.
        print(f"Setting up Gymnasium Robotics environment: {env_name}")
        env = gymnasium_robotics_utils.make_env(env_name, seed=1)
        eval_env = gymnasium_robotics_utils.make_env(env_name, seed=1 + 42)
        env = EpisodeMonitor(env)
        eval_env = EpisodeMonitor(eval_env)
        dataset = gymnasium_robotics_utils.get_dataset(env_name)
        train_dataset, val_dataset = dataset, None
    elif env_name.startswith("robocasa_"):
        # RoboCasa.
        from envs import robocasa_utils

        if anchor_demo is None and subgoal_cfg is not None:
            anchor_demo = subgoal_cfg.get('anchor_demo')
        env = robocasa_utils.make_env(env_name, seed=0, anchor_demo=anchor_demo, subgoal_cfg=subgoal_cfg)
        eval_env = robocasa_utils.make_env(env_name, seed=42, anchor_demo=anchor_demo, subgoal_cfg=subgoal_cfg)
        env = EpisodeMonitor(env)
        eval_env = EpisodeMonitor(eval_env)
        dataset = robocasa_utils.get_dataset(env, env_name)
        train_dataset, val_dataset = dataset, None
    elif env_name.startswith("libero_"):
        # LIBERO.
        from envs import libero_utils

        env = libero_utils.make_env(env_name, seed=0, subgoal_cfg=subgoal_cfg)
        eval_env = libero_utils.make_env(env_name, seed=42, subgoal_cfg=subgoal_cfg)
        env = EpisodeMonitor(env)
        eval_env = EpisodeMonitor(eval_env)
        dataset = libero_utils.get_dataset(env, env_name)
        train_dataset, val_dataset = dataset, None
    elif env_name.startswith("lift") or env_name.startswith("can") or env_name.startswith("square") or \
        env_name.startswith("transport") or env_name.startswith("tool_hang"):
        # RoboMimic.
        from envs import robomimic_utils

        env = robomimic_utils.make_env(env_name, seed=0, frame_stack=frame_stack or 1, post_success_steps=post_success_steps,
                                       subgoal_cfg=subgoal_cfg)
        eval_env = robomimic_utils.make_env(env_name, seed=42, frame_stack=frame_stack or 1, post_success_steps=post_success_steps,
                                            subgoal_cfg=subgoal_cfg)
        env = EpisodeMonitor(env)
        eval_env = EpisodeMonitor(eval_env)
        dataset = robomimic_utils.get_dataset(env, env_name)
        train_dataset, val_dataset = dataset, None
    elif any(env_name.startswith(task) for task in ['stacking', 'gate_insertion']):
        # D3IL environments
        print(f"Setting up D3IL environment: {env_name}")
        env = d3il_utils.make_env(env_name, seed=0)
        eval_env = d3il_utils.make_env(env_name, seed=42)
        env = EpisodeMonitor(env)
        eval_env = EpisodeMonitor(eval_env)
        dataset = d3il_utils.get_dataset(env_name)
        train_dataset, val_dataset = dataset, None
    elif env_name.startswith("PutBallintoBowl"):
        from envs import roboverse_utils
        env = roboverse_utils.make_env(env_name, seed=0)
        eval_env = roboverse_utils.make_env(env_name, seed=42)
        dataset = roboverse_utils.get_dataset(env, env_name)
        train_dataset, val_dataset = dataset, None        
    elif 'D4RL/kitchen' in env_name or env_name.startswith('kitchen-'):
        from envs import franka_kitchen_utils
        kitchen_name = env_name if 'D4RL/kitchen' in env_name else f'D4RL/kitchen/{env_name}'
        env = franka_kitchen_utils.make_env(kitchen_name)
        eval_env = franka_kitchen_utils.make_env(kitchen_name)
        env = EpisodeMonitor(env)
        eval_env = EpisodeMonitor(eval_env)
        dataset = franka_kitchen_utils.get_dataset(kitchen_name)
        train_dataset, val_dataset = dataset, None
    elif 'pusht' in env_name:
        from envs import pusht_utils
        if subgoal_cfg is None:
            env = pusht_utils.make_env(env_name, seed=0, rew_fn=rew_fn)
            eval_env = pusht_utils.make_env(env_name, seed=42, rew_fn=rew_fn)
        else:
            env = pusht_utils.make_env(env_name, seed=0, rew_fn=rew_fn,
                                       obstacle_cfg=subgoal_cfg.get('obstacle'), subgoal_cfg=subgoal_cfg)
            eval_env = pusht_utils.make_env(env_name, seed=42, rew_fn=rew_fn,
                                            obstacle_cfg=subgoal_cfg.get('obstacle'), subgoal_cfg=subgoal_cfg)
        env = EpisodeMonitor(env)
        eval_env = EpisodeMonitor(eval_env)
        dataset = pusht_utils.get_dataset(
            env_name, bc_dataset_path=bc_dataset_path,
            obstacle_cfg=subgoal_cfg.get('obstacle') if subgoal_cfg is not None else None,
            subgoal_cfg=subgoal_cfg)
        dataset = Dataset.create(**dataset)
        train_dataset, val_dataset = dataset, None
    else:
        raise ValueError(f'Unsupported environment: {env_name}')

    if frame_stack is not None:
        env = FrameStackWrapper(env, frame_stack)
        eval_env = FrameStackWrapper(eval_env, frame_stack)

    env.reset()
    eval_env.reset()

    # Clip dataset actions.
    if action_clip_eps is not None:
        train_dataset = train_dataset.copy(
            add_or_replace=dict(actions=np.clip(train_dataset['actions'], -1 + action_clip_eps, 1 - action_clip_eps))
        )
        if val_dataset is not None:
            val_dataset = val_dataset.copy(
                add_or_replace=dict(actions=np.clip(val_dataset['actions'], -1 + action_clip_eps, 1 - action_clip_eps))
            )

    return env, eval_env, train_dataset, val_dataset
