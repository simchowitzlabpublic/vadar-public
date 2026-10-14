"""
LIBERO environment utilities (ported from the QIPM fork).

Multi-task: dispatches per-task obs keys, subgoal predicates, and dataset paths
via TASK_CONFIGS. To add a new LIBERO task, append a new entry there.

Currently supported: libero_KITCHEN_SCENE{3,4,6,8} (subgoals in envs/subgoals.py).

Mirrors the envs/robocasa_utils.py contract: with `subgoal_cfg` set, the
wrapper ratchets the cumulative subgoal dones, reports info['subgoal_dones']
every step and appends the goal suffix when goal_obs is set; get_dataset adds
post-action `subgoal_dones` labels. HER goal_obs modes: 'state' (object-state
goals) and 'latent' (frozen encoder embedding).
"""

import os
import json
import pickle
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box
import h5py
import imageio

from vadar.utils.datasets import Dataset
from envs import subgoal_utils as SG
from envs.libero_subtasks import (
    SUBGOAL_NAMES_KITCHEN_SCENE8, SUBGOAL_CHECKS_KITCHEN_SCENE8,
    SUBGOAL_NAMES_KITCHEN_SCENE3, SUBGOAL_CHECKS_KITCHEN_SCENE3,
    SUBGOAL_NAMES_KITCHEN_SCENE4, SUBGOAL_CHECKS_KITCHEN_SCENE4,
    SUBGOAL_NAMES_KITCHEN_SCENE6, SUBGOAL_CHECKS_KITCHEN_SCENE6,
)


# ---------------------------------------------------------------------------
# Per-task configuration
# ---------------------------------------------------------------------------

DATASET_OBS_KEY = 'observation_state'

_ROBOT_OBS_KEYS = [
    'robot0_eef_pos', 'robot0_eef_quat',
    'robot0_gripper_qpos', 'robot0_joint_pos',
]


def _stove_on_obs(env):
    """Return [1.0] if flat_stove_1 is on, else [0.0]. 1-D float32 array."""
    return np.array(
        [float(env._eval_predicate(("turnon", "flat_stove_1")))],
        dtype=np.float32,
    )


def _drawer_open_obs_ks4(env):
    """Return [1.0] if white_cabinet_1 bottom region is open, else [0.0]."""
    return np.array(
        [float(env._eval_predicate(("open", "white_cabinet_1_bottom_region")))],
        dtype=np.float32,
    )


def _microwave_open_obs_ks6(env):
    """Return [1.0] if microwave_1 is open, else [0.0]."""
    return np.array(
        [float(env._eval_predicate(("open", "microwave_1")))],
        dtype=np.float32,
    )


# Each entry: obs_keys (raw robosuite obs dict keys to flatten + concat) +
# extra_obs_fn (callable env -> 1-D np.array of additional obs) + obs_dim +
# subgoals + filenames.
TASK_CONFIGS = {
    'KITCHEN_SCENE8': {
        'obs_keys': _ROBOT_OBS_KEYS + [
            'moka_pot_1_pos', 'moka_pot_1_quat',
            'moka_pot_2_pos', 'moka_pot_2_quat',
        ],
        'extra_obs_fn': None,
        'extra_obs_dim': 0,
        'obs_dim': 30,  # 16 robot + 14 object
        'max_episode_length': 600,
        'subgoal_names': SUBGOAL_NAMES_KITCHEN_SCENE8,
        'subgoal_checks': SUBGOAL_CHECKS_KITCHEN_SCENE8,
        'bddl_file': 'KITCHEN_SCENE8_put_both_moka_pots_on_the_stove.bddl',
        'init_file': 'KITCHEN_SCENE8_put_both_moka_pots_on_the_stove.pruned_init',
        'dataset_file': 'KITCHEN_SCENE8_put_both_moka_pots_on_the_stove_demo_processed.hdf5',
        # HER object-state goal space: pot1 xyz + pot2 xyz.
        'her_goal_indices': [16, 17, 18, 23, 24, 25],
        'her_pos_blocks': [(0, 3), (3, 6)],
        'her_bool_dims': [],
    },
    'KITCHEN_SCENE3': {
        'obs_keys': _ROBOT_OBS_KEYS + [
            'moka_pot_1_pos', 'moka_pot_1_quat',
        ],
        'extra_obs_fn': _stove_on_obs,
        'extra_obs_dim': 1,
        'obs_dim': 24,  # 16 robot + 7 moka_pot + 1 stove_on
        'max_episode_length': 500,
        'subgoal_names': SUBGOAL_NAMES_KITCHEN_SCENE3,
        'subgoal_checks': SUBGOAL_CHECKS_KITCHEN_SCENE3,
        'bddl_file': 'KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it.bddl',
        'init_file': 'KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it.pruned_init',
        'dataset_file': 'KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo_processed.hdf5',
        # HER object-state goal space: pot xyz + stove_on.
        'her_goal_indices': [16, 17, 18, 23],
        'her_pos_blocks': [(0, 3)],
        'her_bool_dims': [3],
    },
    'KITCHEN_SCENE4': {
        'obs_keys': _ROBOT_OBS_KEYS + [
            'akita_black_bowl_1_pos', 'akita_black_bowl_1_quat',
        ],
        'extra_obs_fn': _drawer_open_obs_ks4,
        'extra_obs_dim': 1,
        'obs_dim': 24,  # 16 robot + 7 bowl + 1 drawer_open
        'max_episode_length': 500,
        'subgoal_names': SUBGOAL_NAMES_KITCHEN_SCENE4,
        'subgoal_checks': SUBGOAL_CHECKS_KITCHEN_SCENE4,
        'bddl_file': 'KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it.bddl',
        'init_file': 'KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it.pruned_init',
        'dataset_file': 'KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_demo_processed.hdf5',
        # HER object-state goal space: bowl xyz + drawer_open.
        'her_goal_indices': [16, 17, 18, 23],
        'her_pos_blocks': [(0, 3)],
        'her_bool_dims': [3],
    },
    'KITCHEN_SCENE6': {
        'obs_keys': _ROBOT_OBS_KEYS + [
            'white_yellow_mug_1_pos', 'white_yellow_mug_1_quat',
        ],
        'extra_obs_fn': _microwave_open_obs_ks6,
        'extra_obs_dim': 1,
        'obs_dim': 24,  # 16 robot + 7 mug + 1 microwave_open
        'max_episode_length': 500,
        'subgoal_names': SUBGOAL_NAMES_KITCHEN_SCENE6,
        'subgoal_checks': SUBGOAL_CHECKS_KITCHEN_SCENE6,
        'bddl_file': 'KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it.bddl',
        'init_file': 'KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it.pruned_init',
        'dataset_file': 'KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it_demo_processed.hdf5',
        # HER object-state goal space: mug xyz + microwave_open.
        'her_goal_indices': [16, 17, 18, 23],
        'her_pos_blocks': [(0, 3)],
        'her_bool_dims': [3],
    },
}


# ---------------------------------------------------------------------------
# Env identification helpers
# ---------------------------------------------------------------------------

def is_libero_env(env_name):
    return env_name.startswith('libero_')


def _get_task_name(env_name):
    return env_name.replace('libero_', '', 1)


def _get_task_config(env_name):
    task_name = _get_task_name(env_name)
    if task_name not in TASK_CONFIGS:
        raise ValueError(
            f"Unknown LIBERO task: {task_name}. "
            f"Supported: {list(TASK_CONFIGS.keys())}"
        )
    return TASK_CONFIGS[task_name]


def get_subtask_config(env_name):
    """Compatibility shim — returns {names, checks} like robocasa_utils."""
    cfg = _get_task_config(env_name)
    return {'names': cfg['subgoal_names'], 'checks': cfg['subgoal_checks']}


def _get_dataset_path(env_name):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = _get_task_config(env_name)
    return os.path.join(project_root, 'data', 'libero', cfg['dataset_file'])


def _get_env_meta(dataset_path):
    with h5py.File(dataset_path, "r") as f:
        return json.loads(f["data"].attrs["env_args"])


def _get_bddl_path(env_name):
    envs_dir = os.path.dirname(os.path.abspath(__file__))
    cfg = _get_task_config(env_name)
    p = os.path.join(envs_dir, 'libero_bddl', cfg['bddl_file'])
    if not os.path.isfile(p):
        raise FileNotFoundError(f"BDDL file not found: {p}")
    return p


def _get_init_states(env_name):
    envs_dir = os.path.dirname(os.path.abspath(__file__))
    cfg = _get_task_config(env_name)
    p = os.path.join(envs_dir, 'libero_data', cfg['init_file'])
    if not os.path.isfile(p):
        raise FileNotFoundError(f"Init states file not found: {p}")
    npz = np.load(p, allow_pickle=True)
    return pickle.loads(npz['archive/data.pkl'])


def _extract_obs_from_raw(raw_obs, obs_keys, extra_obs_fn=None, env=None):
    """Flatten raw obs dict for the configured keys and append extra_obs (if any)."""
    parts = [np.asarray(raw_obs[k]).flatten() for k in obs_keys]
    if extra_obs_fn is not None:
        parts.append(np.asarray(extra_obs_fn(env)).flatten())
    return np.concatenate(parts, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# HER (Hindsight Experience Replay) object-state goal space.
#
# Mirrors the SB3 GoalEnv contract on flat observations: the desired goal is
# the last `goal_dim` dims of the obs vector (goal_obs='state'),
# the achieved goal is `base_obs[her_goal_indices]`, and `her_reward` is the
# vectorized compute_reward equivalent (Fetch convention: success-1 in {-1,0}).
# ---------------------------------------------------------------------------

HER_GC_MODE = 'state'
# Latent-goal HER: goals are z = encoder(base_obs) from the frozen SSL
# bottleneck encoder (scripts/rollout/train_subgoal_classifier.py); success is
# ||z - z_g|| <= eps (calibrated at encoder training time).
HER_LATENT_GC_MODE = 'latent'
HER_MODES = (HER_GC_MODE, HER_LATENT_GC_MODE)


def get_her_spec(env_name):
    """Single source of truth for the HER goal space of a task."""
    cfg = _get_task_config(env_name)
    goal_indices = np.asarray(cfg['her_goal_indices'], dtype=np.int64)
    return dict(
        goal_indices=goal_indices,
        pos_blocks=tuple(tuple(b) for b in cfg['her_pos_blocks']),
        bool_dims=tuple(cfg['her_bool_dims']),
        goal_dim=int(len(goal_indices)),
    )


def _load_her_goal_bank(env_name, full_state=False):
    """Final achieved goals of all demos -> (num_demos, goal_dim) float32.

    full_state=True returns the full final base obs (for latent mode, which
    encodes the whole state) instead of the her_goal_indices slice."""
    spec = get_her_spec(env_name)
    dataset_path = _get_dataset_path(env_name)
    goals = []
    with h5py.File(dataset_path, 'r') as f:
        demos = list(f['data'].keys())
        inds = np.argsort([int(d.split('_')[1]) for d in demos])
        for d in [demos[i] for i in inds]:
            final = np.asarray(f[f'data/{d}/next_obs/{DATASET_OBS_KEY}'][-1], dtype=np.float32)
            goals.append(final if full_state else final[spec['goal_indices']])
    return np.stack(goals, axis=0)


def load_staged_eval_banks(env_name, no_of_subtasks, encoder=None):
    """Per-stage goal banks for the staged (oracle-subgoal-selector) eval.

    banks[i] pools, across demos, the achieved goals of frames where subtask i
    is done but subtask i+1 is not (mid-progress states, not the degenerate
    completion boundary); banks[K-1] = demo final frames. Goal space is the
    object-state slice, or z = encoder(frame) when an encoder is given
    (latent mode)."""
    spec = get_her_spec(env_name)
    banks = [[] for _ in range(no_of_subtasks)]

    def to_goal(frames):
        return encoder(frames).astype(np.float32) if encoder is not None \
            else frames[:, spec['goal_indices']]

    with h5py.File(_get_dataset_path(env_name), 'r') as f:
        demos = sorted(f['data'].keys(), key=lambda d: int(d.split('_')[1]))
        for d in demos:
            next_obs = np.asarray(f[f'data/{d}/next_obs/{DATASET_OBS_KEY}'], dtype=np.float32)
            status = np.asarray(f[f'data/{d}/subtask_status'], dtype=np.int8)[:, :no_of_subtasks]
            for i in range(no_of_subtasks - 1):
                idxs = np.nonzero((status[:, i] == 1) & (status[:, i + 1] == 0))[0]
                if len(idxs):
                    banks[i].append(to_goal(next_obs[idxs]))
            banks[no_of_subtasks - 1].append(to_goal(next_obs[-1:]))
    return [np.concatenate(b, axis=0) for b in banks]


# ---------------------------------------------------------------------------
# Offline subgoal labels
# ---------------------------------------------------------------------------

def _post_action_labels(stored_status):
    """Post-action labels of one demo from the hdf5's `subtask_status`.

    scripts/extract_libero_dataset.py stores the ratchet at states[t] (pre-action),
    and next_obs[t] = obs[t+1] with next_obs[-1] = obs[-1]. The post-action label
    of step t is therefore stored[t+1] (stored[T-1] for the last step, matching
    next_obs[-1]).
    """
    post = np.array(stored_status, dtype=np.float32, copy=True)
    if len(post) > 1:
        post[:-1] = post[1:]
    return post


def _unwrap_lowdim_wrapper(env):
    cur = env
    while cur is not None:
        if isinstance(cur, LiberoLowdimWrapper):
            return cur
        cur = getattr(cur, "env", None)
    return None


# ---------------------------------------------------------------------------
# Environment creation
# ---------------------------------------------------------------------------

def make_env(env_name, seed=0, enable_render=False, enable_render_images=False, subgoal_cfg=None):
    """Create a LIBERO environment. subgoal_cfg: see docs/DESIGN.md (None = no subgoal tracking)."""
    subgoal_cfg = SG.parse_subgoal_cfg(env_name, subgoal_cfg)
    goal_obs = subgoal_cfg['goal_obs'] if subgoal_cfg is not None else None
    if subgoal_cfg is not None and subgoal_cfg['reward_mode'] == 'dense_parity':
        raise ValueError("reward_mode='dense_parity' is PushT-only")
    cfg = _get_task_config(env_name)
    dataset_path = _get_dataset_path(env_name)
    env_meta = _get_env_meta(dataset_path)
    bddl_path = _get_bddl_path(env_name)

    env_kwargs = dict(env_meta.get("env_kwargs", {}))
    env_kwargs.pop("env_name", None)
    env_kwargs["has_renderer"] = False
    env_kwargs["has_offscreen_renderer"] = enable_render or enable_render_images
    env_kwargs["use_camera_obs"] = False
    env_kwargs["bddl_file_name"] = bddl_path

    import robosuite  # noqa: F401
    from libero.libero.envs import TASK_MAPPING

    problem_name = env_meta.get("problem_name", "libero_kitchen_tabletop_manipulation")
    task_cls = TASK_MAPPING.get(problem_name)
    if task_cls is None:
        raise ValueError(
            f"LIBERO task '{problem_name}' not found in TASK_MAPPING. "
            f"Available: {list(TASK_MAPPING.keys())}"
        )

    raw_env = task_cls(
        robots=env_kwargs.pop("robots", ["Panda"]),
        controller_configs=env_kwargs.pop("controller_configs", None),
        **{k: v for k, v in env_kwargs.items()
           if k not in ("camera_names", "camera_heights", "camera_widths",
                        "camera_depths", "camera_segmentations", "reward_shaping")}
    )
    raw_env.reset()

    init_states = _get_init_states(env_name)

    her_kwargs = {}
    if goal_obs == HER_GC_MODE:
        spec = get_her_spec(env_name)
        her_kwargs = dict(
            her_goal_indices=spec['goal_indices'],
            her_pos_blocks=spec['pos_blocks'],
            her_bool_dims=spec['bool_dims'],
            her_pos_threshold=subgoal_cfg['her_pos_threshold'],
            her_goal_bank=_load_her_goal_bank(env_name),
            her_bc_goal=subgoal_cfg['her_bc_goal'],
        )
    elif goal_obs == HER_LATENT_GC_MODE:
        from vadar.utils.datasets import LatentGoalEncoder
        assert subgoal_cfg['her_encoder_path'], "'latent' goal_obs requires her_encoder_path"
        encoder = LatentGoalEncoder(subgoal_cfg['her_encoder_path'])
        finals = _load_her_goal_bank(env_name, full_state=True)
        her_kwargs = dict(
            her_encoder=encoder,
            her_goal_bank=encoder(finals),
            her_bc_goal=subgoal_cfg['her_bc_goal'],
        )

    env = LiberoLowdimWrapper(
        raw_env,
        obs_keys=cfg['obs_keys'],
        extra_obs_fn=cfg['extra_obs_fn'],
        max_episode_length=cfg['max_episode_length'],
        subgoal_checks=cfg['subgoal_checks'],
        init_states=init_states,
        enable_render=enable_render,
        enable_render_images=enable_render_images,
        subgoal_cfg=subgoal_cfg,
        **her_kwargs,
    )
    env.seed(seed)
    return env


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def get_dataset(env, env_name):
    """Load the offline BC dataset from the processed HDF5.

    With subgoal tracking on (the wrapper has a subgoal_cfg), adds post-action
    `subgoal_dones` (N, G) derived from the hdf5's per-state `subtask_status`
    (see _post_action_labels), appends the goal suffix when goal_obs is set
    (obs: status before the action, next_obs: status after; HER modes append
    the goal as in the fork) and relabels rewards for reward_mode hgc/shaped.
    """
    dataset_path = _get_dataset_path(env_name)

    lowdim_wrapper = _unwrap_lowdim_wrapper(env)
    subgoal_cfg = getattr(lowdim_wrapper, "subgoal_cfg", None)
    goal_obs = subgoal_cfg['goal_obs'] if subgoal_cfg is not None else None
    status_dim = subgoal_cfg['num_subgoals'] if subgoal_cfg is not None else 0
    if subgoal_cfg is not None and subgoal_cfg['dag'] is not None:
        # The stored labels use the sequential ratchet; a DAG would make them disagree with the online env.
        raise ValueError("LIBERO subgoal labels are sequential; subgoal_cfg['dag'] must be None")
    her_mode = goal_obs in HER_MODES
    her_latent = goal_obs == HER_LATENT_GC_MODE
    if her_mode:
        from vadar.utils.datasets import her_reward
        her_spec = get_her_spec(env_name)
        her_thresh = float(subgoal_cfg['her_pos_threshold'])
        her_bc_goal = subgoal_cfg['her_bc_goal']
        her_encoder = getattr(lowdim_wrapper, 'her_encoder', None)
        if her_latent:
            assert her_encoder is not None, "'latent' dataset loading requires the wrapper's her_encoder"
        print(f"[HER dataset] mode={goal_obs} bc_goal={her_bc_goal}")

    with h5py.File(dataset_path, "r") as f:
        demos = list(f["data"].keys())
        inds = np.argsort([int(d.split('_')[1]) for d in demos])
        demos = [demos[i] for i in inds]

        num_timesteps = sum(f[f"data/{d}/actions"].shape[0] for d in demos)
        task_name = _get_task_name(env_name)
        print(f"LIBERO [{task_name}] dataset: {len(demos)} demos, {num_timesteps} timesteps "
              f"(goal_obs={goal_obs}, num_subgoals={status_dim})")

        observations, actions, next_observations = [], [], []
        terminals, rewards, masks = [], [], []
        subgoal_dones, episode_starts = [], []

        for d in demos:
            obs = np.array(f[f"data/{d}/obs/{DATASET_OBS_KEY}"]).astype(np.float32)
            next_obs = np.array(f[f"data/{d}/next_obs/{DATASET_OBS_KEY}"]).astype(np.float32)
            a = np.array(f[f"data/{d}/actions"]).astype(np.float32)
            r = np.array(f[f"data/{d}/rewards"]).astype(np.float32)
            dones = np.array(f[f"data/{d}/dones"]).astype(np.float32)

            if subgoal_cfg is not None:
                if "subtask_status" not in f[f"data/{d}"]:
                    raise AssertionError(f"demo {d}: subgoal labels require `subtask_status` in the HDF5 "
                                         f"(see scripts/extract_libero_dataset.py in the QIPM fork).")
                # Stored (pre-action) per-state status; truncate in case of off-by-one.
                ep_cumulative = np.array(f[f"data/{d}/subtask_status"]).astype(np.int8)[:len(a), :status_dim]
                subgoal_dones.append(_post_action_labels(ep_cumulative))

            if her_mode:
                # HER BC labeling. Goal value of a frame j:
                #   'state'  -> next_obs[j][her_goal_indices]
                #   'latent' -> encoder(next_obs[j])
                # bc_goal='final': every frame gets the demo's final state.
                # bc_goal='next': segment labeling — each frame gets the demo's
                # own NEXT-subtask completion representative (mid-interval
                # frame, not the degenerate completion boundary), so BC teaches
                # one increment at a time. Uses the stored (pre-action) status,
                # as in the fork. Rewards recomputed per-frame in goal space
                # ({-1, 0}); masks all 1 (truncation semantics).
                T_ep = len(obs)

                def goal_of(frame_idx):
                    if her_latent:
                        return her_encoder(next_obs[frame_idx]).astype(np.float32)
                    return next_obs[frame_idx][her_spec['goal_indices']]

                ts = [int(np.argmax(ep_cumulative[:, g])) if ep_cumulative[:, g].max() > 0
                      else None for g in range(status_dim)]
                use_next = her_bc_goal == 'next' and all(t is not None for t in ts)
                if use_next:
                    reps = []
                    for g in range(status_dim):
                        rep_idx = (T_ep - 1 if g == status_dim - 1
                                   else (ts[g] + ts[g + 1]) // 2)
                        reps.append(goal_of(rep_idx))
                    n_done = np.minimum(ep_cumulative.sum(axis=1), status_dim - 1)
                    goals = np.stack([reps[int(k)] for k in n_done], axis=0)
                else:
                    goals = np.tile(goal_of(T_ep - 1)[None], (T_ep, 1))
                goals = goals.astype(obs.dtype, copy=False)

                if her_latent:
                    succ = her_encoder.success(her_encoder(next_obs), goals)
                else:
                    achieved = next_obs[:, her_spec['goal_indices']]
                    succ = her_reward(achieved, goals, her_spec['pos_blocks'],
                                      her_spec['bool_dims'], her_thresh)
                r = succ - 1.0
                obs = np.concatenate([obs, goals], axis=-1)
                next_obs = np.concatenate([next_obs, goals], axis=-1)

            observations.append(obs)
            actions.append(a)
            next_observations.append(next_obs)
            rewards.append(r)
            terminals.append(dones)
            masks.append(np.ones_like(dones) if her_mode else 1.0 - dones)
            ep_starts = np.zeros(len(a), dtype=bool)
            ep_starts[0] = True
            episode_starts.append(ep_starts)

    fields = dict(
        observations=np.concatenate(observations, axis=0),
        actions=np.concatenate(actions, axis=0),
        rewards=np.concatenate(rewards, axis=0),
        terminals=np.concatenate(terminals, axis=0),
        masks=np.concatenate(masks, axis=0),
        next_observations=np.concatenate(next_observations, axis=0),
    )
    if subgoal_cfg is not None:
        subgoal_dones = np.concatenate(subgoal_dones, axis=0)
        episode_starts = np.concatenate(episode_starts, axis=0)
        fields['subgoal_dones'] = subgoal_dones
        if subgoal_cfg['reward_mode'] == 'hgc':
            fields['rewards'] = SG.relabel_rewards(
                'hgc', fields['rewards'] - 1.0, subgoal_dones, episode_starts, first_step_bump=True)
        elif subgoal_cfg['reward_mode'] == 'shaped':
            fields['rewards'] = SG.relabel_rewards('shaped', fields['rewards'], subgoal_dones, episode_starts)
        if goal_obs in SG.GOAL_OBS_MODES and goal_obs is not None:
            obs_goal, next_goal = SG.goal_suffixes(subgoal_dones, episode_starts, goal_obs)
            fields['observations'] = np.concatenate([fields['observations'], obs_goal], axis=-1)
            fields['next_observations'] = np.concatenate([fields['next_observations'], next_goal], axis=-1)
    return Dataset.create(**fields)


# ---------------------------------------------------------------------------
# Gymnasium wrapper
# ---------------------------------------------------------------------------

class LiberoLowdimWrapper(gym.Env):
    """Wraps a raw LIBERO/robosuite env as a gymnasium env. Mirrors RoboCasaLowdimWrapper.

    With subgoal_cfg set, every step ratchets the cumulative subgoal dones
    (sequential), sets info['subgoal_dones'] and applies reward_mode: sparse =
    raw {0,1} env reward; hgc = (r - 1) + 5 on a step where a subgoal newly
    completes; shaped = shaped_reward(subgoal_dones). HER goal_obs modes
    ('state'/'latent') replace the reward with the goal-space reward.
    """

    def __init__(
        self,
        env,
        obs_keys,
        extra_obs_fn=None,
        max_episode_length=600,
        render_hw=(256, 256),
        render_camera_name="agentview",
        subgoal_checks=None,
        init_states=None,
        enable_render=False,
        enable_render_images=False,
        subgoal_cfg=None,
        her_goal_indices=None,
        her_pos_blocks=(),
        her_bool_dims=(),
        her_pos_threshold=0.05,
        her_goal_bank=None,
        her_encoder=None,
        her_bc_goal='final',
    ):
        self.env = env
        self.obs_keys = list(obs_keys)
        self.extra_obs_fn = extra_obs_fn
        self.max_episode_length = max_episode_length
        self.render_hw = render_hw
        self.render_camera_name = render_camera_name
        self.video_writer = None
        self.subgoal_checks = subgoal_checks if subgoal_checks is not None else []
        self.init_states = init_states
        self.enable_render = enable_render
        self.enable_render_images = enable_render_images

        # Subgoal tracking (VaDaR contract)
        self.subgoal_cfg = subgoal_cfg
        self.num_subgoals = subgoal_cfg['num_subgoals'] if subgoal_cfg is not None else 0
        self.goal_obs = subgoal_cfg['goal_obs'] if subgoal_cfg is not None else None
        self.reward_mode = subgoal_cfg['reward_mode'] if subgoal_cfg is not None else 'sparse'
        self._prereqs = (SG.prerequisites(self.num_subgoals, subgoal_cfg['dag'], 'sequential')
                         if subgoal_cfg is not None else None)
        self._shaped_reward = SG.get_shaped_reward_fn() if self.reward_mode == 'shaped' else None

        # HER mode fields. _desired_goal must exist before the
        # _get_state_obs() call below (it runs during __init__).
        self.her_bc_goal = her_bc_goal
        self.her_encoder = her_encoder
        if self.goal_obs == HER_GC_MODE:
            if her_goal_indices is None or her_goal_bank is None:
                raise ValueError("'state' mode requires her_goal_indices and her_goal_bank")
            self.her_goal_indices = np.asarray(her_goal_indices, dtype=np.int64)
            self.her_pos_blocks = tuple(her_pos_blocks)
            self.her_bool_dims = tuple(her_bool_dims)
            self.her_pos_threshold = float(her_pos_threshold)
            self.her_goal_bank = np.asarray(her_goal_bank, dtype=np.float32)
            self.her_goal_dim = int(len(self.her_goal_indices))
            self._desired_goal = np.zeros(self.her_goal_dim, dtype=np.float32)
        elif self.goal_obs == HER_LATENT_GC_MODE:
            if her_encoder is None or her_goal_bank is None:
                raise ValueError("'latent' mode requires her_encoder and her_goal_bank")
            self.her_goal_indices = None
            self.her_goal_bank = np.asarray(her_goal_bank, dtype=np.float32)
            self.her_goal_dim = int(her_encoder.embed_dim)
            self._desired_goal = np.zeros(self.her_goal_dim, dtype=np.float32)
        else:
            self.her_goal_indices = None
            self.her_goal_dim = 0
            self._desired_goal = None

        self.env_step = 0
        self.n_episodes = 0
        self.t = 0
        self.episode_return = 0
        self.episode_length = 0
        self.goal_done = None

        # Action space
        low = np.full(env.action_dim, fill_value=-1.0)
        high = np.full(env.action_dim, fill_value=1.0)
        self.action_space = Box(low=low, high=high, shape=low.shape, dtype=np.float32)

        # Observation space
        state_example = self._get_state_obs()
        print(f"[LIBERO] Obs dim: {state_example.shape[0]} "
              f"(goal_obs={self.goal_obs}, num_subgoals={self.num_subgoals}, "
              f"render_images={self.enable_render_images})")
        low_obs = np.full_like(state_example, fill_value=-np.inf)
        high_obs = np.full_like(state_example, fill_value=np.inf)
        state_space = Box(low=low_obs, high=high_obs, shape=low_obs.shape, dtype=np.float32)
        if self.enable_render_images:
            image_space = Box(
                low=0, high=255,
                shape=(self.render_hw[0], self.render_hw[1], 3),
                dtype=np.uint8,
            )
            self.observation_space = gym.spaces.Dict({
                "state": state_space,
                "image": image_space,
            })
        else:
            self.observation_space = state_space

    def _current_goal_vector(self):
        if self.goal_obs is None:
            return np.zeros((0,), dtype=np.float32)
        if self.goal_obs in HER_MODES:
            return self._desired_goal
        if self.goal_done is None:
            cumulative = np.zeros(self.num_subgoals, dtype=np.int8)
        else:
            cumulative = np.asarray(self.goal_done, dtype=np.int8)
        return SG.encode_goal_vector(cumulative, self.goal_obs).astype(np.float32)

    def _get_state_obs(self):
        raw_obs = self.env._get_observations(force_update=True)
        parts = [np.asarray(raw_obs[k]).flatten() for k in self.obs_keys]
        if self.extra_obs_fn is not None:
            parts.append(np.asarray(self.extra_obs_fn(self.env)).flatten())
        if self.goal_obs is not None:
            parts.append(self._current_goal_vector())
        return np.concatenate(parts, axis=0).astype(np.float32)

    def _get_image_obs(self):
        h, w = self.render_hw
        if hasattr(self.env, 'sim'):
            return self.env.sim.render(
                camera_name=self.render_camera_name,
                height=h, width=w,
            )[::-1].astype(np.uint8)
        return np.zeros((h, w, 3), dtype=np.uint8)

    def get_observation(self):
        state = self._get_state_obs()
        if self.enable_render_images:
            return {"state": state, "image": self._get_image_obs()}
        return state

    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed=seed)
        else:
            np.random.seed()

    def reset(self, options=None, **kwargs):
        self.t = 0
        self.episode_return = 0
        self.episode_length = 0
        self.n_episodes += 1
        # All-false at reset (a subgoal completing on the first step bumps the hgc reward).
        self.goal_done = [False] * self.num_subgoals if self.subgoal_cfg is not None else None

        if self.video_writer is not None:
            self.video_writer.close()
            self.video_writer = None
        if options is None:
            options = {}
        if "video_path" in options:
            self.video_writer = imageio.get_writer(options["video_path"], fps=30)

        new_seed = options.get("seed", None)
        if new_seed is not None:
            self.seed(seed=new_seed)

        self.env.reset()
        if self.init_states is not None and len(self.init_states) > 0:
            idx = np.random.randint(0, len(self.init_states))
            self.env.sim.set_state_from_flattened(self.init_states[idx])
            self.env.sim.forward()

        if self.goal_obs in HER_MODES:
            g_idx = np.random.randint(0, len(self.her_goal_bank))
            self._desired_goal = self.her_goal_bank[g_idx].copy()

        return self.get_observation(), {}

    def step(self, action):
        raw_obs, reward, done, info = self.env.step(action)

        if self.subgoal_cfg is not None:
            prev_goal_done = list(self.goal_done)
            self._update_goal_done()
            info['subgoal_dones'] = np.asarray(self.goal_done, dtype=np.float32)
            if self.reward_mode == 'hgc':
                reward = SG.hgc_reward(reward - 1.0, prev_goal_done, self.goal_done)
            elif self.reward_mode == 'shaped':
                reward = float(self._shaped_reward(info['subgoal_dones']))

        parts = [np.asarray(raw_obs[k]).flatten() for k in self.obs_keys]
        if self.extra_obs_fn is not None:
            parts.append(np.asarray(self.extra_obs_fn(self.env)).flatten())
        base_obs = np.concatenate(parts, axis=0).astype(np.float32)

        if self.goal_obs == HER_GC_MODE:
            # Goal-space reward replaces the env reward (Fetch convention:
            # {-1, 0}); must be set before episode_return accumulation below.
            from vadar.utils.datasets import her_reward
            achieved = base_obs[self.her_goal_indices]
            goal_success = her_reward(
                achieved, self._desired_goal,
                self.her_pos_blocks, self.her_bool_dims, self.her_pos_threshold,
            )
            reward = float(goal_success) - 1.0
            info['goal_success'] = int(goal_success)
        elif self.goal_obs == HER_LATENT_GC_MODE:
            goal_success = self.her_encoder.success(
                self.her_encoder(base_obs), self._desired_goal)
            reward = float(goal_success) - 1.0
            info['goal_success'] = int(goal_success)

        if self.goal_obs is not None:
            state_obs = np.concatenate(
                [base_obs, self._current_goal_vector()], axis=0).astype(np.float32)
        else:
            state_obs = base_obs
        if self.enable_render_images:
            obs = {"state": state_obs, "image": self._get_image_obs()}
        else:
            obs = state_obs

        if self.video_writer is not None:
            video_img = self.render(mode="rgb_array")
            self.video_writer.append_data(video_img)

        self.t += 1
        self.env_step += 1
        self.episode_return += reward
        self.episode_length += 1

        check_success_result = self.env._check_success()
        info["success"] = int(check_success_result)
        if info["success"]:
            done = True
            info['return'] = self.episode_return
            info['length'] = self.episode_length

        if done:
            if self.goal_obs in HER_MODES:
                # SB3/Fetch convention: every episode end is a truncation
                # (masks stay 1), so the critic bootstraps through it; goal
                # achievement itself never ends the episode.
                return obs, reward, False, True, info
            return obs, reward, True, False, info
        if self.t >= self.max_episode_length:
            return obs, reward, False, True, info
        return obs, reward, False, False, info

    def render(self, mode="rgb_array"):
        h, w = self.render_hw
        if mode == "rgb_array" and hasattr(self.env, 'sim'):
            return self.env.sim.render(
                camera_name=self.render_camera_name,
                height=h, width=w,
            )[::-1]
        return np.zeros((h, w, 3), dtype=np.uint8)

    def get_episode_info(self):
        return {"return": self.episode_return, "length": self.episode_length}

    def get_info(self):
        return {"env_step": self.env_step, "n_episodes": self.n_episodes}

    def _update_goal_done(self):
        """Sticky/cumulative (sequential) ratchet of self.goal_done from the current env state."""
        if self.goal_done is None:
            self.goal_done = [False] * self.num_subgoals
        raw = [fn(self.env) for fn in self.subgoal_checks[:self.num_subgoals]]
        self.goal_done = SG.ratchet(self.goal_done, raw, self._prereqs)
