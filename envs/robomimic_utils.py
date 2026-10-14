from os.path import expanduser
import os
from collections import deque

import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Dict
import imageio
import h5py

import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic import DATASET_REGISTRY
from vadar.utils.datasets import Dataset
from envs import subgoal_utils as SG

# HER 'state' goal suffix (object-state goals, SB3-style); see get_her_spec.
HER_GC_MODE = 'state'


# Per-task observation key configuration
# image_size: (H, W) matching --camera_height/--camera_width used in dataset_states_to_obs.py
TASK_CONFIG = {
    'lift': {
        'low_dim_keys': ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos', 'object'],
        'proprio_keys': ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos'],
        'image_keys': ['agentview_image'],
        'image_size': (96, 96),
    },
    'can': {
        'low_dim_keys': ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos', 'object'],
        'proprio_keys': ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos'],
        'image_keys': ['agentview_image'],
        'image_size': (96, 96),
    },
    'square': {
        'low_dim_keys': ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos', 'object'],
        'proprio_keys': ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos'],
        'image_keys': ['agentview_image'],
        'image_size': (96, 96),
    },
    'tool_hang': {
        'low_dim_keys': ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos', 'object'],
        'proprio_keys': ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos'],
        'image_keys': ['sideview_image'],
        'image_size': (240, 240),
        # HER object-state goal space (indices into the 53-dim concat obs):
        # frame_pos [30:33] + tool_pos [44:47] + frame_is_assembled [51] +
        # tool_on_frame [52]. Layout from robosuite ToolHang._setup_observables
        # sensor order within the 44-dim 'object' block (offset 9).
        'her_goal_indices': [30, 31, 32, 44, 45, 46, 51, 52],
        'her_pos_blocks': [(0, 3), (3, 6)],
        'her_bool_dims': [6, 7],
    },
    'transport': {
        'low_dim_keys': ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos',
                         'robot1_eef_pos', 'robot1_eef_quat', 'robot1_gripper_qpos', 'object'],
        'proprio_keys': ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos',
                         'robot1_eef_pos', 'robot1_eef_quat', 'robot1_gripper_qpos'],
        'image_keys': ['shouldercamera0_image', 'shouldercamera1_image'],
        'image_size': (96, 96),
    },
}


def _get_task_config(env_name):
    """Get per-task observation key configuration."""
    task = env_name.split("-")[0]
    if task not in TASK_CONFIG:
        raise ValueError(f"Unknown task: {task}. Supported: {list(TASK_CONFIG.keys())}")
    return TASK_CONFIG[task]


def is_robomimic_env(env_name):
    """determine if an env is robomimic"""
    try:
        if ("low_dim" not in env_name) and ("image" not in env_name):
            return False
        task, dataset_type, hdf5_type = env_name.split("-")
        return task in ("lift", "can", "square", "transport", "tool_hang") and dataset_type in ("mh", "ph")
    except ValueError:
        return False

# Default initialization for backward compatibility (overridden per-task in make_env/get_dataset)
low_dim_keys = {"low_dim": ('robot0_eef_pos',
    'robot0_eef_quat',
    'robot0_gripper_qpos',
    'object')}
ObsUtils.initialize_obs_modality_mapping_from_dict(low_dim_keys)


def _get_max_episode_length(env_name):
    if env_name.startswith("lift"):
        return 300
    elif env_name.startswith("can"):
        return 300
    elif env_name.startswith("square"):
        return 400
    elif env_name.startswith("transport"):
        return 800
    elif env_name.startswith("tool_hang"):
        return 1000
    else:
        raise ValueError(f"Unsupported environment: {env_name}")


# ---------------------------------------------------------------------------
# HER ('state' goal_obs) helpers
# ---------------------------------------------------------------------------

def get_her_spec(env_name):
    """Single source of truth for the HER object-state goal space of a task."""
    cfg = _get_task_config(env_name)
    if 'her_goal_indices' not in cfg:
        raise ValueError(f"No HER goal spec defined for task {env_name.split('-')[0]!r}")
    goal_indices = np.asarray(cfg['her_goal_indices'], dtype=np.int64)
    return dict(
        goal_indices=goal_indices,
        pos_blocks=tuple(tuple(b) for b in cfg['her_pos_blocks']),
        bool_dims=tuple(cfg['her_bool_dims']),
        goal_dim=int(len(goal_indices)),
    )


def _load_her_goal_bank(env_name):
    """Final achieved goals of all demos -> (num_demos, goal_dim) float32.

    Robomimic stores obs split across keys, so the final concatenated obs is
    assembled from low_dim_keys before indexing.
    """
    spec = get_her_spec(env_name)
    cfg = _get_task_config(env_name)
    dataset_path = _check_dataset_exists(env_name)
    goals = []
    with h5py.File(dataset_path, 'r') as f:
        demos = list(f['data'].keys())
        inds = np.argsort([int(d[5:]) for d in demos])
        for d in [demos[i] for i in inds]:
            final = np.concatenate(
                [np.asarray(f[f'data/{d}/next_obs/{k}'][-1], dtype=np.float32)
                 for k in cfg['low_dim_keys']], axis=0)
            goals.append(final[spec['goal_indices']])
    bank = np.stack(goals, axis=0)
    # Contact-based booleans occasionally flicker under the state replay used
    # to regenerate dataset obs (e.g. tool_on_frame 0 at 6/200 tool_hang demo
    # ends despite live success). All demos succeeded, so the per-dim majority
    # is the true goal value — drop finals that deviate from it.
    if spec['bool_dims']:
        bool_vals = bank[:, spec['bool_dims']] > 0.5
        majority = np.mean(bool_vals, axis=0) > 0.5
        keep = (bool_vals == majority[None]).all(axis=1)
        if (~keep).any():
            print(f"[HER] goal bank: dropped {int((~keep).sum())}/{len(bank)} demo finals "
                  f"with replay-flickered booleans")
            bank = bank[keep]
    return bank


# ---------------------------------------------------------------------------
# Subgoal checks (order must match envs/subgoals.py)
# ---------------------------------------------------------------------------

def _check_grasp_frame(base_env, lift_threshold=0.02):
    """ToolHang: both fingerpads grasp the frame and it is lifted above the table."""
    gripper = base_env.robots[0].gripper["right"]
    if not base_env._check_grasp(gripper, base_env.frame.contact_geoms):
        return False
    frame_body_id = base_env.sim.model.body_name2id(base_env.frame.root_body)
    frame_z = base_env.sim.data.body_xpos[frame_body_id][2]
    table_z = base_env.model.mujoco_arena.table_offset[2]
    return frame_z > (table_z + lift_threshold)


def _check_grasp_wrench(base_env, lift_threshold=0.02):
    """ToolHang: both fingerpads grasp the wrench (tool) and it is lifted above the table."""
    gripper = base_env.robots[0].gripper["right"]
    if not base_env._check_grasp(gripper, base_env.tool.contact_geoms):
        return False
    tool_body_id = base_env.sim.model.body_name2id(base_env.tool.root_body)
    tool_z = base_env.sim.data.body_xpos[tool_body_id][2]
    table_z = base_env.model.mujoco_arena.table_offset[2]
    return tool_z > (table_z + lift_threshold)


def _check_lid_removed(base_env):
    """Transport: the lid handle moved left of its start position (lid removed)."""
    lid_pos = base_env.transport.lid_handle_pos
    initial_lid_x = 0.2
    left_threshold = 0.08
    return lid_pos[0] < (initial_lid_x - left_threshold)


def _check_payload_grasped(base_env):
    """Transport: robot 0 grasps the payload."""
    robot0 = base_env.robots[0]
    if hasattr(robot0, 'gripper'):
        gripper = robot0.gripper
        if isinstance(gripper, dict):
            gripper = list(gripper.values())[0]
        return base_env._check_grasp(gripper, base_env.transport.payload.contact_geoms)
    return False


def _check_payload_lifted(base_env, lift_threshold=0.1):
    """Transport: the payload is lifted above the table."""
    payload_pos = base_env.transport.payload_pos
    table_z = base_env.table_offsets[0, 2]
    return payload_pos[2] > (table_z + lift_threshold)


def _check_trash_grasped(base_env):
    """Transport: robot 1 grasps the trash."""
    robot1 = base_env.robots[1]
    if hasattr(robot1, 'gripper'):
        gripper = robot1.gripper
        if isinstance(gripper, dict):
            gripper = list(gripper.values())[0]
        return base_env._check_grasp(gripper, base_env.transport.trash.contact_geoms)
    return False


def _check_payload_handed_off(base_env):
    """Transport: robot 1 grasps the payload and robot 0 does not."""
    robot0_grasping = robot1_grasping = False
    for i, robot in enumerate(base_env.robots):
        if not hasattr(robot, 'gripper'):
            continue
        grippers = robot.gripper if isinstance(robot.gripper, dict) else {'default': robot.gripper}
        for _, gripper in grippers.items():
            if base_env._check_grasp(gripper, base_env.transport.payload.contact_geoms):
                if i == 0:
                    robot0_grasping = True
                else:
                    robot1_grasping = True
                break
    return robot1_grasping and not robot0_grasping


def _compute_subtask_status(base_env, goal_done, no_of_subtasks):
    """Sticky update of the cumulative subgoal flags from the current sim state.

    Robomimic subgoals have NO prerequisite enforcement: a subgoal is done once
    its check has held at any step. Returns (updated_goal_done, raw_checks).
    """
    if goal_done is None:
        goal_done = [False] * no_of_subtasks
    else:
        assert len(goal_done) == no_of_subtasks

    if hasattr(base_env, "transport"):
        goals = [
            _check_lid_removed(base_env),                        # 0: LID_REMOVED
            _check_payload_grasped(base_env),                    # 1: PAYLOAD_GRASPED
            _check_payload_lifted(base_env),                     # 2: PAYLOAD_LIFTED
            _check_trash_grasped(base_env),                      # 3: TRASH_GRASPED
            bool(base_env.transport.trash_in_trash_bin),         # 4: TRASH_IN_BIN
            _check_payload_handed_off(base_env),                 # 5: PAYLOAD_HANDED_OFF
            bool(base_env.transport.payload_in_target_bin),      # 6: PAYLOAD_IN_TARGET
        ]
    elif hasattr(base_env, "nuts") and hasattr(base_env, "on_peg"):
        nut_id = getattr(base_env, "nut_id", 0)
        nut_body_id = base_env.sim.model.body_name2id(base_env.nuts[nut_id].root_body)
        nut_pos = base_env.sim.data.body_xpos[nut_body_id]
        peg_body_id = base_env.peg1_body_id if nut_id == 0 else base_env.peg2_body_id
        peg_pos = base_env.sim.data.body_xpos[peg_body_id]
        table_z = base_env.table_offset[2]
        lift_threshold = table_z + 0.0713
        xy_dist = np.linalg.norm(nut_pos[:2] - peg_pos[:2])
        is_grasped = base_env._check_grasp(
            gripper=base_env.robots[0].gripper,
            object_geoms=base_env.nuts[nut_id].contact_geoms,
        )
        goals = [
            is_grasped and (nut_pos[2] < lift_threshold),
            is_grasped and (nut_pos[2] > lift_threshold) and (xy_dist < 0.2),
            bool(base_env._check_success()),
        ]
    elif hasattr(base_env, "tool") and hasattr(base_env, "frame") and hasattr(base_env, "stand"):
        is_frame_grasped = _check_grasp_frame(base_env)
        is_frame_inserted = base_env._check_frame_assembled()
        is_wrench_grasped = _check_grasp_wrench(base_env)
        is_wrench_attached = base_env._check_tool_on_frame()
        if no_of_subtasks == 4:
            goals = [is_frame_grasped, is_frame_inserted, is_frame_inserted and is_wrench_grasped, is_wrench_attached]
        elif no_of_subtasks == 1:
            goals = [is_wrench_attached]
        else:
            raise ValueError("number of subgoals not configured for tool_hang")
    else:
        raise ValueError(f"No subgoal decomposition for env: {type(base_env)}")

    updated = list(goal_done)
    for i in range(min(len(goals), no_of_subtasks)):
        if goals[i]:
            updated[i] = True
    return updated, goals


def _unwrap_robomimic_wrapper(env):
    """Find the Robomimic{Lowdim,Image}Wrapper in a wrapper chain."""
    wrapped = env
    for _ in range(20):
        if isinstance(wrapped, (RobomimicLowdimWrapper, RobomimicImageWrapper)):
            return wrapped
        if not hasattr(wrapped, "env"):
            return None
        wrapped = wrapped.env
    return None


def _infer_post_action_labels(rm_env, states, actions, num_subgoals):
    """Post-action cumulative subgoal labels of one demo, by replaying sim states.

    labels[t] = status after executing actions[t]: the sticky ratchet over the
    states s_1..s_{t+1}, starting from all-false at the reset state (as online).
    s_{t+1} = states[t+1]; the final post-action state is obtained by playing
    the last action from states[T-1] (as robomimic does to build next_obs[-1]).
    """
    T = len(actions)
    labels = np.zeros((T, num_subgoals), dtype=np.float32)
    if T == 0:
        return labels
    states = np.asarray(states)
    assert states.shape[0] == T, f"State/action length mismatch: states={states.shape[0]}, actions={T}"
    base_env = rm_env.env
    goal_done = [False] * num_subgoals
    for t in range(T):
        if t < T - 1:
            rm_env.reset_to({"states": states[t + 1]})
        else:
            if getattr(base_env, "done", False):
                rm_env.reset()
            rm_env.reset_to({"states": states[T - 1]})
            rm_env.step(actions[T - 1])
        goal_done, _ = _compute_subtask_status(base_env, goal_done, num_subgoals)
        labels[t] = np.asarray(goal_done, dtype=np.float32)
    return labels


def _get_subgoal_labels(wrapper, dataset_path, rm_dataset, demos, num_subgoals):
    """(N, G) post-action labels for all demos, cached next to the dataset."""
    num_steps = sum(int(rm_dataset[f"data/{ep}/actions"].shape[0]) for ep in demos)

    def compute():
        print(f"[subgoals] relabelling {len(demos)} robomimic demos by state replay ...")
        out = []
        for i, ep in enumerate(demos):
            out.append(_infer_post_action_labels(
                wrapper.env, rm_dataset[f"data/{ep}/states"][()],
                rm_dataset[f"data/{ep}/actions"][()], num_subgoals))
            if (i + 1) % 50 == 0:
                print(f"[subgoals]   {i + 1}/{len(demos)} demos")
        return np.concatenate(out, axis=0)

    cache_path = SG.label_cache_path(dataset_path, num_subgoals)
    return SG.load_or_compute_labels(cache_path, num_steps, num_subgoals, compute)


def make_env(env_name, seed=0, frame_stack=1, post_success_steps=0, subgoal_cfg=None):
    """
    NOTE: should get_dataset() first, so that the metadata is downloaded before creating the environment

    subgoal_cfg: see docs/DESIGN.md (None = upstream behavior).
    """
    subgoal_cfg = SG.parse_subgoal_cfg(env_name, subgoal_cfg)
    goal_obs = subgoal_cfg['goal_obs'] if subgoal_cfg is not None else None
    if goal_obs == 'latent':
        raise ValueError("'latent' goal_obs is LIBERO-only")
    if subgoal_cfg is not None and subgoal_cfg['reward_mode'] == 'dense_parity':
        raise ValueError("reward_mode='dense_parity' is PushT-only")
    dataset_path = _check_dataset_exists(env_name)
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)
    max_episode_length = _get_max_episode_length(env_name)
    task_cfg = _get_task_config(env_name)

    if "image" in env_name:
        if goal_obs is not None:
            raise ValueError("goal_obs is supported only for low_dim robomimic envs")
        image_keys = task_cfg['image_keys']
        proprio_keys = tuple(task_cfg['proprio_keys'])
        img_h, img_w = task_cfg['image_size']

        shape_meta = {
            'obs': {
                'rgb': {'shape': (3, img_h, img_w)},
            },
        }

        obs_keys = {"low_dim": list(proprio_keys), 'rgb': image_keys}
        ObsUtils.initialize_obs_modality_mapping_from_dict(obs_keys)

        env = EnvUtils.create_env_from_metadata(
            env_meta=env_meta,
            render=False,
            render_offscreen=False,
            use_image_obs=True,
        )
        env = RobomimicImageWrapper(
            env, shape_meta=shape_meta, image_keys=image_keys,
            proprio_keys=proprio_keys,
            low_dim_keys=tuple(task_cfg['low_dim_keys']),
            max_episode_length=max_episode_length, num_stack=frame_stack,
            post_success_steps=post_success_steps,
            subgoal_cfg=subgoal_cfg,
        )
    else:
        task_low_dim_keys = task_cfg['low_dim_keys']
        ObsUtils.initialize_obs_modality_mapping_from_dict({"low_dim": task_low_dim_keys})

        env = EnvUtils.create_env_from_metadata(
            env_meta=env_meta,
            render=False,
            render_offscreen=False,
        )
        her_kwargs = {}
        if goal_obs == HER_GC_MODE:
            spec = get_her_spec(env_name)
            her_kwargs = dict(
                her_goal_indices=spec['goal_indices'],
                her_pos_blocks=spec['pos_blocks'],
                her_bool_dims=spec['bool_dims'],
                her_pos_threshold=subgoal_cfg['her_pos_threshold'],
                her_goal_bank=_load_her_goal_bank(env_name),
            )
        env = RobomimicLowdimWrapper(env, low_dim_keys=task_low_dim_keys, max_episode_length=max_episode_length, post_success_steps=post_success_steps,
                                     subgoal_cfg=subgoal_cfg, **her_kwargs)
    env.seed(seed)
    env.env.hard_reset = False
    return env

def _check_dataset_exists(env_name):
    # enforce that the dataset exists
    task, dataset_type, hdf5_type = env_name.split("-")
    if hdf5_type == "image":
        file_name = "image_v15.hdf5"
    elif dataset_type == "mg":
        file_name = "low_dim_sparse_v15.hdf5"
    else:
        file_name = f"low_dim_v15.hdf5"
    dataset_root = os.environ.get("ROBOMIMIC_DATASET_ROOT")
    repo_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "robomimic")
    if dataset_root is None and os.path.isdir(repo_root):
        # Datasets fetched with scripts/fetch_datasets.sh live in <repo>/data/robomimic.
        dataset_root = repo_root

    if dataset_root is None:
        username = os.environ.get("USER")
        if username:
            dataset_root = os.path.join("/data/user_data", username, "robomimic")
        else:
            dataset_root = os.path.join(expanduser("~"), ".robomimic")

    dataset_path = os.path.join(
        dataset_root,
        task,
        dataset_type,
        file_name,
    )
    # assert os.path.exists(dataset_path)

    return dataset_path

def get_dataset(env, env_name):
    """Load the offline dataset.

    With subgoal tracking on (the env wrapper has a subgoal_cfg), adds
    `subgoal_dones` (N, G): post-action cumulative labels from replaying the
    stored sim states (cached next to the hdf5), appends the goal suffix to
    observations when goal_obs is set (obs: status before the action,
    next_obs: status after), and relabels rewards for reward_mode hgc/shaped
    to match the online wrapper.
    """
    dataset_path = _check_dataset_exists(env_name)
    task_cfg = _get_task_config(env_name)

    rm_dataset = h5py.File(dataset_path, "r")
    demos = list(rm_dataset["data"].keys())
    num_demos = len(demos)
    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]

    num_timesteps = 0
    for ep in demos:
        num_timesteps += int(rm_dataset[f"data/{ep}/actions"].shape[0])

    print(f"the size of the dataset is {num_timesteps}")

    # Subgoal tracking (VaDaR contract)
    wrapper = _unwrap_robomimic_wrapper(env)
    subgoal_cfg = getattr(wrapper, 'subgoal_cfg', None)
    goal_obs = subgoal_cfg['goal_obs'] if subgoal_cfg is not None else None
    her_mode = goal_obs == HER_GC_MODE
    subgoal_dones = None
    if subgoal_cfg is not None:
        subgoal_dones = _get_subgoal_labels(wrapper, dataset_path, rm_dataset, demos,
                                            subgoal_cfg['num_subgoals'])
    if her_mode:
        from vadar.utils.datasets import her_reward
        her_spec = get_her_spec(env_name)
        her_thresh = float(subgoal_cfg['her_pos_threshold'])
    episode_starts = []

    # Check if this is an image environment
    is_image_env = "image" in env_name

    # Use per-task keys
    image_keys = task_cfg['image_keys']
    proprio_keys = task_cfg['proprio_keys']
    task_low_dim_keys = task_cfg['low_dim_keys']

    # Determine image frame-stack once (image envs only)
    num_stack = 1
    if is_image_env:
        current_env = env
        while hasattr(current_env, 'env'):
            if hasattr(current_env, 'num_stack'):
                num_stack = current_env.num_stack
                break
            current_env = current_env.env
        if hasattr(current_env, 'num_stack'):
            num_stack = current_env.num_stack

    # data holders — small fields use lists; large image fields are preallocated
    # below (uint8, written per-episode) so we never hold list + concatenated copy
    # simultaneously. For tool_hang (240x240) the float32 list+concat path peaked
    # at ~190 GB and OOM-killed the process.
    observations = []      # always proprioceptive state
    actions = []
    next_observations = []  # always proprioceptive state
    terminals = []
    rewards = []
    masks = []
    all_full_states = []    # full low-dim state for image envs (includes object pose)
    all_next_full_states = []

    images_arr = None       # preallocated (num_timesteps, H, W, 3*num_cams*num_stack) uint8
    next_images_arr = None
    img_cursor = 0

    # go through and add to the data holder
    for ep in demos:
        a = np.array(rm_dataset["data/{}/actions".format(ep)])
        dones = np.array(rm_dataset["data/{}/dones".format(ep)])
        r = np.array(rm_dataset["data/{}/rewards".format(ep)])
        # Transform rewards from [0, 1] to [-1, 0]
        r = r - 1.0

        if is_image_env:
            # --- Load images (keep uint8; encoders normalize /255 internally) ---
            img_obs_parts = [np.asarray(rm_dataset[f"data/{ep}/obs/{key}"]) for key in image_keys]
            img_next_parts = [np.asarray(rm_dataset[f"data/{ep}/next_obs/{key}"]) for key in image_keys]
            img_obs = np.concatenate(img_obs_parts, axis=3) if len(img_obs_parts) > 1 else img_obs_parts[0]
            img_next = np.concatenate(img_next_parts, axis=3) if len(img_next_parts) > 1 else img_next_parts[0]
            del img_obs_parts, img_next_parts

            if num_stack > 1:
                all_frames = np.concatenate([img_obs, img_next[-1:]], axis=0)
                pad = np.repeat(all_frames[:1], num_stack - 1, axis=0)
                padded_frames = np.concatenate([pad, all_frames], axis=0)
                stacks = [padded_frames[i : i + img_obs.shape[0] + 1] for i in range(num_stack)]
                stacked_frames = np.concatenate(stacks, axis=3)
                img_obs = stacked_frames[:-1]
                img_next = stacked_frames[1:]
                del all_frames, pad, padded_frames, stacks, stacked_frames

            n_ep = img_obs.shape[0]
            if images_arr is None:
                images_arr = np.empty((num_timesteps,) + img_obs.shape[1:], dtype=img_obs.dtype)
                next_images_arr = np.empty((num_timesteps,) + img_next.shape[1:], dtype=img_next.dtype)
            images_arr[img_cursor:img_cursor + n_ep] = img_obs
            next_images_arr[img_cursor:img_cursor + n_ep] = img_next
            img_cursor += n_ep
            del img_obs, img_next

            # --- Load proprioceptive state ---
            state_obs = np.concatenate(
                [np.array(rm_dataset[f"data/{ep}/obs/{k}"]) for k in proprio_keys], axis=-1)
            state_next = np.concatenate(
                [np.array(rm_dataset[f"data/{ep}/next_obs/{k}"]) for k in proprio_keys], axis=-1)
            observations.append(state_obs.astype(np.float32))
            next_observations.append(state_next.astype(np.float32))

            # --- Load full low-dim state (for state-based critic) ---
            full_state_obs = np.concatenate(
                [np.array(rm_dataset[f"data/{ep}/obs/{k}"]) for k in task_low_dim_keys], axis=-1)
            full_state_next = np.concatenate(
                [np.array(rm_dataset[f"data/{ep}/next_obs/{k}"]) for k in task_low_dim_keys], axis=-1)
            all_full_states.append(full_state_obs.astype(np.float32))
            all_next_full_states.append(full_state_next.astype(np.float32))
        else:
            # Load low-dimensional observations
            obs, next_obs = [], []
            for k in task_low_dim_keys:
                obs.append(np.array(rm_dataset[f"data/{ep}/obs/{k}"]))
            for k in task_low_dim_keys:
                next_obs.append(np.array(rm_dataset[f"data/{ep}/next_obs/{k}"]))
            obs = np.concatenate(obs, axis=-1)
            next_obs = np.concatenate(next_obs, axis=-1)
            if her_mode:
                # HER: each demo's goal is its own final achieved state (demos all
                # succeed); rewards recomputed in goal space ({-1, 0}).
                achieved = next_obs[:, her_spec['goal_indices']]
                desired = achieved[-1]
                succ = her_reward(achieved, desired[None], her_spec['pos_blocks'],
                                  her_spec['bool_dims'], her_thresh)
                r = succ - 1.0
                goal_tile = np.tile(desired[None], (len(obs), 1)).astype(obs.dtype, copy=False)
                obs = np.concatenate([obs, goal_tile], axis=-1)
                next_obs = np.concatenate([next_obs, goal_tile], axis=-1)
            observations.append(obs.astype(np.float32))
            next_observations.append(next_obs.astype(np.float32))

        actions.append(a.astype(np.float32))
        rewards.append(r.astype(np.float32))
        terminals.append(dones.astype(np.float32))
        # HER: masks all 1 (truncation semantics, as the online 'state' wrapper).
        masks.append(np.ones_like(dones, dtype=np.float32) if her_mode
                     else 1.0 - dones.astype(np.float32))
        ep_starts = np.zeros(len(a), dtype=bool)
        ep_starts[0] = True
        episode_starts.append(ep_starts)

    dataset_fields = dict(
        observations=np.concatenate(observations, axis=0),
        actions=np.concatenate(actions, axis=0),
        rewards=np.concatenate(rewards, axis=0),
        terminals=np.concatenate(terminals, axis=0),
        masks=np.concatenate(masks, axis=0),
        next_observations=np.concatenate(next_observations, axis=0),
    )
    if is_image_env:
        assert img_cursor == num_timesteps, f"image count {img_cursor} != {num_timesteps}"
        dataset_fields['images'] = images_arr
        dataset_fields['next_images'] = next_images_arr
        dataset_fields['full_states'] = np.concatenate(all_full_states, axis=0)
        dataset_fields['next_full_states'] = np.concatenate(all_next_full_states, axis=0)

    if subgoal_cfg is not None:
        episode_starts = np.concatenate(episode_starts, axis=0)
        dataset_fields['subgoal_dones'] = subgoal_dones
        dataset_fields['rewards'] = SG.relabel_rewards(
            subgoal_cfg['reward_mode'], dataset_fields['rewards'], subgoal_dones, episode_starts,
            first_step_bump=True)
        if goal_obs in SG.GOAL_OBS_MODES and goal_obs is not None:
            obs_goal, next_goal = SG.goal_suffixes(subgoal_dones, episode_starts, goal_obs)
            dataset_fields['observations'] = np.concatenate(
                [dataset_fields['observations'], obs_goal], axis=-1)
            dataset_fields['next_observations'] = np.concatenate(
                [dataset_fields['next_observations'], next_goal], axis=-1)

    return Dataset.create(**dataset_fields)


class _SubgoalTrackingMixin:
    """Cumulative subgoal tracking and reward modes shared by the robomimic wrappers.

    Inactive (upstream behavior) when subgoal_cfg is None. Otherwise every step
    sets info['subgoal_dones'] (float32 (G,), sticky, after the step) and the
    reward follows subgoal_cfg['reward_mode']:
      sparse: upstream reward; hgc: (r - 1) + 5 on a step where a subgoal newly
      completes (the fork's GC reward); shaped: shaped_reward(subgoal_dones).
    """

    def _init_subgoal_tracking(self, subgoal_cfg):
        self.subgoal_cfg = subgoal_cfg
        self.num_subgoals = subgoal_cfg['num_subgoals'] if subgoal_cfg is not None else 0
        self.reward_mode = subgoal_cfg['reward_mode'] if subgoal_cfg is not None else 'sparse'
        self._shaped_reward = SG.get_shaped_reward_fn() if self.reward_mode == 'shaped' else None
        self.goal_done = None

    def _reset_subgoal_tracking(self):
        # All-false at reset (so a subgoal completing on the first step bumps, as in the fork).
        self.goal_done = [False] * self.num_subgoals if self.subgoal_cfg is not None else None

    def _update_subgoal_tracking(self, info):
        """Ratchet the dones from the post-step sim state; returns the previous dones."""
        if self.subgoal_cfg is None:
            return None
        prev = list(self.goal_done)
        self.goal_done, _ = _compute_subtask_status(self.env.env, self.goal_done, self.num_subgoals)
        info['subgoal_dones'] = np.asarray(self.goal_done, dtype=np.float32)
        return prev

    def _subgoal_reward(self, env_reward, reward, prev_dones):
        """Reward under reward_mode; `reward` is the upstream reward (returned for 'sparse')."""
        if self.reward_mode == 'hgc':
            return SG.hgc_reward(env_reward - 1.0, prev_dones, self.goal_done)
        if self.reward_mode == 'shaped':
            return float(self._shaped_reward(np.asarray(self.goal_done, dtype=np.float32)))
        return reward


class RobomimicLowdimWrapper(_SubgoalTrackingMixin, gym.Env):
    """
    Environment wrapper for Robomimic environments with state observations.
    Modified from https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/env/robomimic/robomimic_lowdim_wrapper.py
    """
    def __init__(
        self,
        env,
        normalization_path=None,
        low_dim_keys=[
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
            "object",
        ],
        clamp_obs=False,
        init_state=None,
        render_hw=(256, 256),
        render_camera_name="agentview",
        max_episode_length=None,
        post_success_steps=0,
        subgoal_cfg=None,
        her_goal_indices=None,
        her_pos_blocks=(),
        her_bool_dims=(),
        her_pos_threshold=0.05,
        her_goal_bank=None,
    ):
        self.env = env
        self.obs_keys = low_dim_keys
        self.init_state = init_state
        self.render_hw = render_hw
        self.render_camera_name = render_camera_name
        self.video_writer = None
        self.clamp_obs = clamp_obs
        self.max_episode_length = max_episode_length
        self.post_success_steps = post_success_steps
        self.env_step = 0
        self.n_episodes = 0
        self.t_succ = None

        # Subgoal tracking / goal suffix (VaDaR contract)
        self._init_subgoal_tracking(subgoal_cfg)
        self.goal_obs = subgoal_cfg['goal_obs'] if subgoal_cfg is not None else None
        # HER 'state' mode fields; _desired_goal must exist before get_observation() below.
        if self.goal_obs == HER_GC_MODE:
            if her_goal_indices is None or her_goal_bank is None:
                raise ValueError("'state' goal_obs requires her_goal_indices and her_goal_bank")
            self.her_goal_indices = np.asarray(her_goal_indices, dtype=np.int64)
            self.her_pos_blocks = tuple(her_pos_blocks)
            self.her_bool_dims = tuple(her_bool_dims)
            self.her_pos_threshold = float(her_pos_threshold)
            self.her_goal_bank = np.asarray(her_goal_bank, dtype=np.float32)
            self.her_goal_dim = int(len(self.her_goal_indices))
            self._desired_goal = np.zeros(self.her_goal_dim, dtype=np.float32)
        else:
            self.her_goal_indices = None
            self.her_goal_dim = 0
            self._desired_goal = None

        # set up normalization
        self.normalize = normalization_path is not None
        if self.normalize:
            normalization = np.load(normalization_path)
            self.obs_min = normalization["obs_min"]
            self.obs_max = normalization["obs_max"]
            self.action_min = normalization["action_min"]
            self.action_max = normalization["action_max"]

        # setup spaces - use [-1, 1]
        low = np.full(env.action_dimension, fill_value=-1.)
        high = np.full(env.action_dimension, fill_value=1.)
        self.action_space = Box(
            low=low,
            high=high,
            shape=low.shape,
            dtype=low.dtype,
        )
        obs_example = self.get_observation()
        if self.goal_obs is None:
            low = np.full_like(obs_example, fill_value=-1)
            high = np.full_like(obs_example, fill_value=1)
        else:
            low = np.full_like(obs_example, fill_value=-1, dtype=np.float32)
            high = np.full_like(obs_example, fill_value=1, dtype=np.float32)
            if self.goal_obs == HER_GC_MODE:
                # Continuous state goal suffix; bounds are nominal (not enforced).
                low[-self.her_goal_dim:] = -np.inf
                high[-self.her_goal_dim:] = np.inf
            else:
                low[-self.num_subgoals:] = 0.0
                high[-self.num_subgoals:] = 1.0
        self.observation_space = Box(
            low=low,
            high=high,
            shape=low.shape,
            dtype=low.dtype,
        )

    def normalize_obs(self, obs):
        obs = 2 * (
            (obs - self.obs_min) / (self.obs_max - self.obs_min + 1e-6) - 0.5
        )  # -> [-1, 1]
        if self.clamp_obs:
            obs = np.clip(obs, -1, 1)
        return obs

    def unnormalize_action(self, action):
        action = (action + 1) / 2  # [-1, 1] -> [0, 1]
        return action * (self.action_max - self.action_min) + self.action_min

    def _current_goal_vector(self):
        """Goal suffix: the HER desired goal, or the encoded cumulative dones."""
        if self.goal_obs == HER_GC_MODE:
            return self._desired_goal
        cumulative = np.asarray(self.goal_done if self.goal_done is not None
                                else [False] * self.num_subgoals, dtype=np.int8)
        return SG.encode_goal_vector(cumulative, self.goal_obs)

    def _build_observation(self, raw_obs):
        """Concatenate the low-dim keys (normalized if set) and append the goal suffix if any."""
        obs = np.concatenate([raw_obs[key] for key in self.obs_keys], axis=0)
        if self.normalize:
            obs = self.normalize_obs(obs)
        if self.goal_obs is not None:
            goal_vec = self._current_goal_vector().astype(obs.dtype, copy=False)
            obs = np.concatenate([obs, goal_vec], axis=0)
        return obs

    def get_observation(self):
        raw_obs = self.env.get_observation()
        return self._build_observation(raw_obs)

    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed=seed)
        else:
            np.random.seed()

    def reset(self, seed=None, options=None, **kwargs):
        """Reset environment with optional seed for reproducible diverse resets."""
        self.t = 0
        self.t_succ = None
        self.episode_return, self.episode_length = 0, 0
        self.n_episodes += 1
        self._reset_subgoal_tracking()
        if self.video_writer is not None:
            self.video_writer.close()
            self.video_writer = None

        if options is None:
            options = {}

        if "video_path" in options:
            self.video_writer = imageio.get_writer(options["video_path"], fps=30)

        # Prefer explicit seed kwarg (gymnasium standard), fall back to options
        new_seed = seed if seed is not None else options.get("seed", None)
        if self.init_state is not None:
            self.env.reset_to({"states": self.init_state})
        elif new_seed is not None:
            self.seed(seed=new_seed)
            self.env.reset()
        else:
            self.env.reset()

        if self.goal_obs == HER_GC_MODE:
            g_idx = np.random.randint(0, len(self.her_goal_bank))
            self._desired_goal = self.her_goal_bank[g_idx].copy()

        return self.get_observation(), {}

    def step(self, action):
        if self.normalize:
            action = self.unnormalize_action(action)
        raw_obs, env_reward, done, info = self.env.step(action)
        # Transform reward from [0, 1] to [-1, 0]
        reward = env_reward - 1.0
        prev_dones = self._update_subgoal_tracking(info)
        obs = self._build_observation(raw_obs)

        # render if specified
        if self.video_writer is not None:
            video_img = self.render(mode="rgb_array")
            self.video_writer.append_data(video_img)

        # Success / post-success reward shaping
        is_success_step = reward > -0.5

        if self.t_succ is None:
            if is_success_step:
                self.t_succ = self.t
                info["success"] = 1
                reward += 5.0
                if self.post_success_steps <= 0:
                    done = True
            else:
                info["success"] = 0
        else:
            # Post-success phase
            info["success"] = 1
            steps_since_succ = self.t - self.t_succ
            if is_success_step and steps_since_succ <= self.post_success_steps:
                reward += 5.0
            if steps_since_succ >= self.post_success_steps or not is_success_step:
                done = True

        if self.subgoal_cfg is not None:
            reward = self._subgoal_reward(env_reward, reward, prev_dones)
        if self.goal_obs == HER_GC_MODE:
            # HER: the goal-space reward ({-1, 0}) replaces the env reward.
            from vadar.utils.datasets import her_reward
            achieved = np.concatenate([raw_obs[key] for key in self.obs_keys], axis=0)[self.her_goal_indices]
            goal_success = her_reward(
                achieved, self._desired_goal,
                self.her_pos_blocks, self.her_bool_dims, self.her_pos_threshold,
            )
            reward = float(goal_success) - 1.0
            info['goal_success'] = int(goal_success)

        self.t += 1
        self.env_step += 1
        self.episode_return += reward
        self.episode_length += 1

        # Always include return and length when episode ends
        terminated = done
        truncated = self.t >= self.max_episode_length
        if terminated or truncated:
            info['return'] = self.episode_return
            info['length'] = self.episode_length

        if terminated:
            if self.goal_obs == HER_GC_MODE:
                # SB3/Fetch convention: every episode end is a truncation
                # (masks stay 1) so the critic bootstraps through it.
                return obs, reward, False, True, info
            return obs, reward, True, False, info
        if truncated:
            return obs, reward, False, True, info
        return obs, reward, False, False, info

    def render(self, mode="rgb_array"):
        h, w = self.render_hw
        return self.env.render(
            mode=mode,
            height=h,
            width=w,
            camera_name=self.render_camera_name,
        )

    def get_episode_info(self):
        return {"return": self.episode_return, "length": self.episode_length}
    def get_info(self):
        return {"env_step": self.env_step, "n_episodes": self.n_episodes}



class RobomimicImageWrapper(_SubgoalTrackingMixin, gym.Env):
    """Robomimic image environment wrapper.

    Returns dict observations: {'state': proprio_vec, 'image': image_array}
    so that the pipeline can independently route state and image to actor/critic.
    """
    def __init__(
        self,
        env,
        shape_meta: dict,
        normalization_path=None,
        image_keys=[
            "agentview_image",
            "robot0_eye_in_hand_image",
        ],
        proprio_keys=(
            'robot0_eef_pos',
            'robot0_eef_quat',
            'robot0_gripper_qpos',
        ),
        low_dim_keys=None,
        clamp_obs=False,
        init_state=None,
        render_hw=(256, 256),
        render_camera_name="agentview",
        max_episode_length=None,
        num_stack=1,
        post_success_steps=0,
        subgoal_cfg=None,
    ):
        self.env = env
        self._init_subgoal_tracking(subgoal_cfg)
        self.init_state = init_state
        self.has_reset_before = False
        self.render_hw = render_hw
        self.render_camera_name = render_camera_name
        self.video_writer = None
        self.clamp_obs = clamp_obs
        self.max_episode_length = max_episode_length
        self.post_success_steps = post_success_steps
        self.env_step = 0
        self.n_episodes = 0
        self.t = 0
        self.t_succ = None
        self.num_stack = num_stack
        self.frame_stack = deque(maxlen=num_stack)
        self.proprio_keys = proprio_keys
        self.low_dim_keys = low_dim_keys or list(proprio_keys)

        # set up normalization for actions only
        self.normalize = normalization_path is not None
        if self.normalize:
            normalization = np.load(normalization_path)
            self.action_min = normalization["action_min"]
            self.action_max = normalization["action_max"]

        # setup spaces
        low = np.full(env.action_dimension, fill_value=-1)
        high = np.full(env.action_dimension, fill_value=1)
        self.action_space = Box(
            low=low,
            high=high,
            shape=low.shape,
            dtype=low.dtype,
        )
        self.image_keys = image_keys

        # Image observation space
        rgb_shape = shape_meta["obs"]["rgb"]["shape"]  # (3, 96, 96) as (C, H, W)
        channels, height, width = rgb_shape
        num_cameras = len(image_keys)
        full_image_shape = (height, width, channels * num_cameras * num_stack)

        # Proprioception and full state dimensions from a dummy obs
        raw_obs = self.env.get_observation()
        proprio_dim = sum(raw_obs[k].shape[0] for k in self.proprio_keys if k in raw_obs)
        full_state_dim = sum(raw_obs[k].shape[0] for k in self.low_dim_keys if k in raw_obs)

        self.observation_space = Dict({
            'state': Box(low=-np.inf, high=np.inf, shape=(proprio_dim,), dtype=np.float32),
            'image': Box(low=0, high=255, shape=full_image_shape, dtype=np.float32),
            'full_state': Box(low=-np.inf, high=np.inf, shape=(full_state_dim,), dtype=np.float32),
        })

    def unnormalize_action(self, action):
        action = (action + 1) / 2  # [-1, 1] -> [0, 1]
        return action * (self.action_max - self.action_min) + self.action_min

    def get_single_observation(self, raw_obs):
        """Extract image, proprioception, and full low-dim state from raw robomimic observation.

        Returns:
            dict with 'state' (proprio vec), 'image' (HWC float32),
            and 'full_state' (full low-dim state including object pose).
        """
        # Image
        img_parts = []
        for key in self.image_keys:
            if key in raw_obs:
                img_parts.append(raw_obs[key].astype(np.float32))  # (H, W, C)
        image = np.concatenate(img_parts, axis=2) if len(img_parts) > 1 else img_parts[0]

        # Proprioception
        state_parts = [raw_obs[k].astype(np.float32) for k in self.proprio_keys if k in raw_obs]
        state = np.concatenate(state_parts, axis=0)

        # Full low-dim state (includes object pose for state-based critic)
        full_state_parts = [raw_obs[k].astype(np.float32) for k in self.low_dim_keys if k in raw_obs]
        full_state = np.concatenate(full_state_parts, axis=0)

        return {'state': state, 'image': image, 'full_state': full_state}

    def _get_stacked_observation(self):
        assert len(self.frame_stack) == self.num_stack
        frames = list(self.frame_stack)
        stacked_image = np.concatenate([f['image'] for f in frames], axis=-1)
        current_state = frames[-1]['state']  # latest proprioception
        current_full_state = frames[-1]['full_state']  # full low-dim state
        return {'state': current_state, 'image': stacked_image, 'full_state': current_full_state}

    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed=seed)
        else:
            np.random.seed()

    def reset(self, seed=None, options=None, **kwargs):
        """Reset environment with optional seed for reproducible diverse resets."""
        if options is None:
            options = {}

        self.t = 0
        self.t_succ = None
        self.episode_return, self.episode_length = 0, 0
        self.n_episodes += 1
        self._reset_subgoal_tracking()

        if self.video_writer is not None:
            self.video_writer.close()
            self.video_writer = None

        if "video_path" in options:
            self.video_writer = imageio.get_writer(options["video_path"], fps=30)

        # Prefer explicit seed kwarg (gymnasium standard), fall back to options
        new_seed = seed if seed is not None else options.get("seed", None)
        if self.init_state is not None:
            if not self.has_reset_before:
                # the env must be fully reset at least once to ensure correct rendering
                self.env.reset()
                self.has_reset_before = True

            # always reset to the same state to be compatible with gym
            raw_obs = self.env.reset_to({"states": self.init_state})
        elif new_seed is not None:
            self.seed(seed=new_seed)
            raw_obs = self.env.reset()
        else:
            # random reset
            raw_obs = self.env.reset()
            
        # Initialize frame stack
        current_obs = self.get_single_observation(raw_obs)
        for _ in range(self.num_stack):
            self.frame_stack.append(current_obs)
            
        return self._get_stacked_observation(), {}

    def step(self, action):
        if self.normalize:
            action = self.unnormalize_action(action)
        raw_obs, env_reward, done, info = self.env.step(action)
        # Transform reward from [0, 1] to [-1, 0]
        reward = env_reward - 1.0
        prev_dones = self._update_subgoal_tracking(info)

        # Update stack
        current_obs = self.get_single_observation(raw_obs)
        self.frame_stack.append(current_obs)
        obs = self._get_stacked_observation()

        # render if specified
        if self.video_writer is not None:
            video_img = self.render(mode="rgb_array")
            self.video_writer.append_data(video_img)


        # Success / post-success reward shaping
        is_success_step = reward > -0.5

        if self.t_succ is None:
            if is_success_step:
                self.t_succ = self.t
                info["success"] = 1
                reward += 5.0
                if self.post_success_steps <= 0:
                    done = True
            else:
                info["success"] = 0
        else:
            # Post-success phase
            info["success"] = 1
            steps_since_succ = self.t - self.t_succ
            if is_success_step and steps_since_succ <= self.post_success_steps:
                reward += 5.0
            if steps_since_succ >= self.post_success_steps or not is_success_step:
                done = True

        if self.subgoal_cfg is not None:
            reward = self._subgoal_reward(env_reward, reward, prev_dones)

        self.t += 1
        self.env_step += 1
        self.episode_return += reward
        self.episode_length += 1

        # Always include return and length when episode ends
        terminated = done
        truncated = self.t >= self.max_episode_length
        if terminated or truncated:
            info['return'] = self.episode_return
            info['length'] = self.episode_length

        if terminated:
            return obs, reward, True, False, info
        if truncated:
            return obs, reward, False, True, info
        return obs, reward, False, False, info

    def render(self, mode="rgb_array"):
        h, w = self.render_hw
        return self.env.render(
            mode=mode,
            height=h,
            width=w,
            camera_name=self.render_camera_name,
        )

    def get_episode_info(self):
        return {"return": self.episode_return, "length": self.episode_length}
    
    def get_info(self):
        return {"env_step": self.env_step, "n_episodes": self.n_episodes}

if __name__ == "__main__":
    # for testing
    import sys
    env_name = sys.argv[1] if len(sys.argv) > 1 else "square-mh-image"
    print(f"Testing: {env_name}")
    task_cfg = _get_task_config(env_name)
    print(f"Task config: {task_cfg}")
    env = make_env(env_name)
    dataset = get_dataset(env, env_name)
    print(f"Observations shape: {dataset['observations'].shape}")
    print(f"Actions shape: {dataset['actions'].shape}")
    if 'images' in dataset._dict:
        print(f"Images shape: {dataset['images'].shape}")
    print("OK")
