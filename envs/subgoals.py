"""Subgoal decompositions of every task used with VaDaR (single source of truth).

Each entry gives the ordered subgoal names, an optional prerequisite DAG
{subgoal: [prerequisites]} (None means a sequential chain), and the default
per-subgoal discounts used for the paper's experiments. The completion
checks themselves live in the per-suite env modules (robomimic_utils,
robocasa_utils, libero_utils, ogbench_utils, pusht_utils); they must report
cumulative subgoal-done vectors in this order.
"""

from typing import Optional

SUBGOAL_SPECS = {
    # ---------------- Robomimic ----------------
    'tool_hang': dict(
        names=['frame_grasped', 'frame_inserted', 'frame_inserted_and_wrench_grasped', 'wrench_attached'],
        dag=None,
        discounts=[0.9, 0.99, 0.995, 0.999],
    ),
    'transport': dict(
        names=['lid_removed', 'payload_grasped', 'payload_lifted', 'trash_grasped',
               'trash_in_bin', 'payload_handed_off', 'payload_in_target'],
        # Robot 0: lid -> payload grasped -> lifted; robot 1: trash grasped -> in bin;
        # both branches -> handoff -> payload in target bin.
        dag={0: [], 1: [0], 2: [1], 3: [], 4: [3], 5: [2, 4], 6: [5]},
        discounts=[0.9, 0.99, 0.99, 0.9, 0.99, 0.99, 0.999],
    ),
    'square': dict(
        names=['nut_grasped', 'nut_lifted_above_peg', 'nut_on_peg'],
        dag=None,
        discounts=None,
    ),
    # ---------------- RoboCasa ----------------
    'LoadDishwasher': dict(
        names=['cup_grasped', 'cup_in_dishwasher', 'bowl_grasped', 'bowl_in_dishwasher', 'dishwasher_closed'],
        dag={0: [], 1: [0], 2: [], 3: [2], 4: [1, 3]},
        discounts=[0.9, 0.99, 0.99, 0.995, 0.999],
    ),
    'PreSoakPan': dict(
        names=['pan_grasped_lifted', 'pan_in_sink', 'sponge_grasped_lifted', 'sponge_in_pan_and_sink', 'water_on'],
        dag=None,
        discounts=[0.9, 0.99, 0.99, 0.995, 0.999],
    ),
    'ArrangeTea': dict(
        names=['kettle_grasped', 'kettle_on_tray', 'kettle_on_tray+mug_grasped', 'kettle_on_tray+mug_on_tray',
               'both_on_tray+right_door_closed', 'both_on_tray+cab_closed'],
        dag=None,
        discounts=[0.9, 0.99, 0.99, 0.995, 0.999, 0.9995],
    ),
    # ---------------- LIBERO ----------------
    'KITCHEN_SCENE3': dict(names=['stove_on', 'moka_pot_grasped', 'moka_pot_on_stove'],
                           dag=None, discounts=[0.9, 0.95, 0.99]),
    'KITCHEN_SCENE4': dict(names=['bowl_grasped', 'bowl_in_drawer', 'drawer_closed'],
                           dag=None, discounts=[0.9, 0.95, 0.99]),
    'KITCHEN_SCENE6': dict(names=['mug_grasped', 'mug_in_microwave', 'microwave_closed'],
                           dag=None, discounts=[0.9, 0.95, 0.99]),
    'KITCHEN_SCENE8': dict(names=['pot2_grasped', 'pot2_on_stove', 'pot1_grasped', 'pot1_on_stove'],
                           dag=None, discounts=None),
    # ---------------- OGBench ----------------
    'scene-task4': dict(names=['button_pressed', 'drawer_open', 'block_in_drawer', 'drawer_closed'],
                        dag=None, discounts=None),
    # ---------------- Multi-objective PushT ----------------
    # Not subgoals: head 0 is the task, head 1 an obstacle cost (see 'subgoal_head_types').
    'pusht_obstacle': dict(names=['task_done', 'obstacle_cost'], dag=None, discounts=None,
                           head_types=['subgoal', 'cost']),
}


def task_key(env_name: str) -> Optional[str]:
    """Map an environment name to its key in SUBGOAL_SPECS (None if it has no subgoals)."""
    if env_name.startswith('robocasa_'):
        return env_name[len('robocasa_'):].split('-')[0]
    if env_name.startswith('libero_'):
        for key in SUBGOAL_SPECS:
            if key.startswith('KITCHEN_SCENE') and key in env_name:
                return key
        return None
    if env_name.startswith('tool_hang'):
        return 'tool_hang'
    if env_name.startswith('transport'):
        return 'transport'
    if env_name.startswith('square'):
        return 'square'
    if env_name.startswith('scene-') and 'task4' in env_name:
        return 'scene-task4'
    if env_name.startswith('pusht') and 'obstacle' in env_name:
        return 'pusht_obstacle'
    return None


def get_subgoal_spec(env_name: str) -> dict:
    """Subgoal spec of `env_name`: names, dag, discounts, num_subgoals, head_types."""
    key = task_key(env_name)
    if key is None:
        raise ValueError(f"No subgoal decomposition is defined for {env_name!r}")
    spec = dict(SUBGOAL_SPECS[key])
    spec['num_subgoals'] = len(spec['names'])
    spec.setdefault('head_types', None)
    spec['key'] = key
    return spec
