"""Resolve the `reward_model` config of the preference-reward baselines.

Precedence (lowest to highest):
    configs/pbrl/reward_model.yaml          defaults
    configs/pbrl/<baseline>.yaml            preset ("dense" or "pref_subgoal")
    preset task_overrides[task_key(env)]    per-task values of the reference runs
    config['reward_model']                  the run's yaml section + CLI overrides

Only numpy-free, jax-free code lives here so it can be imported anywhere.
"""

import copy
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

BASELINES = ('dense', 'pref_subgoal')
LABEL_MODES = ('success_only', 'success_return')
SUBGOAL_LABEL_MODES = ('success_only', 'success_time')
ONLINE_SAMPLING = ('uniform', 'disagreement')

_CONFIG_DIR = Path(__file__).resolve().parent.parent / 'configs' / 'pbrl'


def _load_yaml(name: str) -> Dict[str, Any]:
    with open(_CONFIG_DIR / name, 'r') as f:
        return yaml.safe_load(f) or {}


def default_reward_model_config() -> Dict[str, Any]:
    """The defaults in configs/pbrl/reward_model.yaml."""
    return dict(_load_yaml('reward_model.yaml')['reward_model'])


def pbrl_baseline(config) -> Optional[str]:
    """Return 'dense' / 'pref_subgoal' if a preference-reward baseline is requested, else None."""
    rm = config.get('reward_model') or {}
    baseline = rm.get('baseline')
    if baseline in (None, 'null', 'None', '', False):
        return None
    baseline = str(baseline).lower()
    if baseline not in BASELINES:
        raise ValueError(f"reward_model.baseline must be one of {BASELINES} or null, got {baseline!r}")
    return baseline


def resolve_reward_model_config(config, env_name: Optional[str] = None) -> Dict[str, Any]:
    """Merge defaults, the baseline preset, its per-task overrides and the run's own keys.

    Args:
        config: the full run config (needs `reward_model.baseline`).
        env_name: used to pick the preset's `task_overrides` entry.

    Returns:
        A validated flat dict of reward-model settings.
    """
    baseline = pbrl_baseline(config)
    if baseline is None:
        raise ValueError("resolve_reward_model_config called but reward_model.baseline is not set")

    cfg = default_reward_model_config()
    preset = _load_yaml(f'{baseline}.yaml')
    cfg.update(preset.get('reward_model', {}))

    task_overrides = preset.get('task_overrides') or {}
    if env_name is not None and task_overrides:
        from envs.subgoals import task_key  # jax-free
        key = task_key(env_name)
        if key in task_overrides:
            cfg.update(task_overrides[key])

    cfg.update(copy.deepcopy(dict(config.get('reward_model') or {})))
    cfg['baseline'] = baseline
    return validate_reward_model_config(cfg)


def _is_null(v) -> bool:
    return v is None or (isinstance(v, str) and v.lower() in ('', 'null', 'none'))


def validate_reward_model_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Type-coerce and check the settings; returns the same dict."""
    for k in ('ensemble_size', 'hidden_dim', 'n_layers', 'train_batch_size', 'train_epochs',
              'feed_type', 'disagreement_oversample', 'min_offline_pairs', 'online_num_interact',
              'online_feedback_batch', 'online_max_feedback', 'online_train_epochs',
              'online_large_batch', 'online_relabel_chunk'):
        cfg[k] = int(cfg[k])
    for k in ('lr', 'val_fraction', 'reward_dataset_scale', 'subgoal_done_bonus'):
        cfg[k] = float(cfg[k])
    for k in ('all_pairs', 'online_enabled', 'freeze_pretrained'):
        v = cfg[k]
        cfg[k] = v.lower() == 'true' if isinstance(v, str) else bool(v)
    for k in ('rollout_dataset_path', 'pretrained_checkpoint_path', 'subgoal_dag',
              'discount', 'env_max_episode_length'):
        if _is_null(cfg.get(k)):
            cfg[k] = None

    if cfg['label_mode'] not in LABEL_MODES:
        raise ValueError(f"reward_model.label_mode must be one of {LABEL_MODES}, got {cfg['label_mode']!r}")
    if cfg['subgoal_label_mode'] not in SUBGOAL_LABEL_MODES:
        raise ValueError(f"reward_model.subgoal_label_mode must be one of {SUBGOAL_LABEL_MODES}, "
                         f"got {cfg['subgoal_label_mode']!r}")
    cfg['online_sampling'] = str(cfg['online_sampling']).lower()
    if cfg['online_sampling'] not in ONLINE_SAMPLING:
        raise ValueError(f"reward_model.online_sampling must be one of {ONLINE_SAMPLING}")
    if cfg['all_pairs'] and cfg['feed_type'] != 0:
        raise ValueError("reward_model.all_pairs=true requires feed_type=0: every pair is already enumerated")
    if not 0.0 <= cfg['val_fraction'] < 1.0:
        raise ValueError(f"reward_model.val_fraction must be in [0, 1), got {cfg['val_fraction']}")

    if cfg['baseline'] == 'pref_subgoal':
        from vadar.rewards.subgoal_preference_reward import canonical_weighting
        cfg['weighting'] = canonical_weighting(cfg['weighting'])

    if cfg['pretrained_checkpoint_path'] is not None:
        ckpt_dir = str(cfg['pretrained_checkpoint_path'])
        if not os.path.isdir(ckpt_dir):
            raise ValueError(f"reward_model.pretrained_checkpoint_path={ckpt_dir!r} is not a directory")
    if cfg['freeze_pretrained']:
        if cfg['pretrained_checkpoint_path'] is None:
            raise ValueError("reward_model.freeze_pretrained=true needs pretrained_checkpoint_path")
        if cfg['online_enabled']:
            print("[PBRL] freeze_pretrained=true overrides online_enabled=true; no online reward updates.")
        cfg['online_enabled'] = False
    return cfg
