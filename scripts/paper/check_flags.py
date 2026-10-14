"""Check that every flag a paper launcher passes exists in the yaml configs.

For each (task, method) pair it runs `DRY_RUN=1 launch.sh <task> <method>`, parses the
printed command with vadar's own CLI parser, loads the config with
vadar.utils.config_loader.load_config (overrides applied), and reports override keys
that are not in the algo's yaml (configs/algos/<algo>.yaml + its bases) or in
configs/pbrl/reward_model.yaml for reward_model.*.

    PYTHONPATH=. python scripts/paper/check_flags.py                 # every task x method
    PYTHONPATH=. python scripts/paper/check_flags.py toolhang_ogpo:vadar_nf libero_ks6_sac:her_latent
"""
import os
import shlex
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from vadar.utils.config_loader import config_to_agent_dict, flatten_config, load_config, parse_cli_args  # noqa: E402
SKIPPED = []

PAPER = ROOT / "scripts" / "paper"
# Dict-valued sections whose sub-keys are free-form.
OPEN_SECTIONS = ("env.obstacle.",)
# Methods that only make sense for some tasks.
ONLY_FOR = {
    "aggsac": ("pusht_obstacle_sac",),
    "vadar_multiobj": ("pusht_obstacle_sac",),
    # HER goal specs exist for LIBERO and Robomimic (envs/{libero,robomimic}_utils.get_her_spec).
    "her": ("libero_ks3_sac", "libero_ks4_sac", "libero_ks6_sac", "libero_ks8_sac", "toolhang_ogpo"),
    "her_latent": ("libero_ks6_sac",),
    "her_nextbc": ("libero_ks6_sac",),
}


def dry_run(task, method):
    env = dict(os.environ, DRY_RUN="1", VADAR_ROOT=str(ROOT), SEED="1")
    env.pop("SLURM_JOB_ID", None)
    out = subprocess.run(["bash", str(PAPER / "launch.sh"), task, method], env=env,
                         capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"{task}:{method}: launch.sh failed:\n{out.stderr}")
    return shlex.split(out.stdout.strip().splitlines()[-1])


def check(task, method, reward_model_keys):
    argv = dry_run(task, method)
    assert argv[:3] == ["python", "-u", "vadar/main.py"], argv[:3]
    algo, overrides = parse_cli_args(argv[3:])
    base = flatten_config(load_config(algo))
    cfg = load_config(algo, overrides)             # must not raise
    problems = []
    for key, value in overrides.items():
        if key.startswith(OPEN_SECTIONS):
            continue
        if key.startswith("reward_model."):
            if key not in reward_model_keys:
                problems.append(f"{key} not in configs/pbrl/reward_model.yaml")
            continue
        if key not in base:
            problems.append(f"{key} not in configs/algos/{algo}.yaml (or its bases)")
    # Effective values must equal the last occurrence on the command line.
    flat = flatten_config(cfg)
    for key, value in overrides.items():
        if key in flat and flat[key] != value and not key.startswith(OPEN_SECTIONS):
            problems.append(f"{key}: effective {flat[key]!r} != requested {value!r}")
    # Resolve the subgoal settings exactly as vadar/main.py does (no env is built).
    try:
        from vadar.main import resolve_subgoal_config
        agent_config = config_to_agent_dict(cfg)
        env_config = dict(cfg["env"])
        if (cfg.get("reward_model") or {}).get("baseline") == "pref_subgoal":
            env_config["track_subgoals"] = True
        resolve_subgoal_config(cfg["agent"]["agent_name"], env_config, agent_config)
    except (ImportError, FileNotFoundError) as e:   # env package or data file (e.g. HER encoder) missing here
        SKIPPED.append(f"{task}:{method}: subgoal resolution skipped ({e})")
    except Exception as e:
        problems.append(f"resolve_subgoal_config: {type(e).__name__}: {e}")
    return algo, len(overrides), problems


def main():
    rm = yaml.safe_load(open(ROOT / "vadar" / "configs" / "pbrl" / "reward_model.yaml"))
    reward_model_keys = set(flatten_config(rm))
    if len(sys.argv) > 1:
        pairs = [tuple(a.split(":")) for a in sys.argv[1:]]
    else:
        tasks = sorted(p.stem for p in (PAPER / "tasks").glob("*.sh"))
        methods = sorted(p.stem for p in (PAPER / "methods").glob("*.sh"))
        pairs = [(t, m) for t in tasks for m in methods if t in ONLY_FOR.get(m, (t,))]
    n_bad = 0
    for task, method in pairs:
        algo, n, problems = check(task, method, reward_model_keys)
        status = "OK " if not problems else "BAD"
        n_bad += bool(problems)
        print(f"{status} {task:26s} {method:20s} algo={algo:11s} {n:3d} overrides")
        for p in problems:
            print(f"      {p}")
    for s in SKIPPED:
        print("note:", s)
    print(f"\n{len(pairs)} pairs checked, {n_bad} with problems")
    sys.exit(1 if n_bad else 0)


if __name__ == "__main__":
    main()
