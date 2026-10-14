"""Collect the PushT-Obs BC dataset by rolling out a trained SAC checkpoint.

The paper's PushT-Obs runs BC-pretrain on rollouts of a vanilla SAC policy
trained on the obstacle-free task (pusht-keypoints-v0). This script rolls the
policy out on the obstacle-free env, keeps the successful episodes (coverage
>= 0.95), and saves an npz with observations, actions, next_observations,
rewards, terminals and masks. Pass it as the BC dataset path; the obstacle
labels (subgoal_dones = [task_done, cost]) are added at load time by
envs.pusht_utils.get_dataset, so one file serves every obstacle config.

Port of the QIPM fork's scripts/collect_bc_data_from_checkpoint.py
(pushT branch). The paper's file, sac_pusht_2000000_bc.npz, was made with
    --ckpt_path=.../sac/pushT-paper/sac/pusht-keypoints-v0/sac_pushtseed1/params_2000000.pkl
    --num_rollouts=2000 --filter_success_only=true --deterministic=false --seed=42
(1964 accepted episodes, 85,156 transitions; the stochastic policy's actions;
horizon 4 action chunks executed open loop).

Usage:
    python scripts/pusht/collect_bc_data.py --algo=sac \
        --ckpt_path=/path/to/params_2000000.pkl \
        --save_path=bc_data/sac_pusht_2000000_bc.npz
"""
import argparse
import importlib
import inspect
import os
import sys
import time

import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from envs import pusht_utils  # noqa: E402


def _str2bool(s: str) -> bool:
    return str(s).lower() in ("1", "true", "t", "yes", "y")


def _get_agent_class(algo_name: str):
    """The agent class defined in vadar.agents.<algo_name> (e.g. sac -> SACAgent)."""
    module = importlib.import_module(f"vadar.agents.{algo_name}")
    classes = [obj for name, obj in vars(module).items()
               if inspect.isclass(obj) and name.endswith("Agent") and obj.__module__ == module.__name__]
    if len(classes) != 1:
        raise ValueError(f"Expected exactly one *Agent class in vadar.agents.{algo_name}, found {classes}")
    return classes[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="sac", help="config / agent module name (vadar.agents.<algo>)")
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--env_name", type=str, default="pusht-keypoints-v0",
                        help="the env to roll out in; the paper used the obstacle-free env")
    parser.add_argument("--num_rollouts", type=int, default=2000)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--filter_success_only", type=_str2bool, default=True)
    parser.add_argument("--deterministic", type=_str2bool, default=False)
    args = parser.parse_args()

    import jax
    from vadar.agents.modules.flax_utils import restore_agent_with_file
    from vadar.utils.config_loader import config_to_agent_dict, load_config
    from vadar.utils.main_helper import get_q_bounds

    if not os.path.exists(args.ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt_path}")
    os.makedirs(os.path.dirname(os.path.abspath(args.save_path)), exist_ok=True)

    # ---- env (obstacle-free unless the env name says otherwise) ----
    config = load_config(args.algo)
    rew_fn = config.get("env", {}).get("rew_fn", "sparse")
    env = pusht_utils.make_env(args.env_name, seed=args.seed, rew_fn=rew_fn)
    action_dim = env.action_space.shape[0]

    # ---- agent: build for shape inference, then restore the weights ----
    agent_config = config_to_agent_dict(config)
    agent_config["horizon_length"] = agent_config.get("horizon_length") or 4
    q_min, q_max = get_q_bounds(args.env_name, agent_config["discount"])
    agent_config["q_min"], agent_config["q_max"] = q_min, q_max
    ex_obs = np.zeros(env.observation_space.shape, dtype=np.float32)
    ex_act = np.zeros(env.action_space.shape, dtype=np.float32)
    agent = _get_agent_class(args.algo).create(args.seed, ex_obs, ex_act, agent_config)
    agent = restore_agent_with_file(agent, args.ckpt_path)

    if args.deterministic and hasattr(agent, "compute_flow_actions"):
        actor_fn, stochastic = agent.compute_flow_actions, False
    else:
        actor_fn, stochastic = agent.sample_actions, True
    action_chunking = bool(agent_config.get("action_chunking", True))
    print(f"actor: {'stochastic' if stochastic else 'deterministic'}, "
          f"horizon_length={agent_config['horizon_length']}, action_chunking={action_chunking}")

    rng = jax.random.PRNGKey(args.seed)
    keys = ("observations", "actions", "next_observations", "rewards", "terminals", "masks")
    out = {k: [] for k in keys}
    accepted = rejected = 0
    lengths = []
    t0 = time.time()
    for r in range(args.num_rollouts):
        obs, _ = env.reset(seed=args.seed + 10000 + r)
        ep = {k: [] for k in keys}
        queue, success = [], False
        for _ in range(args.max_steps):
            if not queue:
                kwargs = dict(observations=obs[None])
                if stochastic:
                    rng, sub = jax.random.split(rng)
                    kwargs["rng"] = sub
                chunk = np.array(actor_fn(**kwargs)).reshape(-1)
                if action_chunking:
                    queue = [np.asarray(a, dtype=np.float32) for a in chunk.reshape(-1, action_dim)]
                else:
                    queue = [np.asarray(chunk[:action_dim], dtype=np.float32)]
            action = queue.pop(0)
            ep["observations"].append(np.asarray(obs, dtype=np.float32))
            ep["actions"].append(action)
            obs, reward, terminated, truncated, info = env.step(action)
            ep["next_observations"].append(np.asarray(obs, dtype=np.float32))
            ep["rewards"].append(np.float32(reward))
            ep["terminals"].append(np.float32(terminated or truncated))
            ep["masks"].append(np.float32(1.0 - float(terminated)))
            success = success or info.get("success", 0.0) > 0.5
            if terminated or truncated:
                break
        if args.filter_success_only and not success:
            rejected += 1
            continue
        for k in keys:
            out[k].extend(ep[k])
        accepted += 1
        lengths.append(len(ep["actions"]))
        if (r + 1) % 50 == 0:
            print(f"  rollout {r + 1}/{args.num_rollouts}  accepted={accepted}  rejected={rejected}  "
                  f"transitions={sum(lengths)}  ({time.time() - t0:.0f}s)")

    if not lengths:
        raise RuntimeError("No episodes collected (is the checkpoint's success rate 0?).")
    data = {k: np.asarray(np.stack(v) if k in ("observations", "actions", "next_observations") else v,
                          dtype=np.float32) for k, v in out.items()}
    np.savez(args.save_path, **data)
    print(f"saved {args.save_path}: {accepted} episodes ({rejected} rejected), {sum(lengths)} transitions, "
          f"mean length {np.mean(lengths):.1f}")
    for k, v in data.items():
        print(f"  {k:18s} {v.shape} {v.dtype}")


if __name__ == "__main__":
    main()
