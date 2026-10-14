"""Best-of-N steering of a frozen policy with a learned critic, from a saved checkpoint.

Port of the QIPM fork's scripts/rollout/steering_eval.py (paper App. "Test-Time
Policy Steering") to the vadar package.

At every decision point the (stochastic, SDE) flow policy proposes N action
chunks; each is scored by the *target* critic and the arg-max chunk is executed.
N=1 is the unsteered policy.

Critic scalarisation
  * Sparse (algo ogpo / fql): Q ensemble (M, B) -> aggregate over M.
  * VaDaR (ogpo_vadar / fql_vadar): per-subgoal ensemble (G, M, B) -> aggregate over
    M, then either
      - 'weighted': sum_g w(d)[g] Q_g with the run's training weighting
        (agent.weighting, e.g. next_future) driven by the live subgoal_dones from
        the env infos (what the paper's steering experiment used), or
      - 'sum'     : sum_g Q_g (the fixed best-of-N scoring of sample_actions_BON).
  Ensemble aggregation (--ensemble_agg): 'auto' = the run's agent.q_agg
  (mean | min | subsample = min over 2 random members).

Optional reference actors: 'ode' (compute_flow_actions, what eval.csv logs) and
'native_bon' (agent.sample_actions_BON with the run's best_of_n).

Usage (the steering runs save params_<step>.pkl every 20K steps, next to flags.json):
    python scripts/steering/steering_eval.py \
        --ckpt /data/user_data/$USER/vadar/<project>/<group>/<env>/<run>/params_420000.pkl \
        --n_list 1,4,16,64 --n_envs 32 --rollouts 3 --no_ode --no_native_bon \
        --out results/steering/toolhang_vadar_s0_w100k.json

The run's flags.json (saved next to the checkpoint) rebuilds the agent with the
exact training config; pass --flags to override. MIQ critics are not supported.
"""
import argparse
import json
import os
import pickle
import sys
import time
from collections import OrderedDict

import jax
import jax.numpy as jnp
import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from vadar.agents.modules.vadar_helper import subgoal_weights  # noqa: E402
from vadar.main import AGENTS, OGPO_FAMILY, VADAR_AGENTS, resolve_subgoal_config  # noqa: E402
from vadar.utils.config_loader import AttrDict, config_to_agent_dict  # noqa: E402
from vadar.utils.evaluation import _split_obs  # noqa: E402
from vadar.utils.main_helper import get_q_bounds, make_vectorized_env  # noqa: E402


# ----------------------------------------------------------------------------- agent / env
def build_agent_and_envs(config, n_envs, seed):
    """Rebuild the agent and vectorized eval envs exactly as vadar/main.py does."""
    config = AttrDict(config)
    agent_config = config_to_agent_dict(config)
    algo = agent_config["agent_name"]
    if algo not in OGPO_FAMILY:
        raise ValueError(f"steering_eval supports the OGPO-family agents {sorted(OGPO_FAMILY)}, got {algo}")
    if agent_config.get("use_mip_q", False):
        raise ValueError("steering_eval does not support MIQ critics")
    env_config = dict(config["env"])
    env_name = env_config["name"]
    subgoal_cfg = resolve_subgoal_config(algo, env_config, agent_config)
    if algo not in VADAR_AGENTS and subgoal_cfg is None:
        # Sparse critic: subgoal tracking is not needed for scoring, but it gives the
        # per-subgoal completion statistics in the output.
        try:
            env_config["track_subgoals"] = True
            subgoal_cfg = resolve_subgoal_config(algo, env_config, agent_config)
        except ValueError:
            subgoal_cfg = None

    eval_envs = make_vectorized_env(
        env_name, num_envs=n_envs, seed=seed + 1000, rew_fn=env_config["rew_fn"],
        post_success_steps=env_config.get("post_success_steps", 0), subgoal_cfg=subgoal_cfg,
        anchor_demo=env_config.get("anchor_demo"),
    )

    q_min, q_max = get_q_bounds(env_name, agent_config["discount"])
    agent_config["q_min"] = q_min
    agent_config["q_max"] = q_max

    # Example shapes straight from the eval envs (skips loading the offline dataset).
    obs_raw, _ = eval_envs.reset(seed=seed + 1000)
    obs0, img0, _ = _split_obs(obs_raw)
    action_dim = int(np.prod(eval_envs.single_action_space.shape))
    ex_obs = jnp.asarray(obs0[0], dtype=jnp.float32)
    ex_act = jnp.zeros((action_dim,), dtype=jnp.float32)
    ex_img = jnp.asarray(img0[0]) if img0 is not None else None
    print(f"example obs dim={ex_obs.shape} action_dim={action_dim}")

    agent = AGENTS[algo].create(seed, ex_obs, ex_act, agent_config, agent_config["adv_clip_min"],
                                ex_images=ex_img)
    return agent, eval_envs, action_dim, agent_config, env_name


def _check_restored_shapes(fresh, saved):
    """Fail loudly if any checkpoint leaf shape differs from the freshly built agent."""
    fresh_leaves = jax.tree_util.tree_leaves_with_path(fresh)
    saved_flat = {jax.tree_util.keystr(p): v for p, v in jax.tree_util.tree_leaves_with_path(saved)}
    bad, n_missing = [], 0
    for p, v in fresh_leaves:
        k = jax.tree_util.keystr(p)
        if k not in saved_flat:
            n_missing += 1
        elif hasattr(v, "shape") and hasattr(saved_flat[k], "shape") and tuple(v.shape) != tuple(saved_flat[k].shape):
            bad.append((k, f"fresh {tuple(v.shape)} vs ckpt {tuple(saved_flat[k].shape)}"))
    if bad:
        for k, why in bad[:20]:
            print(f"  SHAPE MISMATCH {k}: {why}")
        raise RuntimeError(f"{len(bad)} restored leaves mismatch: config and checkpoint disagree")
    print(f"Checkpoint leaf shapes match the fresh agent "
          f"({len(fresh_leaves) - n_missing}/{len(fresh_leaves)} leaves present).")


def tolerant_restore(agent, ckpt_path):
    """Restore a save_agent() checkpoint, keeping the fresh init for fields it lacks."""
    import flax
    with open(ckpt_path, "rb") as f:
        saved = pickle.load(f)["agent"]
    fresh = flax.serialization.to_state_dict(agent)
    _check_restored_shapes(fresh, saved)
    missing_top = [k for k in fresh if k not in saved]
    if missing_top:
        print(f"  ckpt lacks top-level fields {missing_top} -> keeping the fresh init for those")
    dropped = []

    def _merge(live_d, saved_d, path=""):
        if not isinstance(live_d, dict) or not isinstance(saved_d, dict):
            return saved_d if saved_d is not None else live_d
        out = dict(live_d)
        for k, v in saved_d.items():
            if k in out:
                out[k] = _merge(out[k], v, f"{path}/{k}")
            else:
                dropped.append(f"{path}/{k}")
        return out

    merged = _merge(fresh, saved)
    if dropped:
        print(f"  ckpt has {len(dropped)} entries the live agent lacks -> dropped: {dropped[:8]}")
    agent = flax.serialization.from_state_dict(agent, merged)
    print(f"Restored from {ckpt_path}")
    return agent


def splice_network(agent, ckpt, field):
    """Overwrite one top-level network ('critic_network' or 'actor_network', incl. its
    target and optimizer state) from another checkpoint with the same architecture."""
    import flax
    with open(ckpt, "rb") as f:
        saved_n = pickle.load(f)["agent"][field]
    state = flax.serialization.to_state_dict(agent)
    fresh_n = state[field]
    fl, sl = jax.tree_util.tree_leaves_with_path(fresh_n), jax.tree_util.tree_leaves_with_path(saved_n)
    if len(fl) != len(sl):
        raise RuntimeError(f"{field}: {len(fl)} live leaves vs {len(sl)} in {ckpt}")
    for (pf, vf), (ps, vs) in zip(fl, sl):
        if jax.tree_util.keystr(pf) != jax.tree_util.keystr(ps):
            raise RuntimeError(f"{field} structure mismatch: {jax.tree_util.keystr(pf)} vs {jax.tree_util.keystr(ps)}")
        if hasattr(vf, "shape") and tuple(vf.shape) != tuple(vs.shape):
            raise RuntimeError(f"{field} shape mismatch at {jax.tree_util.keystr(pf)}: {vf.shape} vs {vs.shape}")
    state[field] = saved_n
    agent = flax.serialization.from_state_dict(agent, state)
    print(f"Spliced {field} from {ckpt} ({len(fl)} leaves)")
    return agent


# ----------------------------------------------------------------------------- actors
def make_steer_fn(N, G, score_mode, weighting, dag, constant_weights, ens_agg="mean"):
    """jit(agent, obs[B,D], dones[B,G], rng) -> (best_actions[B,H*A], q_best[B], q_spread[B])."""
    is_vadar = G is not None

    @jax.jit
    def steer(agent, observations, dones, rng):
        B = observations.shape[0]
        rng, a_rng = jax.random.split(rng)
        obs_exp = jnp.repeat(observations, N, axis=0)                       # [B*N, D]
        actions, _, _, _ = agent.sample_actions_with_noise(
            obs_exp, rng=a_rng, use_target=True, is_encoded=False)          # [B*N, H*A]
        critic_obs = agent._encode_for_critic(obs_exp, use_target=True)
        q = agent.critic_network.select("target_critic")(
            critic_obs, actions=actions, is_encoded=agent._critic_is_encoded())
        if is_vadar:
            q = q.reshape(G, -1, B, N)                                       # (G, M, B, N)
            if ens_agg == "subsample":   # as the fork: 2 random members shared by all candidates
                rng, sub_key = jax.random.split(rng)
                idxs = jax.random.randint(sub_key, (2,), 0, q.shape[1])
                q = q[:, idxs].min(axis=1)                                   # (G, B, N)
            else:
                q = q.mean(axis=1) if ens_agg == "mean" else q.min(axis=1)  # (G, B, N)
            if score_mode == "weighted":
                w = subgoal_weights(dones, weighting=weighting, dag=dag,
                                    constant_weights=constant_weights)       # (G, B)
                q = (q * w[:, :, None]).sum(axis=0)                          # (B, N)
            else:
                q = q.sum(axis=0)
        else:
            q = q.reshape(-1, B, N)                                          # (M, B, N)
            if ens_agg == "subsample":   # as the fork: 2 random members per candidate
                rng, sub_key = jax.random.split(rng)
                idxs = jax.random.randint(sub_key, (2, B, N), 0, q.shape[0])
                q = jnp.take_along_axis(q, idxs, axis=0).min(axis=0)         # (B, N)
            else:
                q = q.mean(axis=0) if ens_agg == "mean" else q.min(axis=0)   # (B, N)
        best = jnp.argmax(q, axis=1)
        best_actions = actions.reshape(B, N, -1)[jnp.arange(B), best]
        return best_actions, q[jnp.arange(B), best], q.max(axis=1) - q.min(axis=1)

    return steer


class Actor:
    """(obs[B,D], dones[B,G]) -> action chunks [B, H*A]; owns its rng."""

    def __init__(self, agent, kind, seed, N=1, G=None, score_mode="weighted", weighting="next_future",
                 dag=None, constant_weights=None, ens_agg="mean"):
        self.agent, self.kind = agent, kind
        self.rng = jax.random.PRNGKey(seed)
        self.q_best, self.q_spread = [], []
        if kind == "steer":
            self._fn = make_steer_fn(N, G, score_mode, weighting, dag, constant_weights, ens_agg)

    def __call__(self, obs, dones):
        self.rng, sub = jax.random.split(self.rng)
        if self.kind == "ode":
            return np.asarray(self.agent.compute_flow_actions(observations=jnp.asarray(obs), rng=sub))
        if self.kind == "native_bon":
            return np.asarray(self.agent.sample_actions_BON(observations=jnp.asarray(obs), rng=sub))
        a, qb, qs = self._fn(self.agent, jnp.asarray(obs), jnp.asarray(dones, dtype=jnp.float32), sub)
        self.q_best.append(np.asarray(qb))
        self.q_spread.append(np.asarray(qs))
        return np.asarray(a)


# ----------------------------------------------------------------------------- rollout
def _info_subgoal_dones(infos, i):
    """info['subgoal_dones'] of env i from the vector env's batched infos (None if absent)."""
    if "subgoal_dones" not in infos:
        return None
    mask = infos.get("_subgoal_dones")
    if mask is not None and not np.asarray(mask)[i]:
        return None
    return np.asarray(infos["subgoal_dones"][i], dtype=np.float32)


def run_condition(envs, actor, n_envs, n_rollouts, action_dim, base_seed, G, log_every=0):
    succ, lens, subs = [], [], []
    t0 = time.time()
    width = G or 1
    for r in range(n_rollouts):
        obs_raw, _ = envs.reset(seed=base_seed + r * n_envs)
        obs, _, _ = _split_obs(obs_raw)
        dones_vec = np.zeros((n_envs, width), dtype=np.float32)
        queues = [[] for _ in range(n_envs)]
        active = np.ones(n_envs, dtype=bool)
        steps = np.zeros(n_envs, dtype=int)
        while active.any():
            need = [i for i, q in enumerate(queues) if not q and active[i]]
            if need:
                # Fixed-size batch so every jitted actor compiles once.
                obs_b = np.zeros((n_envs,) + obs.shape[1:], dtype=obs.dtype)
                obs_b[: len(need)] = obs[need]
                d_b = np.zeros((n_envs, width), dtype=np.float32)
                d_b[: len(need)] = dones_vec[need]
                chunks = actor(obs_b, d_b).reshape(n_envs, -1, action_dim)[: len(need)]
                for k, i in enumerate(need):
                    queues[i].extend(list(chunks[k]))
            step_actions = np.stack([
                queues[i].pop(0) if (active[i] and queues[i]) else np.zeros(action_dim) for i in range(n_envs)])
            obs_raw, _, term, trunc, infos = envs.step(step_actions)
            obs, _, _ = _split_obs(obs_raw)
            ended = np.logical_or(term, trunc)
            for i in range(n_envs):
                if not active[i]:
                    continue
                steps[i] += 1
                d = _info_subgoal_dones(infos, i)
                if d is not None:
                    dones_vec[i, : len(d)] = d
                if ended[i]:
                    succ.append(float(np.asarray(infos["success"])[i]))
                    lens.append(int(steps[i]))
                    subs.append(float(dones_vec[i].sum()))
                    active[i] = False
        if log_every:
            print(f"    rollout {r + 1}/{n_rollouts}: running success {np.mean(succ):.3f} "
                  f"({len(succ)} eps, {time.time() - t0:.0f}s)", flush=True)
    succ = np.asarray(succ)
    p = succ.mean()
    out = OrderedDict(
        success_rate=float(p), n_episodes=int(len(succ)),
        stderr=float(np.sqrt(p * (1 - p) / max(1, len(succ)))),
        mean_length=float(np.mean(lens)), mean_subgoals_done=float(np.mean(subs)),
        successes=succ.astype(int).tolist(), wall_s=float(time.time() - t0),
    )
    if isinstance(actor, Actor) and actor.q_best:
        out["q_best_mean"] = float(np.concatenate(actor.q_best).mean())
        out["q_spread_mean"] = float(np.concatenate(actor.q_spread).mean())
    return out


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--flags", default=None, help="flags.json (default: <ckpt dir>/flags.json)")
    ap.add_argument("--critic_ckpt", default=None,
                    help="checkpoint whose critic_network replaces the one in --ckpt")
    ap.add_argument("--actor_ckpt", default=None,
                    help="checkpoint whose actor_network replaces the one in --ckpt (same architecture)")
    ap.add_argument("--n_list", default="1,4,16,64")
    ap.add_argument("--n_envs", type=int, default=32)
    ap.add_argument("--rollouts", type=int, default=3, help="episodes per condition = n_envs * rollouts")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--vadar_score", default="weighted", choices=["weighted", "sum", "both"],
                    help="VaDaR critics: training weighting from live subgoal dones, or the plain sum")
    ap.add_argument("--ensemble_agg", default="auto", choices=["auto", "mean", "min", "subsample"],
                    help="how to collapse the Q ensemble per candidate; auto = the run's agent.q_agg")
    ap.add_argument("--no_ode", action="store_true")
    ap.add_argument("--no_native_bon", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    flags_path = args.flags or os.path.join(os.path.dirname(args.ckpt), "flags.json")
    with open(flags_path) as f:
        config = json.load(f)
    print(f"ckpt : {args.ckpt}\nflags: {flags_path}")

    agent, envs, action_dim, agent_config, env_name = build_agent_and_envs(config, args.n_envs, args.seed)
    agent = tolerant_restore(agent, args.ckpt)
    if args.critic_ckpt:
        agent = splice_network(agent, args.critic_ckpt, "critic_network")
    if args.actor_ckpt:
        agent = splice_network(agent, args.actor_ckpt, "actor_network")

    algo = agent_config["agent_name"]
    G = int(agent_config["num_subgoals"]) if algo in VADAR_AGENTS else None
    weighting = agent_config.get("weighting", "next_future")
    dag = agent_config.get("subgoal_dag")
    constant_weights = agent_config.get("subgoal_weights")
    print(f"algo={algo} env={env_name} G={G} weighting={weighting if G else '-'} dag={dag} "
          f"num_qs={agent_config.get('num_qs')} horizon={agent_config.get('horizon_length')} "
          f"native best_of_n={agent_config.get('best_of_n')}")

    if args.ensemble_agg == "auto":
        args.ensemble_agg = str(agent_config.get("q_agg", "mean"))
        assert args.ensemble_agg in ("mean", "min", "subsample"), args.ensemble_agg
    print(f"ensemble aggregation for steering: {args.ensemble_agg}")

    conditions = []
    if not args.no_ode:
        conditions.append(("ode", dict(kind="ode")))
    if not args.no_native_bon:
        conditions.append((f"native_bon{agent_config.get('best_of_n')}", dict(kind="native_bon")))
    n_list = [int(x) for x in args.n_list.split(",") if x]
    modes = ["weighted", "sum"] if (G and args.vadar_score == "both") else ([args.vadar_score] if G else ["mean"])
    for mode in modes:
        for N in n_list:
            if N == 1 and mode != modes[0]:
                continue  # N=1 is the same for every scoring mode
            tag = f"steer_N{N}" + (f"_{mode}" if G else "") + ("" if args.ensemble_agg == "mean" else f"_{args.ensemble_agg}")
            conditions.append((tag, dict(kind="steer", N=N, G=G, score_mode=mode, weighting=weighting, dag=dag,
                                         constant_weights=constant_weights, ens_agg=args.ensemble_agg)))

    results = OrderedDict(meta=dict(
        ckpt=args.ckpt, critic_ckpt=args.critic_ckpt, actor_ckpt=args.actor_ckpt, flags=flags_path, algo=algo,
        env=env_name, seed=args.seed, n_envs=args.n_envs, rollouts=args.rollouts, G=G, weighting=weighting,
        ensemble_agg=args.ensemble_agg))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    for tag, spec in conditions:
        print(f"\n=== {tag} ===", flush=True)
        actor = Actor(agent, seed=args.seed + 7, **spec)
        res = run_condition(envs, actor, args.n_envs, args.rollouts, action_dim,
                            base_seed=args.seed * 100000, G=G, log_every=1)
        results[tag] = res
        print(f"  -> {tag}: success {res['success_rate']:.3f} +- {res['stderr']:.3f} "
              f"(n={res['n_episodes']}, subgoals {res['mean_subgoals_done']:.2f}, len {res['mean_length']:.0f})",
              flush=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=1)

    print("\n" + "=" * 72)
    print(f"{'condition':28s} {'success':>8s} {'+-':>6s} {'n':>4s} {'subgoals':>9s} {'len':>6s}")
    for tag, res in results.items():
        if tag == "meta":
            continue
        print(f"{tag:28s} {res['success_rate']:8.3f} {res['stderr']:6.3f} {res['n_episodes']:4d} "
              f"{res['mean_subgoals_done']:9.2f} {res['mean_length']:6.0f}")
    print(f"saved -> {args.out}")
    envs.close()


if __name__ == "__main__":
    main()
