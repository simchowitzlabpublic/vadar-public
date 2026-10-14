# VaDaR repository design

This repo is upstream OGPO (`simchowitzlabpublic/OGPO@c230a0e`, package renamed
`ogpo` -> `vadar`) with Value Disaggregation (VaDaR) and the paper's baselines
added on top. Upstream files are kept as close to verbatim as possible; VaDaR
lives in separate files.

## Agents: each sparse agent has a VaDaR twin

| Base algorithm | Sparse (paper: "Sparse") | VaDaR | Notes |
|---|---|---|---|
| OGPO | `vadar/agents/ogpo.py` (upstream, verbatim + goal-obs wrapper) | `vadar/agents/ogpo_vadar.py` | |
| FQL | `algo=fql` -> `OGPOAgent` with the FQL one-step config (`configs/algos/fql.yaml`) | `algo=fql_vadar` -> `OGPOVadarAgent` + FQL config | upstream implements FQL as OGPO's one-step mode |
| SAC | `vadar/agents/sac.py` (from the QIPM fork) | `vadar/agents/sac_vadar.py` | |

`diff agent.py agent_vadar.py` must show only VaDaR changes:
1. the critic returns `(G, M, B)` (one value per subgoal) instead of `(M, B)`;
2. the TD target is `compute_vadar_td_target` on `batch['subgoal_dones'][:, -1, :]`
   instead of the scalar reward;
3. the policy is extracted from `Q_agg = sum_g w(d)[g] Q(s, a, g)`, with
   `w = subgoal_weights(dones, weighting, dag)` from the batch's subgoal dones;
4. rollout-time critic queries (best-of-N) use fixed weights (`bon_weighting`),
   never subgoal information.

Shared VaDaR code: `vadar/agents/modules/vadar_helper.py` (weightings, targets,
shaped reward) and `vadar/networks/vadar_critics.py` (critic parameterizations).

## Naming (paper <-> code <-> old QIPM fork)

| Paper | Config | Old fork name |
|---|---|---|
| VaDaR-NF (next-future, default) | `weighting: next_future` (alias `NF`) | `grandparent_future` / gpf |
| VaDaR-CF (current-future) | `weighting: current_future` (`CF`) | `goal_future` / gf |
| VaDaR-GM (current-myopic) | `weighting: current_myopic` (`GM`) | `goal_myopic` |
| next-myopic | `weighting: next_myopic` (`NM`) | none: the fork's `grandparent_myopic` reduced to last-only in OGPO-QIPM and to all-zero weights in SAC-QIPM |
| VaDaR Last-Q | `weighting: last` | `last_only` / lastq |
| VaDaR critic Q(s,a,g) | `vadar_critic: goal_conditioned` (default) | `scalar_goal_conditioned_critic` |
| Vector-VaDaR | `vadar_critic: vector` | default QIPM critic |
| separate critics | `vadar_critic: separate` | `separate_critic_per_subtask` |
| MIQ | `use_mip_q: true` | same |
| HGC | sparse agent + `goal_obs: cumulative`, `goal_conditioned_actor/critic: true`; `reward_mode: hgc` on robomimic/RoboCasa, `sparse` on LIBERO/OGBench (as the fork's runs); `use_subgoal_dag: false` on LoadDishwasher | `baseline_gc` |
| iHGC | as HGC with `goal_conditioned_actor: false` | `gc_critic_only` |
| Steelman 1/2/3 | VaDaR agent + `goal_conditioned_actor: true`, GM/CF/NF | `steelman_{1,2,3}` |
| VaDaR-Shaped | sparse agent + `reward_mode: shaped` (`sum_g (+5 if d_g else -1)`, paper definition; not the fork's `-#incomplete` dense reward) | separate runs, not in the fork |
| Dense (preference reward) | PbRL baseline | `OGPO-pbrl-baseline` |
| Pref-Subgoal | PbRL subgoal-gated baseline | `pbrl-subgoal-gated` |
| aggSAC | `sac` + PushT `reward_mode: dense_parity` | `sac_dense_parity` |

`subgoal_dones` replaces the fork's `subtask_status`; `num_subgoals` replaces
`no_of_subtasks`; `subgoal_dag` / `subgoal_discounts` replace `subtask_dag` /
`subtask_discounts`.

## Environment contract

Subgoal decompositions (names, DAG, default per-subgoal discounts) are in
`envs/subgoals.py` (`get_subgoal_spec(env_name)`).

`envs.env_utils.make_env_and_datasets(env_name, ..., subgoal_cfg=None)` and
`vadar.utils.main_helper.make_vectorized_env(env_name, ..., subgoal_cfg=None)`.
With `subgoal_cfg=None` the behavior is exactly upstream's. Otherwise
`subgoal_cfg` is a dict:

| key | values | meaning |
|---|---|---|
| `num_subgoals` | int | G (must equal the task's spec) |
| `dag` | dict or None | prerequisite DAG used when ratcheting dones |
| `reward_mode` | `sparse` (default) / `hgc` / `shaped` / `dense_parity` | `sparse`: the env's usual reward. `hgc`: -1 per step plus +5 on every step where a subgoal newly completes (the HGC/iHGC reward). `shaped`: VaDaR-Shaped, `sum_g (+5 if d_g else -1)`. `dense_parity`: PushT aggSAC |
| `goal_obs` | None / `cumulative` / `singular` / `state` / `latent` | append a goal vector to every observation (HGC, iHGC, steelmen, HER) |
| HER keys | `her_pos_threshold`, `her_encoder_path`, `her_bc_goal` | HER only |
| `obstacle` | dict or None | PushT obstacle config |

With subgoal tracking on, every `env.step` returns
`info['subgoal_dones']`: float32 `(G,)`, the cumulative completion vector after
the step, and the offline dataset has a `subgoal_dones` field `(N, G)` holding
post-action labels (relabelled by replaying stored simulator states, cached on
disk). The replay buffer stores it per transition and
`Dataset.sample_sequence` returns it as `(B, H, G)`.

## Behavioural notes vs the QIPM fork

- VaDaR math (weightings, gated TD targets, per-subgoal discounts, critic heads,
  MIQ, multi-objective cost heads) matches the fork; see `tests/` and the
  verification notes in the commit history. The fork's VaDaR runs used the
  **vector** critic (`vadar_critic: vector`); the repo default is the paper's
  Q(s,a,g) critic.
- Offline `subgoal_dones` are post-action labels (status after action t), the
  same convention as online `info['subgoal_dones']`. The fork's offline labels
  were pre-action; its main runs did not sample offline data online.
- OGPO/FQL follow upstream defaults (e.g. `q_warmup_mode: phase_split`,
  robomimic sparse reward with +5 on the success step). The fork interleaved
  the Q-warmup (`buffer.q_warmup_mode: interleaved`, the SAC default here) and
  used `{-1, 0}` robomimic rewards; `scripts/paper/` pins the fork's settings.
- RoboCasa runs should always set `env.anchor_demo` (all fork runs did).
