# Multi-objective PushT (PushT-Obs)

The paper's multi-objective experiment (appendix "Multi-Objective Optimization on
PushT-Obs"): PushT with a static obstacle, a sparse task signal and a dense safety
cost. VaDaR is SAC with a 2-head critic (task, cost) and constant weights
`(1, lambda)`. aggSAC is SAC on the scalar reward `r_task + lambda * (-5 c)`.

The code is in `envs/pusht_utils.py`, ported from the QIPM fork's `pushT` branch
(`OGPO@e08e82b`). The paper's lambda sweep ran on that branch. With no obstacle,
the env and the dataset behave exactly as upstream.

## Environment

| | |
|---|---|
| env name | `pusht-keypoints-obstacle-v0` (`envs/subgoals.py` maps it to `pusht_obstacle`). `pusht-obstacle-v0` gives 5-D state observations. |
| base env | upstream `pusht-keypoints-v0`: 40-D observation (9 T keypoints and the pusher xy, plus 20 visibility flags, all scaled to [-1, 1]), 2-D action (PD target of the pusher), 300-step time limit. The paper text says the observation is a 5-D state, but the runs used keypoints. |
| obstacle | circle with center (320, 220) and radius 30 px, in the 512x512 workspace. It is virtual: nothing blocks the pusher or the T. |
| task signal | `task_done = 1[coverage >= 0.95]`. It ratchets within the episode, and the episode terminates when it fires. |
| cost `c` | 1 if the pusher disk (radius 15) or any part of the T polygon touches the obstacle. Otherwise it decays linearly over a 10 px clearance, `(10 - d) / 10`, and is 0 beyond that. The larger of the pusher and T values is used. |
| success | `task_done` and no step with `c == 1` during the episode. `info['naive_success']` = task only; `info['obstacle_avoidance']` = no collision. |
| start | the usual random PushT start (pusher in [50, 450]^2, T in [100, 400]^2, random angle). If the pusher or the T touches the obstacle, the start is drawn again with seed, seed+1, and so on. `fixed_start_seed` pins one layout; the paper did not use it. |
| `info['subgoal_dones']` | float32 `[task_done, c]`. VaDaR's cost head uses the target `-5 c + gamma^H mask Q'`, without clipping. |
| reward | VaDaR (`reward_mode: sparse`) gets the base env reward, which it ignores. aggSAC (`reward_mode: dense_parity`) gets `(dense_task_bonus if task_done else -1) + lambda * (-5 c)`. |

### Configuration

`make_env(env_name, seed, rew_fn, obstacle_cfg=None, subgoal_cfg=None)` and
`get_dataset(env_name, bc_dataset_path=None, obstacle_cfg=None, subgoal_cfg=None)`.
The obstacle config comes from `obstacle_cfg` or from `subgoal_cfg['obstacle']`,
and its values are merged over `PUSHT_OBSTACLE_DEFAULTS`, which hold the paper's
settings. The `-obstacle` env names therefore need no config.

| key | paper value | meaning |
|---|---|---|
| `shape, x, y, r` | `circle, 320, 220, 30` | obstacle shape and position (`rect` takes `x, y, w, h`) |
| `safe_margin` | 10 | width of the cost margin, in px |
| `coverage_threshold` | 0.95 | |
| `collide_with_pusher`, `collide_with_block` | true, true | which bodies count toward the cost |
| `terminate_on_collision` | false | |
| `sparse_safety` | false | ablation: binarize the cost |
| `dense_safety_coeff` | lambda | aggSAC lambda |
| `dense_safety_scale` | 5 | |
| `dense_task_bonus` | **15** | aggSAC task reward when done. The paper text says +5, but every paper run used 15 (run tag `tb15`), which matches VaDaR's task-head cap `c_clip * r_done = 15`. |
| `dense_sparse_step_cost` | -1 | |
| `dense_reward_mode` | set by `reward_mode` | `parity` for aggSAC; `pbrs` is ported but unused |

For VaDaR, set `subgoal_cfg = {num_subgoals: 2, reward_mode: sparse}`, use constant
weights `(1, lambda)`, and give head 1 the head type `cost`. From the command line:
`--algo=sac_vadar --env.name=pusht-keypoints-obstacle-v0 --agent.weighting=constant
--agent.subgoal_weights=[1.0,LAMBDA] --agent.vadar_critic=vector --agent.discount=0.95
--agent.horizon_length=4 --training.q_warmup_steps=0` (the fork used the vector critic
and no Q-warmup; `subgoal_head_types` comes from `envs/subgoals.py`). For aggSAC, set
`subgoal_cfg = {num_subgoals: 2, reward_mode: dense_parity, obstacle: {dense_safety_coeff: lambda}}`.

## BC dataset

`sac_pusht_2000000_bc.npz` comes from 2000 stochastic rollouts of a vanilla SAC
checkpoint (seed 1, 2M steps) on the obstacle-free env. Only successful episodes
were kept: 1964 episodes and 85,156 transitions, with the keys `observations,
actions, next_observations, rewards, terminals, masks`. Regenerate it with
`scripts/pusht/collect_bc_data.py`. On the cluster it is at
`~/OGPO/paper_experiments/pushT/bc_data/sac_pusht_2000000_bc.npz`.

The npz has no obstacle labels. With subgoal tracking on, `get_dataset` adds
`subgoal_dones` (N, 2). It recovers the T pose from the keypoints with a rigid
fit (error below 1e-5 px) and recomputes coverage and cost with the env's own
geometry. The labels are cached under `~/.vadar/cache/pusht_subgoal_dones/`.
The relabelled task flags match the stored terminations exactly (1964). The
dataset is mostly unsafe: 1455 of the 1964 episodes pass through the obstacle,
and 10.4% of transitions have `c = 1`. In the QIPM fork, the BC phase used only
the actions, and the offline data was not replayed online (`offline_ratio=0`).

## Experiment settings (as run)

The settings come from `paper_experiments/pushT/sweep_{qipm,dense_parity}_3seeds.sh`
and `submit_pushT_sweep.sh` on the cluster.

- 150k BC steps on the npz above, then 800k online steps. `start_training=10000`, UTD 1, `offline_ratio=0`.
- SAC: actor and critic MLPs of width 512 and depth 4 (critic with layer norm), 10 critics averaged (`q_agg=mean`), `tau=0.005`, `init_temperature=1.0`, `entropy_backup=false`, action chunk `H=4`, `best_of_n=1`.
- **`discount=0.95`**. The paper text says 0.99.
- Evaluation every 20k steps on 64 episodes (32 envs x 2).
- Lambda in {0.1, 0.5, 5, 10, 50} with seeds 1, 2, 3, plus lambda = 1 from earlier single runs (wandb `pushT-paper`: VaDaR `e5snzbf3, 0lmk7nnb, dwblp6zr`, aggSAC `zrzkfbm2, lyrzlj6x, fe7sv3ak`). The paper figure shows lambda in {0.1, 1, 5, 10}; 0.5 and 50 were dropped.
- One lambda = 1 aggSAC seed (`pushT_sac_dense_parity_bc_..._seed2`) hit a missing line continuation in its launcher. It ran with the yaml defaults instead: 200k BC steps, 1M online steps, `discount=0.99`. The plot keeps its last 800k steps.

Plot the sweep with
`python scripts/pusht/plot_lambda_sweep.py --logs_dir <logs> --out_dir <dir> [--lam1_wandb]`.
It reads both the fork's log names (`sac_qipm_avoid_pusht_*`, `sac_dense_parity_pusht_*`)
and the new ones (`sac_vadar_pusht_*`, `sac_aggsac_pusht_*`).

## Older version (fork `main` branch)

The fork's `main` branch has an earlier `PushTObstacleWrapper`, which the paper
did not use. It supports only rectangles, and it measures the cost from the
pusher and T *centers*, ignoring the pusher radius and the T polygon. It has no
start re-draw: starts can lie inside the obstacle. It sets success to task
completion alone, ignoring collisions. It has no `sparse_safety` or
`fixed_start_seed`. The `fixcoll` / `randomstart` / `circ` tags in the paper's
run names refer to the pushT-branch fixes: full-geometry collisions, re-drawn
random starts, and the circular obstacle.
