# Paper experiment launchers

Launchers that reproduce the VaDaR paper's experiments with this repo. The paper's
runs were made with the QIPM fork of OGPO; every hyperparameter below comes from
that fork's `paper_experiments/` launchers (and, for the preference baselines,
from `OGPO-pbrl-baseline` / `OGPO-pbrl-subgoal`). Names follow `docs/DESIGN.md`.

```
scripts/paper/
  launch.sh              one run: <task> x <method> x seed (SLURM-ready)
  fork_pins.sh           fork defaults that differ from this repo's yaml defaults
  tasks/<task>.sh        env + the fork's hyperparameters for one task/backbone
  methods/<method>.sh    algo variant (sparse vs *_vadar) + method flags
  steering.sh            critic-only warm start from a BC checkpoint (steering experiment)
  sweep_pusht_lambda.sh  PushT-Obs lambda sweep (VaDaR vs aggSAC)
  sweep_miq_utd.sh       MIQ critic-UTD sweep
  check_flags.py         checks every launcher flag against the yaml configs
scripts/steering/steering_eval.py   best-of-N steering evaluation of a checkpoint
```

## Running

```bash
export VADAR_VENV=/path/to/venv          # or VADAR_VENV_{ROBOMIMIC,ROBOCASA,LIBERO,OGBENCH,PUSHT}
export LIBERO_CLONE=/path/to/LIBERO      # LIBERO tasks (put on PYTHONPATH, as in the fork)
export ROBOCASA_DATA_DIR=$HOME/OGPO/data/robocasa   # RoboCasa hdf5s (see tasks/*.sh for file names)
export VADAR_WANDB_ENTITY=...            # optional; project defaults to vadar-paper
cd ~/vadar                               # submit from the repo root
sbatch --array=1-3 scripts/paper/launch.sh presoakpan_ogpo vadar_nf
DRY_RUN=1 bash scripts/paper/launch.sh presoakpan_ogpo vadar_nf   # print the python command
```

`launch.sh <task> <method> [--section.key=value ...]` builds
`python -u vadar/main.py --algo=<algo> <naming> <task flags> <method flags> <task-method extras> <your flags>`;
later flags win. The seed is `$SLURM_ARRAY_TASK_ID` (or `$SEED`, default 0); the
wandb group is the task and the run name `<method>_<task>[_<TAG>]`. Other data
locations: `OGBENCH_TASK4_BC_DIR`, `PUSHT_BC_DATA`, `HER_ENCODER_PATH`, and the
LIBERO hdf5s under `<repo>/data/libero/`.

Each task file sets the fork's settings of its **VaDaR** runs. Where the fork's
baselines in the same folder used different settings (e.g. `q_agg=subsample` and
conservative advantages for Sparse OGPO, a different `actor_tau`, discount or
layout for HGC), the task records them as `EXTRA_<method>` arrays, which
`launch.sh` appends for that method (`EXTRA_FROM` in a method file shares them:
iHGC uses HGC's, the preference baselines and HER use Sparse's).

Seeds: the LIBERO, OGBench and PushT runs used seeds 1-3, the steering runs 0-2.
The Robomimic and RoboCasa launchers took the seed from the SLURM array index;
use three seeds (`--array=1-3`).

Check every flag against the yaml configs (no GPU needed):
`PYTHONPATH=. python scripts/paper/check_flags.py` (all task x method pairs, or `task:method ...`).

## Tasks

| task | suite / backbone | fork launchers | notes |
|---|---|---|---|
| `toolhang_ogpo` | Robomimic ToolHang, OGPO | `toolhang/` | full data |
| `toolhang_lowdata_ogpo` | ToolHang, OGPO | `toolhang-low-data/` | `dataset.proportion=0.3` |
| `transport_ogpo` | Transport, OGPO | `transport/` | low data, `proportion=0.08` |
| `transport_highdata_ogpo` | Transport, OGPO | `transport-high-data/`, `mipQ-experiments/high_dataset/` | full data |
| `presoakpan_ogpo` | RoboCasa PreSoakPan, OGPO | `preSoakPan/`, `mipQ-experiments/preSoakPan/` | |
| `presoakpan_steelman_ogpo` | PreSoakPan, OGPO | `steelman-PreSoakPan/`, `understanding-experiments/` | 500K BC, 2.5-3M online |
| `loaddishwasher_ogpo` | RoboCasa LoadDishwasher, OGPO | `loadDishWasher/` | subgoal DAG |
| `arrangetea_ogpo` | RoboCasa ArrangeTea, OGPO | `arrangeTea/` | layouts differ per method (below) |
| `toolhang_fql` | ToolHang, FQL | `fql-toolhang/` | |
| `transport_fql` | Transport, FQL | `fql-transport/` | |
| `libero_ks{3,4,6}_sac` | LIBERO KITCHEN_SCENE3/4/6, SAC | `libero_kitchen{3,4,6}/`, `mipQ-experiments/libero-kitchen-scene-*` | |
| `libero_ks8_sac` | LIBERO KITCHEN_SCENE8, SAC | `libero_kitchen8/sac_her.sh` | not in the paper; only HER existed |
| `ogbench_task4_sac` | OGBench scene task4, SAC | `scene_task4_bc/` | BC data from SAC-VaDaR rollouts |
| `pusht_obstacle_sac` | PushT-Obs, SAC | `pushT/sweep_*_3seeds.sh` (pushT branch) | |

## Methods

| method | algo | paper | fork |
|---|---|---|---|
| `sparse` | backbone | Sparse | `ogpo_conservative`, `sac` |
| `hgc` | backbone + `env.goal_obs=cumulative`, actor and critic goal conditioned, `env.reward_mode=$HGC_REWARD_MODE` | HGC | `baseline_gc` |
| `ihgc` | as `hgc`, `goal_conditioned_actor=false` | iHGC | `gc_critic_only` |
| `vadar_nf` / `vadar_cf` / `vadar_gm` / `vadar_lastq` | `*_vadar`, weighting NF / CF / GM / last, vector critic | VaDaR-NF / -CF / -GM / LastQ | `qipm_{grandparent_future,goal_future,goalmyopic,lastq}` |
| `vadar_gc_critic_nf` / `_cf` | `*_vadar`, `vadar_critic=goal_conditioned` (Q(s,a,g)) | critic-parameterization ablation | (`qipm_gpf_scalar`) |
| `separate_critic` | `*_vadar`, `vadar_critic=separate`, NF | separate critics | `separateCritic` |
| `miq` | `*_vadar`, `use_mip_q=true`, NF | MIQ + VaDaR | `*_mipq_*` |
| `steelman1` / `2` / `3` | `*_vadar` + goal obs, actor-only goal conditioning, GM / CF / NF | Steelman 1 / 2 / 3 | `steelman_3` / `_2` / `_1` (reversed numbering) |
| `vadar_shaped` | backbone, `env.reward_mode=shaped` | VaDaR-Shaped | (no launcher in the fork's folder) |
| `dense_pbrl` | backbone, `reward_model.baseline=dense` | Dense | `pbrl/*_preference.sh` |
| `pref_subgoal` | backbone, `reward_model.baseline=pref_subgoal` | Pref-Subgoal | `pbrl-subgoal/*_subgoal_gated.sh` |
| `her` / `her_nextbc` / `her_latent` | backbone, `env.goal_obs=state` / `state` / `latent` | HER and its two steelmen | `sac_her{,_nextbc,_latent}` |
| `vadar_multiobj` | `sac_vadar`, constant weights (1, `$LAMBDA`), vector critic | VaDaR on PushT-Obs | `sac_qipm_avoid_obstacle` |
| `aggsac` | `sac`, `env.reward_mode=dense_parity`, `env.obstacle.dense_safety_coeff=$LAMBDA` | aggSAC | `sac_dense_parity` |

**Critic.** Every fork QIPM launcher used the vector critic S x A -> R^|G|, so the
`vadar_*` methods set `agent.vadar_critic=vector` to reproduce those runs. The
paper describes VaDaR with the goal-conditioned critic Q(s,a,g) (this repo's
default), which is `vadar_gc_critic_{nf,cf}`; App. "Critic parameterization" finds
the two equivalent.

**HGC reward.** On Robomimic and RoboCasa the fork's GC wrapper rewarded HGC/iHGC
with {-1, 0} plus +5 per newly completed subgoal (`env.reward_mode=hgc`). On LIBERO
and OGBench it kept the env's own reward (`HGC_REWARD_MODE=sparse` in those tasks).

## Paper figures and tables

S = 3 seeds (`sbatch --array=1-3 scripts/paper/launch.sh <task> <method>`).
"RoboCasa" = `presoakpan_ogpo loaddishwasher_ogpo arrangetea_ogpo`.

| figure / table | tasks | methods |
|---|---|---|
| Fig. main results (`main_bigbang`) | RoboCasa; `toolhang_ogpo toolhang_lowdata_ogpo transport_highdata_ogpo transport_ogpo`; `toolhang_fql transport_fql`; `libero_ks3_sac libero_ks4_sac libero_ks6_sac`; `ogbench_task4_sac` | `vadar_nf vadar_cf hgc ihgc sparse`, plus `dense_pbrl` (all but OGBench) |
| Fig. weighting ablation (`gating_all`) | RoboCasa, `toolhang_fql` | `vadar_gm vadar_cf vadar_nf hgc ihgc` (the fork has GM launchers only for ArrangeTea and FQL ToolHang) |
| Table + Fig. steelmen (`steelmans_small`) | `presoakpan_steelman_ogpo` | `hgc steelman1 steelman2 steelman3 vadar_cf vadar_nf` |
| Fig. LastQ (`bigbang_toprow_lastq_all`) and App. LastQ (`lastq_plots_combined`) | all tasks of the main figure; PreSoakPan's LastQ is `presoakpan_steelman_ogpo` | `vadar_lastq vadar_nf sparse` |
| Fig. steering (`steering_warmstart_n4`), App. steering grid + table | ToolHang, ArrangeTea, PreSoakPan | `steering.sh` (below) |
| App. multi-objective PushT-Obs (`pusht_panel`) | `pusht_obstacle_sac` | `sweep_pusht_lambda.sh` (below) |
| App. HGC limitations (`ihgc_vs_hgc_toolhang_fql`) | `toolhang_fql` | `hgc ihgc` |
| App. HER | `libero_ks6_sac` (and `libero_ks4_sac`) | `her her_nextbc her_latent`, vs `ihgc vadar_lastq` |
| App. Pref-Subgoal (`pref_subgoal_all`) | RoboCasa, `toolhang_fql` | `pref_subgoal vadar_nf` |
| App. reward shaping (`dense_reward_all`) | RoboCasa, `toolhang_fql` | `vadar_shaped vadar_nf` |
| App. critic parameterization (`scalar_vs_vector_all`) | RoboCasa, `toolhang_fql` | `vadar_gc_critic_nf vadar_gc_critic_cf vadar_nf vadar_cf` |
| App. MIQ (`mipq_grid`, `kitchen6_overview`) | `transport_highdata_ogpo transport_ogpo libero_ks4_sac libero_ks6_sac` | `sweep_miq_utd.sh` (below) |
| Separate critics (fork `understanding-experiments`; not in the current text) | `presoakpan_steelman_ogpo` | `separate_critic vadar_lastq vadar_nf` |

Not reproducible here: the VLM-judged subgoal completions (App. "VaDaR with VLMs")
and the Robometer progress-reward runs (`libero_kitchen*/sac_robometer.sh`).

## Steering (critic-only warm start + best-of-N evaluation)

Paper Sec. "Test-Time Steering", App. "Test-Time Policy Steering". Fork launchers:
`steering_warmstart/{toolhang,arrangetea}_{ogpo,qipm}_ws2.sh` and
`presoakpan_{ogpo,qipm}_wsold2.sh`, seeds 0-2.

1. **BC checkpoint.** Every OGPO run saves `params_<bc steps>.pkl` right after BC
   (`<save_dir>/<project>/<group>/<env>/<run>/`). The fork used its Sparse OGPO
   runs' BC checkpoints: ToolHang `params_300000.pkl` (`toolhang_ogpo sparse`),
   ArrangeTea `params_500000.pkl` (`arrangetea_ogpo sparse`, demo_0 layout), and for
   PreSoakPan an older 500K-step BC run with actor `[1024,1024,512,512,512]` and no
   denoiser (`steering.sh presoakpan` sets that architecture; a matching BC run is
   `launch.sh presoakpan_ogpo sparse --training.bc_pi_steps=500000 --training.online_steps=0
   --agent.actor_hidden_dims=[1024,1024,512,512,512] --agent.use_denoiser=false`).
2. **Critic training**, Sparse and VaDaR-NF critics, 3 seeds:
   ```bash
   sbatch --array=0-2 scripts/paper/steering.sh toolhang sparse   $BC/params_300000.pkl
   sbatch --array=0-2 scripts/paper/steering.sh toolhang vadar_nf $BC/params_300000.pkl
   # same for arrangetea and presoakpan (BC checkpoint at 500000)
   ```
   This restores the actor (`checkpoint.restore_actor_path`, `ep_resume` = BC steps,
   `bc_pi_steps=0`), sets `q_warmup_steps=400000 > online_steps=300000` so the whole
   online phase is the interleaved critic-only warmup (`q_warmup_mode=interleaved`):
   20K env steps seed the buffer (`start_training`), then 4 critic updates per env
   step (`utd_warmup=4`) on batches half offline demos, half frozen-policy rollouts
   (`warmup_offline_ratio=0.5`). A checkpoint is saved every 20K steps. As in the
   fork, the rollouts are collected with the runner's usual rollout actor, i.e.
   best-of-8 of the frozen policy under the critic being trained.
3. **Evaluation.** The paper's critic budgets 40K / 100K / 200K / 280K are the
   checkpoints `params_<ep_resume + 20K + budget>.pkl`, e.g. for ToolHang
   `params_{360000,420000,520000,600000}.pkl`, for ArrangeTea / PreSoakPan
   `params_{560000,620000,720000,800000}.pkl` (the fork's ArrangeTea evaluation used
   `params_570000.pkl` for the smallest budget). Each is evaluated on 96 episodes with
   N in {1, 4, 16, 64}:
   ```bash
   python scripts/steering/steering_eval.py --ckpt <run dir>/params_420000.pkl \
       --n_list 1,4,16,64 --n_envs 32 --rollouts 3 --no_ode --no_native_bon \
       --out results/steering/toolhang_vadar_s0_w100k.json
   ```
   The VaDaR critic is scored with its next-future weights computed from the live
   `subgoal_dones` (`--vadar_score weighted`, default); the Sparse critic with its
   ensemble aggregate (`--ensemble_agg auto` = the run's `q_agg`). The script is a
   port of the fork's `scripts/rollout/steering_eval.py`.

## Sweeps

**PushT-Obs lambda sweep** (`sweep_pusht_lambda.sh`): `vadar_multiobj` and `aggsac`
on `pusht_obstacle_sac` for lambda in {0.1, 0.5, 1, 5, 10, 50}, seeds 1-3
(`LAMBDAS=... SEEDS=...` to change). Logs are named for
`scripts/pusht/plot_lambda_sweep.py --logs_dir logs`; the paper's figure shows
lambda in {0.1, 1, 5, 10}. Single run: `LAMBDA=5 sbatch --array=1-3 scripts/paper/launch.sh pusht_obstacle_sac vadar_multiobj`.

**MIQ critic-UTD sweep** (`sweep_miq_utd.sh`): `miq` and `vadar_nf` at
`--buffer.utd_q` in {1, 4} with `--buffer.utd_pi=1` (SAC also `--buffer.utd_online=q`,
as in the fork), on `transport_highdata_ogpo` (with `bc_coeff=0.1`, as in the
fork's sweep), `transport_ogpo`, `libero_ks4_sac`, `libero_ks6_sac`; run names get
`_utdq<q>`.

## Differences from the fork and from the paper text

* **Pinned defaults** (`fork_pins.sh`): this repo's yaml defaults are upstream OGPO's.
  Pinned to the fork: `agent.time_embedding=scalar` for OGPO (FQL launchers passed
  `sinusoidal`), `agent.ft_flow_steps=0` (full chain), `bc.clip_bc_wrt=ODE`,
  `buffer.q_warmup_mode=interleaved`, per-subgoal discounts on
  for OGPO/FQL and off for SAC (the fork's SAC-VaDaR ignored them),
  `bon_weighting=sum`, and the shared OGPO policy settings.
* **Robomimic sparse reward.** Upstream's (+5 on the success step) replaces the fork's
  {-1, 0}. This changes the Sparse / Dense / HER baselines' reward scale on
  Robomimic; VaDaR's TD target uses `subgoal_dones` only.
* **VaDaR runs** do not plot Q vs MC (`plot_q_vs_mc` is turned off for vector critics
  by `vadar/main.py`); the task files keep the fork's value for the baselines.
* **Paper text vs launchers** (launcher values kept): RoboCasa `bc_coeff` 0.1 (text: 1)
  and BC steps 300K / 200K / 500K for PreSoakPan / LoadDishwasher / ArrangeTea (text:
  500K); Robomimic table values (`proportion` 0.1 vs 0.08 for Transport-low, clip
  thresholds, BC steps, actor tau for ToolHang-low; see each task file); FQL
  Transport `q_agg=subsample` (text: mean); PushT discount 0.95 (text: 0.99) and
  aggSAC task bonus 15 (text: +5).
* **Per-method settings inside one task**, kept as the fork ran them: ArrangeTea
  VaDaR-CF, HGC and iHGC used the `demo_11` layout (the rest `demo_0`), HGC/iHGC
  discount 0.999 and 3M steps; RoboCasa VaDaR-CF used `warmup_offline_ratio=0.5`;
  Transport (full) VaDaR-NF `bc_coeff=0.05` (others 0.1); HGC/iHGC on LoadDishwasher
  ratcheted subgoals sequentially (`use_subgoal_dag=false`); LIBERO VaDaR discount
  0.995 vs 0.99 for the baselines.
* **Steelman numbering**: the fork's `steelman_1.sh` is NF (paper Steelman 3) and
  `steelman_3.sh` is GM (paper Steelman 1); the methods follow the paper.

### Fork flags with no equivalent here

| fork flag | status |
|---|---|
| `agent.use_dense_rewards`, `agent.normalize_rewards` | dropped: the QIPM critic never used the env reward; subgoal tracking is automatic and rewards are chosen by `env.reward_mode` |
| `agent.gate_on_self` (false), `agent.policy_delay` (1) | legacy options left at their no-op values; not ported |
| `agent.floq.*` (`sac_qipm_floq`, older `OGBench-sac/` runs) | FloQ critic not ported (not in the paper) |
| `robometer.*` (`libero_kitchen*/sac_robometer.sh`) | Robometer progress rewards not ported |
| `env.obstacle.render_size`, `env.obstacle.viz_seed` | GIF rendering of the fork's PushT runner; no effect on training |
| `reward_model.reward_src`, `critic_checkpoint_path`, `v_*`, `segment_*`, `time_interp_alpha`, `preference_unit`, `rollout_video_enabled` | PbRL options not ported: every reference run used `label_mode=success_only`, under which they never change a label (see `vadar/rewards/preference_reward.py`) |
| `env.live_render_images` | only for Robometer |

Renamed (handled by the launchers): `training.offline_steps` -> `training.bc_pi_steps`,
`advantage_gating_mode` -> `weighting`, `td_loss_mode` (only `gated` is kept),
`subtask_based_discount` / `subtask_discounts` -> `use_subgoal_discounts` /
`subgoal_discounts`, `use_subtask_dag` -> `use_subgoal_dag`,
`goal_conditioning_mode` -> `env.goal_obs`, `no_of_subtasks` -> `num_subgoals` (automatic),
`scalar_goal_conditioned_critic` -> `vadar_critic=goal_conditioned`,
`separate_critic_per_subtask` -> `vadar_critic=separate`,
`safety_coeff` -> `weighting=constant` + `subgoal_weights=[1, lambda]`.
