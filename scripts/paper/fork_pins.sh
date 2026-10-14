# Sourced by launch.sh before the task file.
#
# The paper's runs were made with the QIPM fork of OGPO. This repo is upstream
# OGPO, whose yaml defaults differ from the fork's effective values in several
# places. Every task appends one of these arrays so that each such default is
# pinned to the fork's value. (The fork defaults are in the fork's
# ogpo/configs/algos/{common,ogpo,qipm,sac,sac_qipm}.yaml plus the code-level
# defaults of its agents, e.g. config.get('time_embedding', 'scalar').)
#
# Flags that only matter for one agent are harmless for the others (all keys
# exist in configs/algos/common.yaml).

# OGPO backbone (Sparse = ogpo, VaDaR = ogpo_vadar; fork: algo=ogpo / algo=qipm).
OGPO_FORK_PINS=(
    # Fork had no time-embedding module: the flow time t was concatenated raw.
    # Upstream defaults to sinusoidal.
    --agent.time_embedding=scalar
    # Fork computed the PPO log-prob ratio over the full flow chain. 0 = flow_steps.
    --agent.ft_flow_steps=0
    --agent.flow_steps=10
    # Fork's BC early stop (clip_bc) gated on the ODE eval.
    --bc.clip_bc_wrt=ODE
    # Fork interleaved the critic warmup with data collection (collect and train
    # together, rollout actor = best-of-N); upstream defaults to phase_split.
    --buffer.q_warmup_mode=interleaved
    --buffer.offline_ratio=0.0
    # BC phase only (fork training.offline_steps -> training.bc_pi_steps, set per task).
    --training.bc_q_steps=0
    --training.calql_steps=0
    --training.bc_refine_steps=0
    # OGPO policy-update settings shared by every fork run (fork ogpo.yaml / qipm.yaml).
    --agent.tau=0.05
    --agent.num_qs=10
    --agent.best_of_n=8
    --agent.clip_epsilon=0.01
    --agent.entropy_coeff=0.0
    --agent.use_bc_regularization=true
    --agent.use_constant_noise=true
    --agent.error_correct_sde_to_ode=true
    --agent.normalize_denoising_horizon=true
    --agent.normalize_act_space_dimension=true
    --agent.ppo_batch_size=256
    --agent.batch_size=256
    # VaDaR (fork QIPM) critic: gated TD target (td_loss_mode=gated), per-subgoal
    # discounts (subtask_based_discount=true), best-of-N on the unweighted sum of
    # subgoal values (fork sample_actions_BON).
    --agent.use_subgoal_discounts=true
    --agent.bon_weighting=sum
)

# FQL backbone (Sparse = fql, VaDaR = fql_vadar; fork: algo=ogpo / algo=qipm with
# the one-step policy). Same as OGPO except that the fork's FQL scripts passed
# time_embedding=sinusoidal explicitly, and the FQL one-step settings.
FQL_FORK_PINS=(
    "${OGPO_FORK_PINS[@]}"
    --agent.time_embedding=sinusoidal
    --agent.time_embedding_dim=32
    --agent.use_one_step_policy=true
    --agent.train_one_step_in_offline=true
    --agent.train_one_step_in_online=true
    --agent.use_one_step_for_targets=true
    --agent.use_one_step_for_rollouts=true
    --agent.fql_distillation_coeff=1000.0
    --agent.fql_train_main_policy=false
    --agent.layer_norm=true
    --agent.actor_layer_norm=false
    --agent.actor_backbone=mlp
    --agent.critic_backbone=mlp
)

# SAC backbone (Sparse = sac, VaDaR = sac_vadar; fork: algo=sac / algo=sac_qipm).
SAC_FORK_PINS=(
    # Fork interleaved the critic warmup (see above).
    --buffer.q_warmup_mode=interleaved
    --bc.clip_bc_wrt=ODE
    --training.bc_q_steps=0
    --training.calql_steps=0
    --training.bc_refine_steps=0
    --agent.batch_size=256
    --agent.init_temperature=1.0
    --agent.entropy_backup=false
    --agent.best_of_n=1
    # G from envs/subgoals.py (an older sac_vadar.yaml hard-coded 4).
    --agent.num_subgoals=null
    # Fork SAC-QIPM ignored subtask_discounts (it used the
    # scalar agent.discount for every subgoal), although the launchers passed them.
    --agent.use_subgoal_discounts=false
)
