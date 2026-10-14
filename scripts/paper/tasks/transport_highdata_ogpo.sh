# Robomimic Transport (mh), OGPO backbone, all demos ("Transport (full)" in the paper;
# also the high-data panel of the MIQ UTD sweep).
# Fork launchers: paper_experiments/transport-high-data/*.sh (project transport-high-data)
# and mipQ-experiments/high_dataset/transport/*.
# Note: the paper's table lists 300K BC steps / 3M online steps / bc_coeff 0.1 for
# this column; the VaDaR-NF launcher used 500K / 2M / 0.05 (kept here).
BACKBONE=ogpo
SUITE=robomimic
ENV_NAME=transport-mh-low_dim
HGC_REWARD_MODE=hgc

TASK_FLAGS=(
    "${OGPO_FORK_PINS[@]}"
    --env.name=$ENV_NAME
    --dataset.proportion=1.0
    --training.bc_pi_steps=500000
    --training.q_warmup_steps=20000
    --training.online_steps=2000000
    --buffer.start_training=40000
    --buffer.utd_warmup=1
    --buffer.utd_online=1
    --buffer.utd_q=1
    --buffer.utd_pi=1
    --buffer.warmup_offline_ratio=0.0
    --bc.clip_bc=true
    --bc.clip_bc_threshold=0.3
    --success_buffer.use_success_buffer=true
    --eval.n_eval_envs=32
    --eval.eval_episodes=2
    --eval.plot_q_vs_mc=true
    --intervals.eval_interval=20000
    --intervals.eval_interval_bc=20000
    --intervals.log_interval=5000
    --intervals.save_interval=-1
    --agent.horizon_length=8
    --agent.discount=0.999
    --agent.actor_tau=0.005
    --agent.subsample_bon=true
    --agent.bc_coeff=0.05                # VaDaR-NF launcher; VaDaR-CF / Sparse / HGC used 0.1
    --agent.group_num_samples=32
    --agent.use_denoiser=true
    --agent.use_constant_scheduler_for_bc=false
    --agent.constant_noise_std=0.01
    --agent.min_noise_std=0.01
    --agent.max_noise_std=0.01
    --agent.actor_hidden_dims=[512,512,512,512,512]
    --agent.value_hidden_dims=[512,512,512,512,512]
    --agent.adv_strategy=vanilla
    --agent.q_agg=mean
    --agent.subgoal_discounts=[0.9,0.99,0.99,0.9,0.99,0.99,0.999]
    --agent.use_subgoal_dag=true
)

EXTRA_vadar_cf=(--agent.bc_coeff=0.1)
EXTRA_sparse=(
    --agent.bc_coeff=0.1 --training.q_warmup_steps=0 --agent.adv_strategy=conservative
    --agent.q_agg=subsample --eval.plot_q_vs_mc=false --intervals.log_interval=10000
    --intervals.save_interval=1000000
)
EXTRA_hgc=(--agent.bc_coeff=0.1 --agent.actor_tau=0.05 --agent.adv_strategy=conservative)
# MIQ sweep (mipQ-experiments/high_dataset/transport/qipm/*): bc_coeff 0.1.
EXTRA_miq=(--agent.bc_coeff=0.1)
