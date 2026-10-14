# Robomimic Transport (mh), FQL backbone (fork: OGPO's one-step mode), all demos.
# Fork launchers: paper_experiments/fql-transport/*.sh (FQL-QIPM-TRANSPORT).
# Note: the paper's FQL table lists q_agg mean; every fql-transport launcher
# (VaDaR included) used q_agg=subsample (kept here).
BACKBONE=fql
SUITE=robomimic
ENV_NAME=transport-mh-low_dim
HGC_REWARD_MODE=hgc

TASK_FLAGS=(
    "${FQL_FORK_PINS[@]}"
    --env.name=$ENV_NAME
    --env.sparse=false
    --dataset.proportion=1.0
    --training.bc_pi_steps=1000000
    --training.q_warmup_steps=50000
    --training.online_steps=6000000
    --buffer.start_training=40000
    --buffer.utd_warmup=1
    --buffer.utd_online=1
    --buffer.utd_q=1
    --buffer.utd_pi=1
    --buffer.warmup_offline_ratio=0.0
    --bc.clip_bc=true
    --bc.clip_bc_threshold=0.45
    --success_buffer.use_success_buffer=true
    --eval.n_eval_envs=32
    --eval.eval_episodes=2
    --eval.plot_q_vs_mc=false
    --intervals.eval_interval=50000
    --intervals.eval_interval_bc=50000
    --intervals.log_interval=10000
    --intervals.save_interval=1000000
    --agent.horizon_length=8
    --agent.discount=0.999
    --agent.actor_tau=0.005
    --agent.subsample_bon=true
    --agent.bc_coeff=1.0
    --agent.group_num_samples=32
    --agent.use_denoiser=true
    --agent.use_constant_scheduler_for_bc=false
    --agent.constant_noise_std=0.01
    --agent.min_noise_std=0.01
    --agent.max_noise_std=0.01
    --agent.actor_hidden_dims=[512,512,512,512]
    --agent.value_hidden_dims=[512,512,512,512]
    --agent.adv_strategy=vanilla
    --agent.q_agg=subsample
    --agent.subgoal_discounts=[0.9,0.99,0.99,0.9,0.99,0.99,0.999]
    --agent.use_subgoal_dag=true
)
