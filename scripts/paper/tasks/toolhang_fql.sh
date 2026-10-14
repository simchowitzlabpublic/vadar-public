# Robomimic ToolHang (ph), FQL backbone (fork: OGPO's one-step mode), all demos.
# Fork launchers: paper_experiments/fql-toolhang/*.sh (FQL-QIPM-TOOLHANG).
BACKBONE=fql
SUITE=robomimic
ENV_NAME=tool_hang-ph-low_dim
HGC_REWARD_MODE=hgc

TASK_FLAGS=(
    "${FQL_FORK_PINS[@]}"
    --env.name=$ENV_NAME
    --env.sparse=false
    --dataset.proportion=1.0
    --training.bc_pi_steps=500000
    --training.q_warmup_steps=50000
    --training.online_steps=3000000
    --buffer.start_training=10000
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
    --intervals.eval_interval=20000
    --intervals.eval_interval_bc=20000
    --intervals.log_interval=5000
    --intervals.save_interval=-1
    --agent.horizon_length=8
    --agent.discount=0.999
    --agent.actor_tau=0.005
    --agent.subsample_bon=true
    --agent.bc_coeff=1.0
    --agent.group_num_samples=32
    --agent.use_denoiser=true
    --agent.use_constant_scheduler_for_bc=true   # not set in these launchers -> fork default true
    --agent.constant_noise_std=0.005
    --agent.min_noise_std=0.005
    --agent.max_noise_std=0.005
    --agent.actor_hidden_dims=[512,512,512,512]
    --agent.value_hidden_dims=[512,512,512,512]
    --agent.adv_strategy=vanilla
    --agent.q_agg=mean
    --agent.subgoal_discounts=[0.9,0.99,0.995,0.999]
    --agent.use_subgoal_dag=false
)

# Fork Sparse / HGC / iHGC FQL runs used subsampled-min Q (VaDaR runs: mean).
EXTRA_sparse=(--agent.q_agg=subsample)
EXTRA_hgc=(--agent.q_agg=subsample)
