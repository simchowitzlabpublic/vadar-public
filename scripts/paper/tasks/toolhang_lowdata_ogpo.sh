# Robomimic ToolHang (ph), OGPO backbone, 30% of the demos ("ToolHang (low)" in the paper).
# Fork launchers: paper_experiments/toolhang-low-data/*.sh (project toolhang-low-data).
# Note: the paper's table lists offline_steps 300K and actor tau 0.005 for this
# column; the launchers used 500K BC steps, actor_tau 0.05 and subsample_bon=false
# (kept here).
BACKBONE=ogpo
SUITE=robomimic
ENV_NAME=tool_hang-ph-low_dim
HGC_REWARD_MODE=hgc

TASK_FLAGS=(
    "${OGPO_FORK_PINS[@]}"
    --env.name=$ENV_NAME
    --dataset.proportion=0.3
    --training.bc_pi_steps=500000
    --training.q_warmup_steps=20000
    --training.online_steps=3000000
    --buffer.start_training=10000
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
    --agent.actor_tau=0.05
    --agent.subsample_bon=false
    --agent.bc_coeff=0.1
    --agent.group_num_samples=16
    --agent.use_denoiser=true
    --agent.use_constant_scheduler_for_bc=false
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

EXTRA_sparse=(--agent.adv_strategy=conservative --agent.q_agg=subsample --eval.plot_q_vs_mc=false)
EXTRA_hgc=(--agent.adv_strategy=conservative)
