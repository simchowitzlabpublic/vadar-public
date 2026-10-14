# Robomimic ToolHang (ph), OGPO backbone, full demo set ("ToolHang (full)" in the paper).
# Fork launchers: paper_experiments/toolhang/*.sh (QIPM_TOOLHANG_GC).
# Note: the paper's hyperparameter table lists clip_bc_threshold 0.30 and
# start_training 10K for this column; the launchers used 0.25 / 10K (kept here).
BACKBONE=ogpo
SUITE=robomimic
ENV_NAME=tool_hang-ph-low_dim
HGC_REWARD_MODE=hgc        # fork robomimic GC wrapper: {-1,0} reward + 5 per newly completed subgoal

TASK_FLAGS=(
    "${OGPO_FORK_PINS[@]}"
    --env.name=$ENV_NAME
    --dataset.proportion=1.0
    --training.bc_pi_steps=300000
    --training.q_warmup_steps=20000
    --training.online_steps=3000000
    --buffer.start_training=10000
    --buffer.utd_warmup=1
    --buffer.utd_online=1
    --buffer.utd_q=1
    --buffer.utd_pi=1
    --buffer.warmup_offline_ratio=0.0
    --bc.clip_bc=true
    --bc.clip_bc_threshold=0.25
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
    --agent.use_subgoal_dag=false        # sequential chain
)

# Fork Sparse run (toolhang_ogpo_conservative.sh): conservative advantages, subsampled-min Q.
EXTRA_sparse=(--agent.adv_strategy=conservative --agent.q_agg=subsample --eval.plot_q_vs_mc=false)
# Fork HGC / iHGC runs (toolhang_baseline_gc.sh, toolhang_gc_critic_only.sh).
EXTRA_hgc=(--agent.adv_strategy=conservative)
