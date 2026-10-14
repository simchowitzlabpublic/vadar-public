# Robomimic Transport (mh), OGPO backbone, 8% of the demos ("Transport (low)" in the paper).
# Fork launchers: paper_experiments/transport/*.sh (QIPM_TRANSPORT_GC, run tag p008_bc01).
# Note: the paper's table lists dataset.proportion 0.1, clip_bc_threshold 0.30 and
# 300K BC steps for this column; the launchers used 0.08, 0.1 and 200K (kept here).
BACKBONE=ogpo
SUITE=robomimic
ENV_NAME=transport-mh-low_dim
HGC_REWARD_MODE=hgc

TASK_FLAGS=(
    "${OGPO_FORK_PINS[@]}"
    --env.name=$ENV_NAME
    --dataset.proportion=0.08
    --training.bc_pi_steps=200000
    --training.q_warmup_steps=20000
    --training.online_steps=3000000
    --buffer.start_training=40000
    --buffer.utd_warmup=1
    --buffer.utd_online=1
    --buffer.utd_q=1
    --buffer.utd_pi=1
    --buffer.warmup_offline_ratio=0.0
    --bc.clip_bc=true
    --bc.clip_bc_threshold=0.1
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
    --agent.use_subgoal_dag=true         # two-arm DAG of envs/subgoals.py (fork: use_subtask_dag default true)
)

# Fork Sparse run (transport_ogpo_conservative.sh): no critic warmup, conservative
# advantages, subsampled-min Q, coarser logging.
EXTRA_sparse=(
    --training.q_warmup_steps=0 --agent.adv_strategy=conservative --agent.q_agg=subsample
    --eval.plot_q_vs_mc=false --intervals.log_interval=10000 --intervals.save_interval=1000000
)
# Fork HGC / iHGC runs did not set actor_tau (fork default 0.05).
EXTRA_hgc=(--agent.actor_tau=0.05 --agent.adv_strategy=conservative)
