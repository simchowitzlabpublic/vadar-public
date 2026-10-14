# RoboCasa LoadDishwasher, OGPO backbone.
# Fork launchers: paper_experiments/loadDishWasher/*.sh (QIPM_LOADDISHWASHER_GC /
# vadar-dw-ablations).
# Note: the paper says bc_coeff 1 and 500K BC steps for RoboCasa; the launchers
# used the fork default bc_coeff 0.1 and 200K (kept here).
BACKBONE=ogpo
SUITE=robocasa
ENV_NAME=robocasa_LoadDishwasher
HGC_REWARD_MODE=hgc
ROBOCASA_DATASET_PATH=${LOADDISHWASHER_DATASET:-${ROBOCASA_DATA_DIR:-$VADAR_ROOT/data/robocasa}/LoadDishWasher/LoadDishwasher_noisy_replay_ready_31dim_v2.hdf5}

TASK_FLAGS=(
    "${OGPO_FORK_PINS[@]}"
    --env.name=$ENV_NAME
    --env.anchor_demo=demo_0
    --dataset.proportion=1.0
    --training.bc_pi_steps=200000
    --training.q_warmup_steps=20000
    --training.online_steps=5000000
    --buffer.start_training=20000
    --buffer.utd_warmup=1
    --buffer.utd_online=1
    --buffer.warmup_offline_ratio=0.0
    --bc.clip_bc=true
    --bc.clip_bc_threshold=0.3
    --success_buffer.use_success_buffer=true
    --eval.n_eval_envs=32
    --eval.eval_episodes=1
    --eval.plot_q_vs_mc=true
    --intervals.eval_interval=20000
    --intervals.eval_interval_bc=20000
    --intervals.log_interval=5000
    --intervals.save_interval=-1
    --agent.horizon_length=8
    --agent.discount=0.995
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
    # The launchers passed these discounts explicitly (the fork's code default for
    # LoadDishwasher was [0.99, 0.99, 0.99, 0.99, 0.999], not used).
    --agent.subgoal_discounts=[0.9,0.99,0.99,0.995,0.999]
    --agent.use_subgoal_dag=true         # cup / bowl pipelines -> close (fork use_subtask_dag=true)
)

EXTRA_vadar_cf=(--buffer.warmup_offline_ratio=0.5 --intervals.eval_interval_bc=10000)
EXTRA_sparse=(--buffer.start_training=10000 --agent.adv_strategy=conservative --intervals.eval_interval_bc=10000)
# The fork's HGC env ratcheted subgoals sequentially (no DAG for algo=ogpo).
EXTRA_hgc=(--buffer.start_training=10000 --intervals.eval_interval_bc=10000 --agent.use_subgoal_dag=false)
