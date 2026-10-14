# RoboCasa ArrangeTea, OGPO backbone.
# Fork launchers: paper_experiments/arrangeTea/*.sh (QIPM_ARRANGETEA_GC).
# Note: the fork's runs did not all use the same layout: VaDaR-NF / GM / LastQ and
# Sparse reset to demo_0's layout, while VaDaR-CF, HGC and iHGC used demo_11.
# HGC / iHGC also used discount 0.999 and 3M online steps.
BACKBONE=ogpo
SUITE=robocasa
ENV_NAME=robocasa_ArrangeTea
HGC_REWARD_MODE=hgc
ROBOCASA_DATASET_PATH=${ARRANGETEA_DATASET:-${ROBOCASA_DATA_DIR:-$VADAR_ROOT/data/robocasa}/ArrangeTea/ArrangeTea_VLA_single150.hdf5}

TASK_FLAGS=(
    "${OGPO_FORK_PINS[@]}"
    --env.name=$ENV_NAME
    --env.anchor_demo=demo_0
    --dataset.proportion=1.0
    --training.bc_pi_steps=500000
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
    --intervals.save_interval=1000000
    --agent.horizon_length=8
    --agent.discount=0.995
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
    --agent.subgoal_discounts=[0.9,0.99,0.99,0.995,0.999,0.9995]
    --agent.use_subgoal_dag=true         # ArrangeTea is a chain
)

EXTRA_vadar_cf=(--env.anchor_demo=demo_11 --buffer.warmup_offline_ratio=0.5 --intervals.save_interval=-1)
EXTRA_sparse=(--buffer.start_training=10000 --agent.adv_strategy=conservative --intervals.save_interval=-1)
EXTRA_hgc=(
    --env.anchor_demo=demo_11 --agent.discount=0.999 --training.online_steps=3000000
    --buffer.start_training=10000 --intervals.save_interval=-1
)
