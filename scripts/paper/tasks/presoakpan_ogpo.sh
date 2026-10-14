# RoboCasa PreSoakPan, OGPO backbone (main results and weighting ablation).
# Fork launchers: paper_experiments/preSoakPan/*.sh (QIPM_PRESOAKPAN_GC).
# The steelman / LastQ / separate-critic runs used a different schedule: see
# presoakpan_steelman_ogpo.sh.
# Note: the paper says bc_coeff 1 and 500K BC steps for RoboCasa; the launchers
# used the fork default bc_coeff 0.1 and 300K (kept here).
BACKBONE=ogpo
SUITE=robocasa
ENV_NAME=robocasa_PreSoakPan
HGC_REWARD_MODE=hgc        # fork RoboCasa GC wrapper: {-1,0} reward + 5 per newly completed subgoal
ROBOCASA_DATASET_PATH=${PRESOAKPAN_DATASET:-${ROBOCASA_DATA_DIR:-$VADAR_ROOT/data/robocasa}/PreSoakPan/PreSoakPan_200_success_p9_28_full_obs.hdf5}

TASK_FLAGS=(
    "${OGPO_FORK_PINS[@]}"
    --env.name=$ENV_NAME
    --env.anchor_demo=demo_0             # every reset uses demo_0's layout
    --dataset.proportion=1.0
    --training.bc_pi_steps=300000
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
    --agent.subgoal_discounts=[0.9,0.99,0.99,0.995,0.999]
    --agent.use_subgoal_dag=true         # PreSoakPan is a chain (no DAG in envs/subgoals.py)
)

# Fork VaDaR-CF run mixed offline data into the critic warmup.
EXTRA_vadar_cf=(--buffer.warmup_offline_ratio=0.5)
EXTRA_sparse=(--buffer.start_training=10000 --agent.adv_strategy=conservative)
EXTRA_hgc=(--buffer.start_training=10000)
# mipQ-experiments/preSoakPan (MIQ UTD sweep) used the VaDaR-NF settings above.
