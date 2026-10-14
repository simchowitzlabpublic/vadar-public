# RoboCasa PreSoakPan, OGPO backbone, the schedule of the steelman sweep
# (paper Fig. "Steelman sweep"), which also hosted the LastQ ("last goal only") and
# separate-critic runs on PreSoakPan.
# Fork launchers: paper_experiments/steelman-PreSoakPan/*.sh and
# understanding-experiments/{lastGoalOnly,separateCritic}/PreSoakPan/*.sh.
# Differences from presoakpan_ogpo.sh: 500K BC steps, start_training 20K for every
# method (incl. HGC), shorter online phase.
BACKBONE=ogpo
SUITE=robocasa
ENV_NAME=robocasa_PreSoakPan
HGC_REWARD_MODE=hgc
ROBOCASA_DATASET_PATH=${PRESOAKPAN_DATASET:-${ROBOCASA_DATA_DIR:-$VADAR_ROOT/data/robocasa}/PreSoakPan/PreSoakPan_200_success_p9_28_full_obs.hdf5}

TASK_FLAGS=(
    "${OGPO_FORK_PINS[@]}"
    --env.name=$ENV_NAME
    --env.anchor_demo=demo_0
    --dataset.proportion=1.0
    --training.bc_pi_steps=500000
    --training.q_warmup_steps=20000
    --training.online_steps=3000000      # steelman runs; HGC / VaDaR-CF / VaDaR-NF / LastQ / separate used 2.5M
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
    --agent.use_subgoal_dag=true
)

EXTRA_hgc=(--training.online_steps=2500000)          # presoakpan_gc.sh (vanilla advantages)
EXTRA_vadar_nf=(--training.online_steps=2500000)
EXTRA_vadar_cf=(--training.online_steps=2500000)
EXTRA_vadar_lastq=(--training.online_steps=2500000)  # lastGoalOnly
EXTRA_separate_critic=(--training.online_steps=2500000)
