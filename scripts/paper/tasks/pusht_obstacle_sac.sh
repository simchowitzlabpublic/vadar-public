# PushT-Obs (multi-objective PushT: task + obstacle cost), SAC backbone.
# Fork launchers (pushT branch, OGPO@e08e82b): paper_experiments/pushT/
# sweep_{qipm,dense_parity}_3seeds.sh, seeds 1-3. See docs/PUSHT_MULTIOBJECTIVE.md.
# Methods: vadar_multiobj (2-head critic, weights (1, LAMBDA)) and aggsac.
#
# Obstacle: the paper's circle (320, 220, r=30, 10 px margin) is the default for
# the -obstacle env name (envs/pusht_utils.py PUSHT_OBSTACLE_DEFAULTS), incl.
# dense_task_bonus 15 for aggSAC.
# Note: discount 0.95 (the paper text says 0.99) and q_warmup_steps 0, as run.
BACKBONE=sac
SUITE=pusht
ENV_NAME=pusht-keypoints-obstacle-v0
PUSHT_BC_DATA=${PUSHT_BC_DATA:-$VADAR_ROOT/data/pusht/sac_pusht_2000000_bc.npz}

TASK_FLAGS=(
    "${SAC_FORK_PINS[@]}"
    --env.name=$ENV_NAME
    --env.sparse=false
    --dataset.bc_dataset_path=$PUSHT_BC_DATA
    --training.bc_pi_steps=150000
    --training.q_warmup_steps=0
    --training.online_steps=800000
    --buffer.start_training=10000
    --buffer.utd_warmup=1
    --buffer.utd_online=1
    --buffer.offline_ratio=0.0
    --buffer.warmup_offline_ratio=0.0
    --bc.clip_bc=false
    --success_buffer.use_success_buffer=false
    --eval.n_eval_envs=32
    --eval.eval_episodes=2
    --eval.plot_q_vs_mc=false
    --intervals.eval_interval=20000
    --intervals.eval_interval_bc=20000
    --intervals.log_interval=5000
    --intervals.save_interval=500000
    --agent.horizon_length=4
    --agent.discount=0.95
    --agent.tau=0.005
    --agent.num_qs=10
    --agent.q_agg=mean
)
