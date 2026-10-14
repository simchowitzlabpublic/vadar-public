# OGBench scene-play-singletask-task4 (put_in_drawer), SAC backbone.
# Fork launchers: paper_experiments/scene_task4_bc/*.sh (SCENE_TASK4_GC_VADAR,
# run group bc_from_ckpt2_2M), seeds 1-3. (The older OGBench-sac/ folder, 5M online
# steps on the stock OGBench data, is not the paper's setting.)
#
# BC data: not the stock OGBench dataset but rollouts of a 2M-step SAC-VaDaR
# checkpoint (fork run noGC_future_utd4_s1, collected with the fork's
# scripts/rollout/collect_scene_task4_bc_data.py). Point OGBENCH_TASK4_BC_DIR at the
# directory of .npz files.
BACKBONE=sac
SUITE=ogbench
ENV_NAME=scene-play-singletask-task4-v0
# The fork's OGBench HGC / iHGC runs kept the env's singletask reward (no GC reward).
HGC_REWARD_MODE=sparse
OGBENCH_TASK4_BC_DIR=${OGBENCH_TASK4_BC_DIR:-$VADAR_ROOT/data/ogbench/scene_task4_bc}

TASK_FLAGS=(
    "${SAC_FORK_PINS[@]}"
    --env.name=$ENV_NAME
    --env.sparse=false
    --env.max_episode_steps=100          # TimeLimit override (paper: episode length 100)
    --dataset.ogbench_dir=$OGBENCH_TASK4_BC_DIR
    --training.bc_pi_steps=5000
    --training.q_warmup_steps=10000
    --training.online_steps=300000
    --buffer.start_training=10000
    --buffer.utd_warmup=4
    --buffer.utd_online=1
    --buffer.offline_ratio=0.0
    --buffer.warmup_offline_ratio=0.0
    --bc.clip_bc=true
    --bc.clip_bc_threshold=0.3
    --success_buffer.use_success_buffer=false
    --eval.n_eval_envs=32
    --eval.eval_episodes=2
    --intervals.eval_interval=5000
    --intervals.eval_interval_bc=500
    --intervals.log_interval=5000
    --intervals.save_interval=500000
    --agent.horizon_length=4
    --agent.discount=0.9
    --agent.tau=0.005
    --agent.num_qs=10
    --agent.q_agg=mean
    --agent.use_subgoal_dag=false
)
