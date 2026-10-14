# LIBERO-10 KITCHEN_SCENE8 (put both moka pots on the stove), SAC backbone.
# Fork launcher: paper_experiments/libero_kitchen8/sac_her.sh only (SAC-libero-VADAR).
# KITCHEN_SCENE8 is not in the paper, and the fork has no VaDaR / Sparse / HGC
# launcher for it; its HER runs never produced results. The settings below are the
# HER launcher's, which match the other LIBERO scenes; the VaDaR discount 0.995 is
# carried over from KITCHEN_SCENE3/4/6 by analogy.
BACKBONE=sac
SUITE=libero
ENV_NAME=libero_KITCHEN_SCENE8
# The fork's LIBERO HGC / iHGC runs (algo=sac + goal_conditioning_mode=cumulative)
# kept the env's own sparse reward: the LIBERO wrapper never added the +5 bump.
HGC_REWARD_MODE=sparse

TASK_FLAGS=(
    "${SAC_FORK_PINS[@]}"
    --env.name=$ENV_NAME
    --env.sparse=false
    --training.bc_pi_steps=200000
    --training.q_warmup_steps=10000
    --training.online_steps=1000000
    --buffer.start_training=10000
    --buffer.utd_warmup=1
    --buffer.utd_online=1
    --buffer.offline_ratio=0.0
    --buffer.warmup_offline_ratio=0.0
    --bc.clip_bc=true
    --bc.clip_bc_threshold=0.45
    --success_buffer.use_success_buffer=false
    --eval.n_eval_envs=16
    --eval.eval_episodes=2
    --eval.plot_q_vs_mc=false
    --intervals.eval_interval=20000
    --intervals.eval_interval_bc=20000
    --intervals.log_interval=5000
    --intervals.save_interval=500000
    --agent.horizon_length=8
    --agent.discount=0.995               # VaDaR runs; Sparse / HGC / iHGC / HER used 0.99
    --agent.tau=0.005
    --agent.num_qs=10
    --agent.q_agg=mean
    --agent.use_subgoal_dag=false
)

EXTRA_sparse=(--agent.discount=0.99)
EXTRA_hgc=(--agent.discount=0.99)
