# Sparse: the backbone (OGPO / FQL / SAC) on the task's sparse reward.
# Fork: ogpo_conservative / sac launchers.
ALGO=$SPARSE_ALGO
METHOD_FLAGS=(
    --env.reward_mode=sparse
    --env.goal_obs=null
)
