# VaDaR-Shaped: the per-subgoal rewards summed into one shaped reward,
# sum_g (+5 if d_g else -1), before the Bellman backup, trained with the sparse
# backbone's scalar critic (env.reward_mode=shaped).
# No launcher for these runs is in the fork's paper_experiments/; this uses the
# task's VaDaR settings.
ALGO=$SPARSE_ALGO
METHOD_FLAGS=(
    --env.goal_obs=null
    --env.reward_mode=shaped
)
