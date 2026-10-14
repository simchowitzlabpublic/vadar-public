# aggSAC on PushT-Obs: plain SAC on the aggregated scalar reward
# (15 if task_done else -1) + LAMBDA * (-5 cost)   (env.reward_mode=dense_parity).
# Set LAMBDA in the environment (default 1).
# Fork: pushT/sweep_dense_parity_3seeds.sh (algo=sac_dense_parity, dense_task_bonus 15).
if [ "$SUITE" != pusht ]; then
    echo "aggsac is for the PushT-Obs task (pusht_obstacle_sac)" >&2
    exit 1
fi
LAMBDA=${LAMBDA:-1.0}
TAG=${TAG:-lam$(echo "$LAMBDA" | tr . p)}
ALGO=$SPARSE_ALGO
METHOD_FLAGS=(
    --env.goal_obs=null
    --env.reward_mode=dense_parity
    --env.obstacle.dense_safety_coeff=$LAMBDA
)
