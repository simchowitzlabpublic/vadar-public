# VaDaR on PushT-Obs: SAC-VaDaR with a 2-head critic (task head, obstacle-cost
# head) and constant weights (1, LAMBDA). Set LAMBDA in the environment (default 1).
# Fork: pushT/sweep_qipm_3seeds.sh (algo=sac_qipm_avoid_obstacle, safety_coeff=LAMBDA,
# vector critic). Head types [subgoal, cost] come from envs/subgoals.py.
if [ "$SUITE" != pusht ]; then
    echo "vadar_multiobj is for the PushT-Obs task (pusht_obstacle_sac)" >&2
    exit 1
fi
LAMBDA=${LAMBDA:-1.0}
TAG=${TAG:-lam$(echo "$LAMBDA" | tr . p)}
ALGO=$VADAR_ALGO
METHOD_FLAGS=(
    --env.goal_obs=null
    --env.reward_mode=sparse
    --agent.weighting=constant
    --agent.subgoal_weights=[1.0,$LAMBDA]
    --agent.vadar_critic=vector
)
