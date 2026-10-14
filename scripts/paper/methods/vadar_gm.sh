# VaDaR with current_myopic policy-extraction weighting (fork QIPM advantage_gating_mode=goal_myopic).
# Critic: the vector critic S x A -> R^|G| that every fork QIPM run used
# (vadar_critic=vector). For the goal-conditioned critic Q(s,a,g) see vadar_gc_critic_*.
ALGO=$VADAR_ALGO
METHOD_FLAGS=(
    --env.goal_obs=null
    --agent.weighting=current_myopic
    --agent.vadar_critic=vector
)
