# VaDaR with next_future policy-extraction weighting (fork QIPM advantage_gating_mode=grandparent_future).
# Critic: the vector critic S x A -> R^|G| that every fork QIPM run used
# (vadar_critic=vector). For the goal-conditioned critic Q(s,a,g) see vadar_gc_critic_*.
ALGO=$VADAR_ALGO
METHOD_FLAGS=(
    --env.goal_obs=null
    --agent.weighting=next_future
    --agent.vadar_critic=vector
)
