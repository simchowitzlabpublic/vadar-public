# Steelman 1 (paper Table "steelman variants"): all-subgoal critic supervision,
# current-myopic extraction, goal-conditioned policy pi(a|s,g). VaDaR agent with the
# cumulative goal appended to the observation and seen by the actor only; vector
# critic as in the fork.
# Fork: steelman-PreSoakPan/presoakpan_steelman_3.sh (advantage_gating_mode=goal_myopic).
# The fork's file numbering is reversed w.r.t. the paper: fork steelman_1 = NF
# (paper Steelman 3), fork steelman_3 = GM (paper Steelman 1).
ALGO=$VADAR_ALGO
METHOD_FLAGS=(
    --env.goal_obs=cumulative
    --agent.goal_conditioned_actor=true
    --agent.goal_conditioned_critic=false
    --agent.goal_embed_dim=32
    --agent.weighting=current_myopic
    --agent.vadar_critic=vector
)
