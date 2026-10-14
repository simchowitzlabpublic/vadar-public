# Steelman 2 (paper Table "steelman variants"): all-subgoal critic supervision,
# current-future extraction, goal-conditioned policy pi(a|s,g) (actor-only goal
# conditioning, vector critic).
# Fork: steelman-PreSoakPan/presoakpan_steelman_2.sh (advantage_gating_mode=goal_future).
ALGO=$VADAR_ALGO
METHOD_FLAGS=(
    --env.goal_obs=cumulative
    --agent.goal_conditioned_actor=true
    --agent.goal_conditioned_critic=false
    --agent.goal_embed_dim=32
    --agent.weighting=current_future
    --agent.vadar_critic=vector
)
