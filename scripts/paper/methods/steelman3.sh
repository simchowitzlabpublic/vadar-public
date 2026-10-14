# Steelman 3 (paper Table "steelman variants"): all-subgoal critic supervision,
# next-future extraction, goal-conditioned policy pi(a|s,g) (actor-only goal
# conditioning, vector critic).
# Fork: steelman-PreSoakPan/presoakpan_steelman_1.sh (advantage_gating_mode=grandparent_future);
# see steelman1.sh for the numbering.
ALGO=$VADAR_ALGO
METHOD_FLAGS=(
    --env.goal_obs=cumulative
    --agent.goal_conditioned_actor=true
    --agent.goal_conditioned_critic=false
    --agent.goal_embed_dim=32
    --agent.weighting=next_future
    --agent.vadar_critic=vector
)
