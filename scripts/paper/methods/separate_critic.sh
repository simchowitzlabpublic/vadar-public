# VaDaR-NF with one independent critic per subgoal (vadar_critic=separate).
# Fork: understanding-experiments/separateCritic (separate_critic_per_subtask=true),
# run on PreSoakPan with the steelman schedule (task presoakpan_steelman_ogpo).
ALGO=$VADAR_ALGO
METHOD_FLAGS=(
    --env.goal_obs=null
    --agent.weighting=next_future
    --agent.vadar_critic=separate
)
