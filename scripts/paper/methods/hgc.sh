# HGC: hierarchical goal conditioning. Sparse backbone; the cumulative subgoal
# vector is appended to the observation and both actor and critic condition on it.
# Fork: baseline_gc (goal_conditioning_mode=cumulative, goal_conditioned_actor/critic=true).
# Reward: the fork's GC reward ({-1,0} + 5 per newly completed subgoal) on
# Robomimic / RoboCasa; the env's own reward on LIBERO / OGBench (the task file
# sets HGC_REWARD_MODE).
ALGO=$SPARSE_ALGO
METHOD_FLAGS=(
    --env.goal_obs=cumulative
    --env.reward_mode=$HGC_REWARD_MODE
    --agent.goal_conditioned_actor=true
    --agent.goal_conditioned_critic=true
    --agent.goal_embed_dim=32
)
