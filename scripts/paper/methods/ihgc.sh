# iHGC: implicit HGC. As HGC, but the policy is not goal conditioned (only the critic is).
# Fork: gc_critic_only.
ALGO=$SPARSE_ALGO
EXTRA_FROM=hgc     # the fork ran HGC and iHGC with identical settings
METHOD_FLAGS=(
    --env.goal_obs=cumulative
    --env.reward_mode=$HGC_REWARD_MODE
    --agent.goal_conditioned_actor=false
    --agent.goal_conditioned_critic=true
    --agent.goal_embed_dim=32
)
