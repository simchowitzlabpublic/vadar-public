# HER: hindsight relabelling with object-state goals (SB3-style HerReplayBuffer,
# future strategy, k=4, 0.05 m success ball). Goal-space reward {-1, 0}; episode
# ends are truncations; each demo frame is labelled with the demo's final goal.
# Fork: libero_kitchen{3,4,6,8}/sac_her.sh and toolhang/toolhang_ogpo_her.sh
# (goal_conditioning_mode=state), run with the Sparse runs' settings.
ALGO=$SPARSE_ALGO
EXTRA_FROM=sparse
METHOD_FLAGS=(
    --env.goal_obs=state
    --env.reward_mode=sparse
    --agent.goal_conditioned_actor=true
    --agent.goal_conditioned_critic=true
    --agent.goal_embed_dim=32
    --agent.her_k=4
    --agent.her_strategy=future
    --agent.her_pos_threshold=0.05
    --agent.her_bc_goal=final
)
