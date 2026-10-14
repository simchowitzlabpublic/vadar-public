# HER, variant B ("nextbc"): as her.sh, but each demo frame is labelled with the
# demo's next-subgoal goal (her_bc_goal=next). LIBERO KITCHEN_SCENE6 only.
# Fork: libero_kitchen6/sac_her_nextbc.sh.
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
    --agent.her_bc_goal=next
)
