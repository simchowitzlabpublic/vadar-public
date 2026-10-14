# HER, variant C ("latent"): goals are 16-D embeddings of the base observation from
# a frozen subgoal encoder (HER_ENCODER_PATH; the fork's
# data/libero/subgoal_encoder_k6.npz from its scripts/rollout/train_subgoal_classifier.py),
# with next-subgoal demo labels. LIBERO KITCHEN_SCENE6 only.
# Fork: libero_kitchen6/sac_her_latent.sh.
ALGO=$SPARSE_ALGO
EXTRA_FROM=sparse
HER_ENCODER_PATH=${HER_ENCODER_PATH:-$VADAR_ROOT/data/libero/subgoal_encoder_k6.npz}
METHOD_FLAGS=(
    --env.goal_obs=latent
    --env.reward_mode=sparse
    --agent.goal_conditioned_actor=true
    --agent.goal_conditioned_critic=true
    --agent.goal_embed_dim=32
    --agent.her_k=4
    --agent.her_strategy=future
    --agent.her_pos_threshold=0.05
    --agent.her_bc_goal=next
    --agent.her_encoder_path=$HER_ENCODER_PATH
)
