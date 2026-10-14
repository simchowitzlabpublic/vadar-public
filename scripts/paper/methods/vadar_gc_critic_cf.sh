# VaDaR-CF with the goal-conditioned critic Q(s,a,g) (vadar_critic=goal_conditioned,
# the repo default), for the critic-parameterization ablation against the vector
# critic of vadar_cf. The fork's QIPM launchers in paper_experiments/ all used the
# vector critic; this variant reuses vadar_cf's per-task settings.
ALGO=$VADAR_ALGO
EXTRA_FROM=vadar_cf
METHOD_FLAGS=(
    --env.goal_obs=null
    --agent.weighting=current_future
    --agent.vadar_critic=goal_conditioned
    --agent.subgoal_embed_dim=32
)
