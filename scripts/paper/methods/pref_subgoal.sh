# Pref-Subgoal: a preference reward r(s, g) per subgoal (per-subgoal Bradley-Terry
# labels from subgoal_dones), aggregated with next-future weights and used as the
# reward of the sparse backbone's scalar critic.
# Settings: vadar/configs/pbrl/{reward_model,pref_subgoal}.yaml.
# Fork: OGPO-pbrl-subgoal/paper_experiments/pbrl-subgoal/*_subgoal_gated.sh, run
# with the Sparse runs' settings.
ALGO=$SPARSE_ALGO
EXTRA_FROM=sparse
METHOD_FLAGS=(
    --env.goal_obs=null
    --env.reward_mode=sparse
    --reward_model.baseline=pref_subgoal
)
