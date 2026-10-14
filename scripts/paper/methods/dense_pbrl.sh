# Dense: a preference-based reward r(s) (Bradley-Terry on whole-episode success
# preferences, PEBBLE-style online updates) replaces the env reward of the sparse
# backbone. Reward-model settings: vadar/configs/pbrl/{reward_model,dense}.yaml
# (incl. the per-task rollout datasets and batch sizes).
# Fork: OGPO-pbrl-baseline/paper_experiments/pbrl/*_preference.sh, which used the
# Sparse runs' settings (hence EXTRA_FROM=sparse), with the interleaved warmup.
ALGO=$SPARSE_ALGO
EXTRA_FROM=sparse
METHOD_FLAGS=(
    --env.goal_obs=null
    --env.reward_mode=sparse
    --reward_model.baseline=dense
)
