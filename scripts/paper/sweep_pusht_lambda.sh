#!/bin/bash
# PushT-Obs lambda sweep (paper App. "Multi-Objective Optimization on PushT-Obs"):
# VaDaR (2-head critic, weights (1, lambda)) vs aggSAC (scalar reward with weight
# lambda on the cost), 3 seeds each. Fork: paper_experiments/pushT/submit_pushT_sweep.sh
# (lambda in {0.1, 0.5, 5, 10, 50}, seeds 1-3) plus earlier lambda = 1 runs; the
# paper's figure shows lambda in {0.1, 1, 5, 10}.
#
#   bash scripts/paper/sweep_pusht_lambda.sh              # submit (from the repo root)
#   DRY_RUN=1 bash scripts/paper/sweep_pusht_lambda.sh    # print the sbatch commands
#
# Logs are named <prefix>_lam<TAG>_seed<S>_<job>.out for scripts/pusht/plot_lambda_sweep.py.
set -euo pipefail
LAMBDAS=${LAMBDAS:-"0.1 0.5 1 5 10 50"}
SEEDS=${SEEDS:-1-3}
SBATCH_ARGS=${SBATCH_ARGS:-"--cpus-per-task=12 --mem=48G --time=24:00:00"}

mkdir -p logs
for lam in $LAMBDAS; do
    tag=$(echo "$lam" | tr . p)
    for method in vadar_multiobj aggsac; do
        prefix=sac_vadar_pusht
        [ "$method" = aggsac ] && prefix=sac_aggsac_pusht
        cmd="sbatch $SBATCH_ARGS --array=$SEEDS --job-name=${prefix}_lam${tag}"
        cmd+=" --export=ALL,LAMBDA=$lam --output=logs/${prefix}_lam${tag}_seed%a_%A.out"
        cmd+=" --error=logs/${prefix}_lam${tag}_seed%a_%A.out"
        cmd+=" scripts/paper/launch.sh pusht_obstacle_sac $method"
        echo "+ $cmd"
        [ "${DRY_RUN:-0}" = "1" ] || eval "$cmd"
    done
done
