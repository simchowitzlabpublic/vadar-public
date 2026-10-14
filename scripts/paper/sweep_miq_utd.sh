#!/bin/bash
# MIQ critic-UTD sweep (paper App. "Combining VaDaR with Iterative Critics", Fig.
# "MIQ+VaDaR vs monolithic vector critics"): MIQ + VaDaR-NF (miq) vs the monolithic
# vector critic (vadar_nf) at critic UTD q in {1, 4}, with one actor update per env
# step. Fork: paper_experiments/mipQ-experiments/:
#   OGPO: high_dataset/transport/qipm/{low,high}_utd (transport_highdata_ogpo, MIQ and
#         monolithic) and transport/qipm/{low,high}_utd (transport_ogpo, MIQ only),
#         --buffer.utd_q=q --buffer.utd_pi=1;
#   SAC:  libero-kitchen-scene-{4,6} (libero_ks{4,6}_sac),
#         --buffer.utd_online=q --buffer.utd_q=q --buffer.utd_pi=1;
#   preSoakPan (presoakpan_ogpo) was also run but is not in the paper's figure.
# The high-data Transport sweep used bc_coeff 0.1 for both critics.
#
#   bash scripts/paper/sweep_miq_utd.sh            # submit (from the repo root)
#   DRY_RUN=1 bash scripts/paper/sweep_miq_utd.sh  # print the sbatch commands
set -euo pipefail
TASKS=${TASKS:-"transport_highdata_ogpo transport_ogpo libero_ks4_sac libero_ks6_sac"}
SEEDS=${SEEDS:-1-3}
UTDS=${UTDS:-"1 4"}

for task in $TASKS; do
    for method in miq vadar_nf; do
        for q in $UTDS; do
            flags="--buffer.utd_q=$q --buffer.utd_pi=1"
            case $task in
                *_sac) flags+=" --buffer.utd_online=$q" ;;
                transport_highdata_ogpo) flags+=" --agent.bc_coeff=0.1" ;;
            esac
            cmd="sbatch --array=$SEEDS --job-name=${method}_${task}_q$q --export=ALL,TAG=utdq$q"
            cmd+=" scripts/paper/launch.sh $task $method $flags"
            echo "+ $cmd"
            [ "${DRY_RUN:-0}" = "1" ] || eval "$cmd"
        done
    done
done
