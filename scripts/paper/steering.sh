#!/bin/bash
#SBATCH --partition=general
#SBATCH --job-name=steer
#SBATCH --gres=gpu:1
#SBATCH --constraint=VRAM_48GB
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.out
#SBATCH --requeue
#SBATCH --signal=B:TERM@120
#
# Test-time steering, critic training (paper Sec. "Test-Time Steering with a Frozen
# Policy", App. "Test-Time Policy Steering"). Warm-starts from a BC checkpoint,
# never updates the policy, and trains only the critic:
#   * the actor is restored from BC_CKPT (the end-of-BC params_<bc steps>.pkl that
#     every OGPO run saves; the fork used its Sparse OGPO run's BC checkpoint, seed 2
#     for ArrangeTea, seed 200 for ToolHang) and BC is skipped (bc_pi_steps=0,
#     ep_resume=<bc steps> so step counters continue);
#   * q_warmup_steps (400K) > online_steps (300K): the whole online phase is the
#     interleaved critic-only warmup. The first start_training=20K env steps only
#     seed the buffer; then 4 critic updates per env step (utd_warmup=4) on batches
#     half offline demos, half frozen-policy rollouts (warmup_offline_ratio=0.5);
#   * a checkpoint every 20K steps; no online evaluation.
# The paper's critic budgets 40K / 100K / 200K / 280K are the checkpoints
# params_<ep_resume + 20K + budget>.pkl, scored by best-of-N with
# scripts/steering/steering_eval.py (see scripts/paper/README.md).
#
#   sbatch --array=0-2 scripts/paper/steering.sh <toolhang|arrangetea|presoakpan> <sparse|vadar_nf> <BC_CKPT>
#   DRY_RUN=1 bash scripts/paper/steering.sh toolhang vadar_nf /path/params_300000.pkl
#
# Fork launchers: paper_experiments/steering_warmstart/{toolhang,arrangetea}_{ogpo,qipm}_ws2.sh
# and presoakpan_{ogpo,qipm}_wsold2.sh (the paper's runs), seeds 0-2.
set -euo pipefail

if [ $# -lt 3 ]; then
    echo "usage: $0 <toolhang|arrangetea|presoakpan> <sparse|vadar_nf> <BC_CKPT> [--section.key=value ...]" >&2
    exit 1
fi
STEER_TASK=$1
CRITIC=$2
BC_CKPT=$3
shift 3

if [ -z "${VADAR_ROOT:-}" ]; then
    if [ -n "${SLURM_JOB_ID:-}" ]; then VADAR_ROOT=$SLURM_SUBMIT_DIR
    else VADAR_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd); fi
fi
export VADAR_ROOT

case $CRITIC in
    sparse|vadar_nf) ;;
    *) echo "critic must be sparse or vadar_nf" >&2; exit 1 ;;
esac

STEER_FLAGS=(
    --checkpoint.restore_actor_path=$BC_CKPT
    --training.bc_pi_steps=0
    --training.q_warmup_steps=400000
    --training.online_steps=300000
    --buffer.q_warmup_mode=interleaved
    --buffer.start_training=20000
    --buffer.utd_warmup=4
    --buffer.warmup_offline_ratio=0.5
    --intervals.eval_interval=10000000
    --intervals.eval_interval_bc=20000
    --intervals.save_interval=20000
    --eval.n_eval_envs=1
    --eval.eval_episodes=1
)

case $STEER_TASK in
    toolhang)      # ws2: BC checkpoint of the ToolHang OGPO run (300K BC steps)
        TASK=toolhang_ogpo
        STEER_FLAGS+=(--checkpoint.ep_resume=300000)
        ;;
    arrangetea)    # ws2: BC checkpoint of the ArrangeTea OGPO run (500K BC steps), demo_0 layout
        TASK=arrangetea_ogpo
        STEER_FLAGS+=(--checkpoint.ep_resume=500000 --env.anchor_demo=demo_0)
        ;;
    presoakpan)    # wsold2: an older PreSoakPan OGPO BC checkpoint (500K BC steps) with a
                   # wider actor and no denoiser
        TASK=presoakpan_ogpo
        STEER_FLAGS+=(
            --checkpoint.ep_resume=500000
            --agent.actor_hidden_dims=[1024,1024,512,512,512]
            --agent.use_denoiser=false
        )
        ;;
    *) echo "steering task must be toolhang, arrangetea or presoakpan" >&2; exit 1 ;;
esac

# The fork's Sparse steering critics are its Sparse OGPO runs' critics (conservative
# advantages; ToolHang also q_agg=subsample), which launch.sh applies via EXTRA_sparse.
# The VaDaR critic is VaDaR-NF with the vector critic (fork qipm, grandparent_future).
TAG=${TAG:-steer} exec bash "$VADAR_ROOT/scripts/paper/launch.sh" "$TASK" "$CRITIC" "${STEER_FLAGS[@]}" "$@"
