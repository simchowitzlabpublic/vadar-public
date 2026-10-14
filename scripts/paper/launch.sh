#!/bin/bash
#SBATCH --partition=general
#SBATCH --job-name=vadar
#SBATCH --gres=gpu:1
#SBATCH --constraint=VRAM_48GB
#SBATCH --cpus-per-task=20
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.out
#SBATCH --requeue
#SBATCH --signal=B:TERM@120
#
# Launch one paper run: <task> x <method> x seed.
#
#   sbatch --array=1-3 scripts/paper/launch.sh <task> <method> [--section.key=value ...]
#   bash scripts/paper/launch.sh <task> <method> [...]          # seed 0 (or $SEED)
#   DRY_RUN=1 bash scripts/paper/launch.sh <task> <method> [...] # print the command only
#
# <task>   = scripts/paper/tasks/<task>.sh   (env + the fork's hyperparameters)
# <method> = scripts/paper/methods/<method>.sh (algo variant + method flags)
# Extra arguments are passed to vadar/main.py last, so they override everything.
#
# Environment variables (set them in your shell or ~/.bashrc):
#   VADAR_ROOT        repo root (default: $SLURM_SUBMIT_DIR under sbatch, else this script's repo)
#   VADAR_VENV        python venv to activate (bin/activate); VADAR_VENV_<SUITE> overrides it
#                     per suite (ROBOMIMIC, ROBOCASA, LIBERO, OGBENCH, PUSHT)
#   VADAR_SAVE_DIR    experiment.save_dir (default /data/user_data/$USER/vadar)
#   VADAR_WANDB_PROJECT  wandb project (default vadar-paper); VADAR_WANDB_ENTITY sets WANDB_ENTITY
#   LIBERO_CLONE      LIBERO checkout put on PYTHONPATH for LIBERO tasks
#   ROBOCASA_DATA_DIR / ROBOCASA_DATASET_PATH, OGBENCH_TASK4_BC_DIR, PUSHT_BC_DATA: see tasks/*.sh
#   SEED              seed (default: $SLURM_ARRAY_TASK_ID, else 0)
#   TAG               optional suffix for run names (e.g. utdq4)
set -euo pipefail

if [ $# -lt 2 ]; then
    echo "usage: $0 <task> <method> [--section.key=value ...]" >&2
    exit 1
fi
TASK=$1
METHOD=$2
shift 2

# --- Repo root. Under sbatch, BASH_SOURCE points at slurm's copy of this script,
# so submit from the repo root (or set VADAR_ROOT).
if [ -z "${VADAR_ROOT:-}" ]; then
    if [ -n "${SLURM_JOB_ID:-}" ]; then
        VADAR_ROOT=$SLURM_SUBMIT_DIR
    else
        VADAR_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
    fi
fi
if [ ! -f "$VADAR_ROOT/vadar/main.py" ]; then
    echo "VADAR_ROOT=$VADAR_ROOT is not the vadar repo; submit from the repo root or set VADAR_ROOT" >&2
    exit 1
fi
PAPER_DIR=$VADAR_ROOT/scripts/paper
SEED=${SEED:-${SLURM_ARRAY_TASK_ID:-0}}

# --- Task: sets BACKBONE, SUITE, ENV_NAME, TASK_FLAGS, HGC_REWARD_MODE, optional EXTRA_<name>
if [ ! -f "$PAPER_DIR/tasks/$TASK.sh" ]; then
    echo "unknown task '$TASK'; available: $(ls "$PAPER_DIR/tasks" | sed 's/\.sh$//' | tr '\n' ' ')" >&2
    exit 1
fi
HGC_REWARD_MODE=hgc
source "$PAPER_DIR/fork_pins.sh"    # OGPO_FORK_PINS / FQL_FORK_PINS / SAC_FORK_PINS
source "$PAPER_DIR/tasks/$TASK.sh"

# --- Method: sets ALGO, METHOD_FLAGS and (optionally) EXTRA_FROM
if [ ! -f "$PAPER_DIR/methods/$METHOD.sh" ]; then
    echo "unknown method '$METHOD'; available: $(ls "$PAPER_DIR/methods" | sed 's/\.sh$//' | tr '\n' ' ')" >&2
    exit 1
fi
SPARSE_ALGO=$BACKBONE              # ogpo | fql | sac
VADAR_ALGO=${BACKBONE}_vadar       # ogpo_vadar | fql_vadar | sac_vadar
EXTRA_FROM=$METHOD
source "$PAPER_DIR/methods/$METHOD.sh"

# Per-(task, method) deviations recorded in the task file (EXTRA_<method>, e.g. the
# fork's Sparse runs used a different q_agg); methods may share them via EXTRA_FROM.
EXTRA_FLAGS=()
if declare -p "EXTRA_${EXTRA_FROM}" >/dev/null 2>&1; then
    ref="EXTRA_${EXTRA_FROM}[@]"
    EXTRA_FLAGS=("${!ref}")
fi

# --- Experiment naming
RUN_NAME="${METHOD}_${TASK}${TAG:+_$TAG}"
COMMON_FLAGS=(
    --experiment.project="${VADAR_WANDB_PROJECT:-vadar-paper}"
    --experiment.run_group="$TASK"
    --experiment.run_name="$RUN_NAME"
    --experiment.wandb_name="${RUN_NAME}_seed${SEED}"
    --experiment.seed="$SEED"
    --experiment.save_dir="${VADAR_SAVE_DIR:-/data/user_data/$USER/vadar}"
)

# Later flags override earlier ones: task < method < per-(task, method) extras < CLI.
# (The ${a[@]+"${a[@]}"} form keeps empty arrays safe under `set -u` on old bash.)
CMD=(python -u vadar/main.py --algo="$ALGO"
     "${COMMON_FLAGS[@]}"
     ${TASK_FLAGS[@]+"${TASK_FLAGS[@]}"}
     ${METHOD_FLAGS[@]+"${METHOD_FLAGS[@]}"}
     ${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"}
     "$@")

if [ "${DRY_RUN:-0}" = "1" ]; then
    printf '%q ' "${CMD[@]}"
    echo
    exit 0
fi

# --- Environment
SUITE_UPPER=$(echo "$SUITE" | tr '[:lower:]' '[:upper:]')
venv_var="VADAR_VENV_${SUITE_UPPER}"
# Default: the uv environments built by scripts/setup_env.sh (LIBERO has its own).
if [ "$SUITE" = libero ]; then default_venv="$VADAR_ROOT/.venv-libero"; else default_venv="$VADAR_ROOT/.venv"; fi
VENV=${!venv_var:-${VADAR_VENV:-$default_venv}}
if [ ! -f "$VENV/bin/activate" ]; then
    echo "no environment at $VENV: run scripts/setup_env.sh or set VADAR_VENV / $venv_var" >&2
    exit 1
fi
source "$VENV/bin/activate"

export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
[ -n "${VADAR_WANDB_ENTITY:-}" ] && export WANDB_ENTITY=$VADAR_WANDB_ENTITY
export PYTHONPATH="$VADAR_ROOT:${PYTHONPATH:-}"
if [ "$SUITE" = libero ]; then
    # LIBERO's packaging is incomplete upstream; setup_env.sh registers the checkout with a .pth file (also added here).
    LIBERO_CLONE=${LIBERO_CLONE:-$VADAR_ROOT/third_party/LIBERO}
    export PYTHONPATH="$LIBERO_CLONE:$PYTHONPATH"
fi
if [ "$SUITE" = robocasa ]; then
    : "${ROBOCASA_DATASET_PATH:?set ROBOCASA_DATASET_PATH (tasks/*.sh derive it from ROBOCASA_DATA_DIR)}"
    export ROBOCASA_DATASET_PATH
fi

cd "$VADAR_ROOT"
mkdir -p logs
echo "=== $TASK | $METHOD | algo=$ALGO | seed=$SEED | node=${SLURMD_NODENAME:-local} | job=${SLURM_JOB_ID:-none} ==="
printf '%q ' "${CMD[@]}"
echo
exec "${CMD[@]}"
