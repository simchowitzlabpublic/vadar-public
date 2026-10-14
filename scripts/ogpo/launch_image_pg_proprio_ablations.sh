#!/bin/bash
# Launch image-based (PaliGemma frozen backbone) + proprio-only runs for the
# six ablation algorithms across the three Robomimic image envs.
#
#   Algorithms (--algos):
#     dsrl           -> DSRL  (BoN over a flow-policy noise distribution)
#     expo           -> EXPO  (expressive policy distillation)
#     qc             -> QC / ACFQL (action-chunked FQL)
#     fql            -> FQL   (--algo=ogpo + use_one_step_policy=true)
#     ogpo_fpo_pp    -> OFPO++ in the paper plots (--algo=ogpo + use_fpo=true + fpo_per_sample=true)
#     ogpo_awr_nn    -> AW-OGPO-NN in the paper plots
#                       (--algo=ogpo + use_awr=true + awr_mode=positive_only +
#                        adv_strategy=awr + awr_beta=10.0 + awr_weight_max=20.0)
#
#   Environments (--envs):
#     square         (square-mh-image,    online_steps=2M, discount=0.99 )
#     toolhang       (tool_hang-ph-image, online_steps=3M, discount=0.999)
#     transport      (transport-mh-image, online_steps=3M, discount=0.999)
#
#   Every job runs:
#     * PaliGemma SigLIP+Gemma frozen VL encoder (env-conditioned text prompt)
#     * Two-tier obs encoder (image-feat tower + proprio tower fused before trunk)
#     * use_state=proprio (9D for square/can/toolhang, 18D for transport)
#
# Usage:
#   scripts/ogpo/launch_image_pg_proprio_ablations.sh                    # dry-run summary
#   scripts/ogpo/launch_image_pg_proprio_ablations.sh --submit           # actually sbatch all 18
#   scripts/ogpo/launch_image_pg_proprio_ablations.sh --submit \
#       --algos=ogpo_awr_nn,ogpo_fpo_pp --envs=square --seeds=82723,82724
#
# Notes:
#   * Each job requests 2 GPUs (1 training, 1 PaliGemma encoder) on VRAM_48GB.
#   * Encoder checkpoint must already exist at
#       /data/user_data/$USER/physical_intelligence/pi05_libero/params
#     (gcloud storage cp -r gs://openpi-assets/checkpoints/pi05_libero/ ...).
#   * On `--submit`, each invocation is sbatch'd with `--wrap` so this file
#     stays a single launcher and the per-job SBATCH directives are inline.

set -e

# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------
SUBMIT=false
ALGOS=("dsrl" "expo" "qc" "fql" "ogpo_fpo_pp" "ogpo_awr_nn")
ENVS=("square" "toolhang" "transport")
SEEDS=(82723)
PARTITION="general"
CONSTRAINT="VRAM_48GB"
TIME="48:00:00"
EXTRA_ARGS=()

for arg in "$@"; do
  case $arg in
    --submit)           SUBMIT=true ;;
    --algos=*)          IFS=',' read -ra ALGOS    <<< "${arg#*=}" ;;
    --envs=*)           IFS=',' read -ra ENVS     <<< "${arg#*=}" ;;
    --seeds=*)          IFS=',' read -ra SEEDS    <<< "${arg#*=}" ;;
    --partition=*)      PARTITION="${arg#*=}" ;;
    --constraint=*)     CONSTRAINT="${arg#*=}" ;;
    --time=*)           TIME="${arg#*=}" ;;
    --*=*|--*)          EXTRA_ARGS+=("$arg") ;;  # forwarded verbatim to python vadar/main.py
    *)                  echo "Unknown argument: $arg" >&2; exit 1 ;;
  esac
done

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    PROJECT_ROOT="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
    PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
fi
mkdir -p "$PROJECT_ROOT/logs"

# ---------------------------------------------------------------------------
# Encoder checkpoint
# ---------------------------------------------------------------------------
RESTORE_ENCODER_PATH="/data/user_data/$USER/physical_intelligence/pi05_libero/params"
if [ ! -d "$RESTORE_ENCODER_PATH" ]; then
    echo "WARNING: pi05_libero checkpoint not found at $RESTORE_ENCODER_PATH"
    echo "  Jobs will fail unless this exists. Download with:"
    echo "    gcloud storage cp -r gs://openpi-assets/checkpoints/pi05_libero/ \\"
    echo "        /data/user_data/\$USER/physical_intelligence/"
fi

# ---------------------------------------------------------------------------
# Per-env settings (training horizon + discount come from the low-dim
# ablation launcher; image runs reuse them).
# ---------------------------------------------------------------------------
env_name_for() {
    case "$1" in
        square)    echo "square-mh-image" ;;
        toolhang)  echo "tool_hang-ph-image" ;;
        transport) echo "transport-mh-image" ;;
        *) echo "ERROR: unknown env '$1'" >&2; exit 1 ;;
    esac
}
online_steps_for() {
    case "$1" in
        square)    echo 2000000 ;;
        toolhang)  echo 3000000 ;;
        transport) echo 3000000 ;;
    esac
}
discount_for() {
    case "$1" in
        square)    echo 0.99 ;;
        toolhang)  echo 0.999 ;;
        transport) echo 0.999 ;;
    esac
}

# ---------------------------------------------------------------------------
# Shared args (image + PaliGemma + proprio). These match the active
# square_pg_libero.sh configuration on this branch.
# ---------------------------------------------------------------------------
SAVE_DIR="/data/user_data/$USER/ogpo"

COMMON_ARGS=(
    # Training phases
    --training.bc_pi_steps=500000
    --training.bc_q_steps=0
    --training.calql_steps=0
    --training.q_warmup_steps=0
    --training.bc_refine_steps=0
    --training.bc_only=false
    --training.force_jax_sync=true
    --training.gpu_mode=reserve:1

    # Buffer
    --buffer.size=2000000
    --buffer.start_training=20000
    --buffer.utd_warmup=1
    --buffer.utd_online=1
    --buffer.utd_q=1
    --buffer.utd_pi=1
    --buffer.offline_ratio=0.0

    # Env
    --env.sparse=false
    --env.post_success_steps=1

    # Eval
    --eval.n_eval_envs=1
    --eval.eval_episodes=20
    --eval.plot_q_vs_mc=false
    --intervals.log_interval=5000
    --intervals.eval_interval=20000
    --intervals.eval_interval_bc=20000
    --intervals.save_interval=500000

    # BC
    --bc.clip_bc=true
    --bc.clip_bc_threshold=0.45
    --bc.clip_bc_wrt=SDE

    # Success buffer
    --success_buffer.use_success_buffer=true
    --agent.use_success_buffer_q=true

    # Learning rates
    --agent.lr=3e-4
    --agent.actor_lr=3e-4
    --agent.critic_lr=3e-4
    --agent.ppo_lr=4.5e-5
    --agent.clip_grad_norm=1000.0

    # Core
    --agent.horizon_length=4
    --agent.tau=0.05
    --agent.actor_tau=0.005
    --agent.best_of_n=8
    --agent.num_qs=10
    --agent.q_agg=subsample
    --agent.subsample_bon=true
    --agent.flow_steps=10
    --agent.ft_flow_steps=10
    --agent.time_embedding=sinusoidal
    --agent.time_embedding_dim=32
    --agent.bc_coeff=1.0

    # PPO / group
    --agent.ppo_batch_size=256
    --agent.clip_epsilon=0.01
    --agent.entropy_coeff=0.0
    --agent.group_num_samples=32

    # LR schedules
    --agent.use_constant_scheduler_for_bc=false
    --agent.actor_scheduler=cosine
    --agent.actor_warmup_steps=2000
    --agent.actor_decay_steps=50000
    --agent.actor_end_value=2e-5
    --agent.actor_weight_decay=1e-6
    --agent.critic_scheduler=constant
    --agent.critic_warmup_steps=500
    --agent.critic_decay_steps=5000
    --agent.critic_end_value=1e-8
    --agent.critic_weight_decay=1e-6

    # SDE settings (OGPO score-correction defaults from this branch)
    --agent.use_bc_regularization=true
    --agent.error_correct_ode_to_sde=false
    --agent.use_denoiser=false
    --agent.use_constant_noise=false
    --agent.use_tapered_noise=true
    --agent.error_correct_sde_to_ode=true
    --agent.constant_noise_std=0.1
    --agent.min_noise_std=0.01
    --agent.max_noise_std=0.01

    # Architecture (matches active square_pg_libero.sh)
    --agent.actor_backbone=mlp
    --agent.critic_backbone=mlp
    --agent.actor_layer_norm=false
    --agent.layer_norm=true
    --agent.actor_hidden_dims=[512,512,512,512]
    --agent.value_hidden_dims=[512,512,512,512]

    # Q variance reduction
    --agent.q_variance_reduction=false
    --agent.q_vr_num_samples=8
    --agent.q_vr_reduction=mean

    # MC regression / pessimism off by default for these ablations
    --agent.mc_regression=false
    --agent.chi_po=false
    --agent.kl_reg=false
    --agent.fwd_kl_reg=false

    # Vision (PaliGemma frozen, two-tier obs encoder, proprio-only state)
    --agent.actor_obs=image
    --agent.critic_obs=image
    --agent.vit_encoder=paligemma
    --agent.vit_img_proj_dim=0
    --agent.vit_state_proj_dim=0
    --agent.freeze_vision_encoder=true
    --agent.obs_two_tier=true
    --agent.two_tier_fused_dim=0
    --agent.use_state=proprio
    --agent.encoder_device=1
    --checkpoint.restore_encoder_path="$RESTORE_ENCODER_PATH"

    # Data aug off for frozen encoder
    --dataset.p_aug=0.0
)

# ---------------------------------------------------------------------------
# Per-algo overrides. Each function echoes the algo's CLI args (space-
# separated). The caller appends them after COMMON_ARGS so the algo can
# override any common default (e.g. adv_strategy for AW-OGPO-NN).
# ---------------------------------------------------------------------------
algo_cli_for() {
    case "$1" in
        dsrl)
            echo "--algo=dsrl --agent.noise_chunk_length=-1 --agent.action_magnitude=1.5"
            ;;
        expo)
            echo "--algo=expo"
            ;;
        qc)
            echo "--algo=qc"
            ;;
        fql)
            # FQL is OGPO with one-step distillation
            echo "--algo=ogpo --agent.use_one_step_policy=true --agent.fql_distillation_coeff=1000.0"
            ;;
        ogpo_fpo_pp)
            # OFPO++ in the paper: FPO with per-sample ratio clipping
            echo "--algo=ogpo --agent.adv_strategy=vanilla --agent.use_fpo=true --agent.fpo_per_sample=true --agent.fpo_use_aspo=false --agent.fpo_n_mc=16 --agent.fpo_cfm_loss_clamp=4.0 --agent.fpo_cfm_diff_clamp=5.0 --agent.fpo_use_huber=true --agent.fpo_huber_delta=1.0"
            ;;
        ogpo_awr_nn)
            # AW-OGPO-NN: AWR with positive_only (zero-grad on negative advantages)
            echo "--algo=ogpo --agent.adv_strategy=awr --agent.use_awr=true --agent.awr_mode=positive_only --agent.awr_beta=10.0 --agent.awr_weight_max=20.0"
            ;;
        *)
            echo "ERROR: unknown algo '$1'" >&2; exit 1 ;;
    esac
}

# ---------------------------------------------------------------------------
# Submission
# ---------------------------------------------------------------------------
TOTAL_JOBS=$(( ${#ALGOS[@]} * ${#ENVS[@]} * ${#SEEDS[@]} ))
echo "============================================================"
echo "  Image + PaliGemma + proprio ablation launcher"
echo "============================================================"
echo "  Algos      : ${ALGOS[*]}"
echo "  Envs       : ${ENVS[*]}"
echo "  Seeds      : ${SEEDS[*]}"
echo "  Total jobs : $TOTAL_JOBS"
echo "  Mode       : $([ "$SUBMIT" = true ] && echo SUBMIT || echo DRY-RUN)"
echo "  Partition  : $PARTITION  (constraint=$CONSTRAINT, time=$TIME)"
[ ${#EXTRA_ARGS[@]} -gt 0 ] && echo "  Extra args : ${EXTRA_ARGS[*]}"
echo "============================================================"

job_idx=0
for ALGO in "${ALGOS[@]}"; do
    ALGO_CLI=$(algo_cli_for "$ALGO")
    for ENV in "${ENVS[@]}"; do
        ENV_NAME=$(env_name_for "$ENV")
        ONLINE_STEPS=$(online_steps_for "$ENV")
        DISCOUNT=$(discount_for "$ENV")
        for SEED in "${SEEDS[@]}"; do
            job_idx=$((job_idx + 1))
            RUN_NAME="${ALGO}_${ENV}_pg_proprio_s${SEED}"
            WANDB_NAME="$RUN_NAME"
            PROJECT="OGPO_image_pg_proprio_${ENV}"
            RUN_GROUP="${ALGO}_${ENV}_pg_proprio"

            ENV_ARGS=(
                --env.name="$ENV_NAME"
                --training.online_steps=$ONLINE_STEPS
                --agent.discount=$DISCOUNT
            )

            EXP_ARGS=(
                --experiment.project="$PROJECT"
                --experiment.run_group="$RUN_GROUP"
                --experiment.run_name="$RUN_NAME"
                --experiment.wandb_name="$WANDB_NAME"
                --experiment.seed=$SEED
                --experiment.log=true
                --experiment.save_dir="$SAVE_DIR"
            )

            # Compose python command. Order: algo CLI -> experiment -> env -> common -> user extras.
            PY_CMD=(
                python vadar/main.py
                $ALGO_CLI
                "${EXP_ARGS[@]}"
                "${ENV_ARGS[@]}"
                "${COMMON_ARGS[@]}"
                "${EXTRA_ARGS[@]}"
            )

            echo ""
            echo "[$job_idx/$TOTAL_JOBS] $RUN_NAME"
            if [ "$SUBMIT" != true ]; then
                echo "    ${PY_CMD[*]}"
                continue
            fi

            JOB_NAME="${ALGO}_${ENV}_pg"
            LOG_FILE="$PROJECT_ROOT/logs/${JOB_NAME}_%j.out"

            # Activate venv + run python inside the sbatch script. MUJOCO_GL=egl
            # required for off-screen rendering on the eval envs.
            WRAP_CMD="set -e; cd '$PROJECT_ROOT'; \
source '$PROJECT_ROOT/.venv/bin/activate'; \
export PYTHONPATH='$PROJECT_ROOT:\$PYTHONPATH'; \
export MUJOCO_GL=egl; \
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90; \
${PY_CMD[*]}"

            sbatch \
                --partition="$PARTITION" \
                --job-name="$JOB_NAME" \
                --gres=gpu:2 \
                --constraint="$CONSTRAINT" \
                --cpus-per-task=20 \
                --mem=150G \
                --time="$TIME" \
                --output="$LOG_FILE" \
                --error="$LOG_FILE" \
                --wrap="$WRAP_CMD"
        done
    done
done

echo ""
if [ "$SUBMIT" = true ]; then
    echo "Submitted $job_idx job(s). Tail logs at: $PROJECT_ROOT/logs/"
else
    echo "Dry run complete. Re-run with --submit to actually sbatch."
fi
