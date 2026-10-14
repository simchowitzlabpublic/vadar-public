#!/bin/bash
# =============================================================================
# OGPO Ablation Study — Per-Node Launcher
# =============================================================================
#
# Designed to be called once per node by launch_training.py.
# Each invocation handles exactly 2 ablation configs x 3 timestamp seeds
# = 6 runs, one per GPU, all 6 in parallel on GPUs 0-5.
#
# Ablation matrix (48 configs total):
#   Envs:       square, tool_hang                          (2)
#   PPO clip:   0.01, 0.001                                (2)
#   Obs config: State/State, State/Image, Image/Image      (3)
#   Backbone:   MLP|MLP, MLP|TF, TF|MLP, TF|TF           (4)
#   Total:      2 x 2 x 3 x 4 = 48 configs
#   Nodes needed: 48 / 2 = 24 invocations
#
# Usage:
#   bash scripts/ogpo/launch_ablations.sh --node_id=0       # configs 0,1
#   bash scripts/ogpo/launch_ablations.sh --node_id=5       # configs 10,11
#   bash scripts/ogpo/launch_ablations.sh --node_id=23      # configs 46,47
#   bash scripts/ogpo/launch_ablations.sh --dry_run --node_id=0
#
# The 48 configs are ordered as:
#   Idx  0-11:  square     clip=0.01  (state/state, state/image, image/image) x 4 backbones
#   Idx 12-23:  square     clip=0.001 (same)
#   Idx 24-35:  toolhang   clip=0.01  (same)
#   Idx 36-47:  toolhang   clip=0.001 (same)
# =============================================================================

set -e

# ─── Defaults ────────────────────────────────────────────────────────────────
NODE_ID=""
CONFIGS_PER_NODE=2
NUM_GPUS=6
DRY_RUN=0

# Auto-detect project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SAVE_DIR="${OGPO_SAVE_DIR:-/data/user_data/$USER/ogpo}"

# ─── Parse CLI overrides ────────────────────────────────────────────────────
for arg in "$@"; do
  case $arg in
    --node_id=*)    NODE_ID="${arg#*=}" ;;
    --save_dir=*)   SAVE_DIR="${arg#*=}" ;;
    --dry_run)      DRY_RUN=1 ;;
    *)              echo "Unknown argument: $arg"; exit 1 ;;
  esac
done

if [[ -z "$NODE_ID" ]]; then
  echo "ERROR: --node_id=N is required (0-23 for 48 configs, 2 per node)"
  exit 1
fi

# ─── Generate 3 timestamp-based seeds ────────────────────────────────────────
TS=$(date +%s)
SEEDS=($TS $((TS + 10000)) $((TS + 20000)))

LOG_DIR="$PROJECT_ROOT/logs/ablations/node${NODE_ID}_$(date +%Y%m%d_%H%M%S)"

export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
cd "$PROJECT_ROOT"

# ─── Run function ────────────────────────────────────────────────────────────
run_one() {
    local gpu=$1 seed=$2 env_name=$3 horizon=$4 discount=$5 online=$6 noise=$7 \
          clip=$8 cobs=$9 aobs=${10} cbb=${11} abb=${12} tag=${13}

    local plot_q=true
    [[ "$aobs" == "image" || "$cobs" == "image" ]] && plot_q=false

    CUDA_VISIBLE_DEVICES=$gpu python vadar/main.py \
        --algo=ogpo \
        --experiment.project=OGPO \
        --experiment.run_group="$tag" \
        --experiment.run_name="${tag}_s${seed}" \
        --experiment.wandb_name="${tag}_s${seed}" \
        --experiment.seed=$seed \
        --experiment.log=true \
        --experiment.save_dir="$SAVE_DIR" \
        --env.name="$env_name" \
        --env.sparse=false \
        --env.post_success_steps=0 \
        --training.bc_pi_steps=500000 \
        --training.online_steps=$online \
        --training.bc_only=false \
        --buffer.start_training=10000 \
        --buffer.utd_warmup=1 \
        --buffer.utd_online=1 \
        --buffer.offline_ratio=0.0 \
        --eval.n_eval_envs=32 \
        --eval.eval_episodes=2 \
        --eval.plot_q_vs_mc=$plot_q \
        --intervals.log_interval=5000 \
        --intervals.eval_interval=20000 \
        --intervals.eval_interval_bc=10000 \
        --intervals.save_interval=500000 \
        --bc.clip_bc=true \
        --bc.clip_bc_threshold=0.45 \
        --success_buffer.use_success_buffer=true \
    --agent.use_success_buffer_q=false \
        --agent.horizon_length=$horizon \
        --agent.discount=$discount \
        --agent.tau=0.05 \
        --agent.actor_tau=0.005 \
        --agent.best_of_n=8 \
        --agent.num_qs=10 \
        --agent.q_agg=subsample \
        --agent.subsample_bon=true \
        --agent.flow_steps=10 \
        --agent.ft_flow_steps=10 \
        --agent.bc_coeff=1.0 \
        --agent.adv_strategy=vanilla \
        --agent.awr_beta=10 \
        --agent.ppo_batch_size=256 \
        --agent.clip_epsilon=$clip \
        --agent.entropy_coeff=0.0 \
        --agent.group_num_samples=32 \
        --agent.use_bc_regularization=true \
        --agent.error_correct_ode_to_sde=false \
        --agent.use_denoiser=false \
        --agent.use_constant_noise=false \
        --agent.use_tapered_noise=true \
        --agent.error_correct_sde_to_ode=true \
        --agent.constant_noise_std=$noise \
        --agent.min_noise_std=$noise \
        --agent.max_noise_std=$noise \
        --agent.actor_backbone=$abb \
        --agent.critic_backbone=$cbb \
        --agent.actor_obs=$aobs \
        --agent.critic_obs=$cobs \
        --agent.vit_encoder=minvit
}

# ─── Build full job list (48 configs) ────────────────────────────────────────
JOBS=()

OBS_CFGS=("state state" "state image" "image image")
BB_CFGS=("mlp mlp" "mlp tf" "tf mlp" "tf tf")

for env in square toolhang; do
  for clip in 0.01 0.001; do
    for obs_cfg in "${OBS_CFGS[@]}"; do
      read -r cobs aobs <<< "$obs_cfg"
      for bb_cfg in "${BB_CFGS[@]}"; do
        read -r cbb abb <<< "$bb_cfg"

        case $env in
          square)
            horizon=4; discount=0.99; online=2000000; noise=0.01
            if [[ "$aobs" == "image" || "$cobs" == "image" ]]; then
              env_name="square-mh-image"
            else
              env_name="square-mh-low_dim"
            fi
            ;;
          toolhang)
            horizon=8; discount=0.999; online=3000000; noise=0.005
            if [[ "$aobs" == "image" || "$cobs" == "image" ]]; then
              env_name="tool_hang-ph-image"
            else
              env_name="tool_hang-ph-low_dim"
            fi
            ;;
        esac

        tag="${env}_clip${clip}_${cobs}Q${cbb}_${aobs}Pi${abb}"
        JOBS+=("${env_name}|${horizon}|${discount}|${online}|${noise}|${clip}|${cobs}|${aobs}|${cbb}|${abb}|${tag}")
      done
    done
  done
done

# ─── Select this node's 2 configs ───────────────────────────────────────────
CONFIG_START=$((NODE_ID * CONFIGS_PER_NODE))
CONFIG_END=$((CONFIG_START + CONFIGS_PER_NODE - 1))
NUM_TOTAL=${#JOBS[@]}

if (( CONFIG_START >= NUM_TOTAL )); then
  echo "ERROR: node_id=$NODE_ID gives config_start=$CONFIG_START but only $NUM_TOTAL configs exist."
  echo "Valid node_id range: 0-$((NUM_TOTAL / CONFIGS_PER_NODE - 1))"
  exit 1
fi

# Clamp end if last node has fewer than CONFIGS_PER_NODE
if (( CONFIG_END >= NUM_TOTAL )); then
  CONFIG_END=$((NUM_TOTAL - 1))
fi

NODE_CONFIGS=()
for i in $(seq $CONFIG_START $CONFIG_END); do
  NODE_CONFIGS+=("${JOBS[$i]}")
done

NUM_NODE_CONFIGS=${#NODE_CONFIGS[@]}
NUM_RUNS=$((NUM_NODE_CONFIGS * ${#SEEDS[@]}))

echo "============================================================"
echo "OGPO Ablation — Node $NODE_ID"
echo "============================================================"
echo "Project root:  $PROJECT_ROOT"
echo "Save dir:      $SAVE_DIR"
echo "Log dir:       $LOG_DIR"
echo "Seeds:         ${SEEDS[*]}"
echo "Configs:       $CONFIG_START..$CONFIG_END (${NUM_NODE_CONFIGS} configs)"
echo "Total runs:    $NUM_RUNS  (${NUM_NODE_CONFIGS} configs x ${#SEEDS[@]} seeds)"
echo "GPUs used:     0..$((NUM_RUNS - 1))"
echo ""

# ─── Print this node's assignments ──────────────────────────────────────────
echo "GPU assignments:"
gpu=0
for cfg_idx in "${!NODE_CONFIGS[@]}"; do
  IFS='|' read -r env_name horizon discount online noise clip cobs aobs cbb abb tag <<< "${NODE_CONFIGS[$cfg_idx]}"
  global_idx=$((CONFIG_START + cfg_idx))
  for seed in "${SEEDS[@]}"; do
    printf "  GPU %d  ->  [%2d] %s  seed=%d\n" "$gpu" "$global_idx" "$tag" "$seed"
    gpu=$((gpu + 1))
  done
done
echo ""

if [[ $DRY_RUN -eq 1 ]]; then
  echo "[DRY RUN] Would launch $NUM_RUNS runs on GPUs 0-$((NUM_RUNS - 1)). Exiting."
  exit 0
fi

mkdir -p "$LOG_DIR"

# ─── Launch: 2 configs x 3 seeds = 6 runs, one per GPU, all in parallel ─────
gpu=0
for cfg_idx in "${!NODE_CONFIGS[@]}"; do
  IFS='|' read -r env_name horizon discount online noise clip cobs aobs cbb abb tag <<< "${NODE_CONFIGS[$cfg_idx]}"

  for seed in "${SEEDS[@]}"; do
    echo "[GPU $gpu] Starting: ${tag}_s${seed}"
    run_one "$gpu" "$seed" "$env_name" "$horizon" "$discount" "$online" "$noise" \
            "$clip" "$cobs" "$aobs" "$cbb" "$abb" "$tag" \
            > "$LOG_DIR/${tag}_s${seed}.log" 2>&1 &
    gpu=$((gpu + 1))
  done
done

echo "All $NUM_RUNS runs launched on GPUs 0-$((gpu - 1))."
echo "Logs: $LOG_DIR/"
echo "Monitor: tail -f $LOG_DIR/*.log"
echo ""

wait
echo "============================================================"
echo "Node $NODE_ID complete! All $NUM_RUNS runs finished."
echo "============================================================"
