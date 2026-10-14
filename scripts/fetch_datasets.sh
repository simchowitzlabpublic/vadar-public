#!/bin/bash
# Download the offline datasets (GitHub release data-v1 of this repo; each file is
# too large for git) into <repo>/data, where the code looks by default:
#
#   data/robomimic/{tool_hang/ph,transport/mh}/low_dim_v15.hdf5
#   data/robocasa/{PreSoakPan,LoadDishWasher,ArrangeTea}/*.hdf5 (+ precomputed subgoal labels)
#   data/libero/KITCHEN_SCENE{3,4,6,8}_*_demo_processed.hdf5 (+ HER latent encoder)
#   data/pusht/sac_pusht_2000000_bc.npz           (PushT-Obs BC data)
#   data/ogbench/scene_task4_bc/                  (OGBench scene task4 BC data)
#
# Needs the GitHub CLI (`gh auth login`). OGBench's stock datasets and the PushT demos
# are downloaded by those libraries on first use.
#   VADAR_DATA_REPO=owner/repo VADAR_DATA_TAG=data-v1 bash scripts/fetch_datasets.sh
set -euo pipefail

REPO="${VADAR_DATA_REPO:-simchowitzlabpublic/vadar-public}"
TAG="${VADAR_DATA_TAG:-data-v1}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
D="$ROOT/data"

get() {  # <release-asset> <destination>
    local asset="$1" dest="$2"
    mkdir -p "$(dirname "$dest")"
    if [ -s "$dest" ]; then echo "  [skip] ${dest#$ROOT/}"; return; fi
    echo "  [get ] $asset -> ${dest#$ROOT/}"
    gh release download "$TAG" --repo "$REPO" --pattern "$asset" --output "$dest" --clobber
}

echo "=== robomimic ==="
get tool_hang_ph_low_dim_v15.hdf5 "$D/robomimic/tool_hang/ph/low_dim_v15.hdf5"
get transport_mh_low_dim_v15.hdf5 "$D/robomimic/transport/mh/low_dim_v15.hdf5"

echo "=== RoboCasa ==="
get PreSoakPan_200_success_p9_28_full_obs.hdf5 "$D/robocasa/PreSoakPan/PreSoakPan_200_success_p9_28_full_obs.hdf5"
get LoadDishwasher_noisy_replay_ready_31dim_v2.hdf5 "$D/robocasa/LoadDishWasher/LoadDishwasher_noisy_replay_ready_31dim_v2.hdf5"
get ArrangeTea_VLA_single150.hdf5 "$D/robocasa/ArrangeTea/ArrangeTea_VLA_single150.hdf5"
# Precomputed subgoal labels (anchor demo_0); computing them replays every demo (~1 h per task).
for f in PreSoakPan/PreSoakPan_200_success_p9_28_full_obs.hdf5.subgoal_dones_post_G5_anchor-demo_0_resync.npy \
         LoadDishWasher/LoadDishwasher_noisy_replay_ready_31dim_v2.hdf5.subgoal_dones_post_G5_dag4b28bc28_anchor-demo_0_resync.npy \
         ArrangeTea/ArrangeTea_VLA_single150.hdf5.subgoal_dones_post_G6_anchor-demo_0_resync.npy; do
    get "$(basename "$f")" "$D/robocasa/$f"
done

echo "=== LIBERO ==="
for f in KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo_processed.hdf5 \
         KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_demo_processed.hdf5 \
         KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it_demo_processed.hdf5 \
         KITCHEN_SCENE8_put_both_moka_pots_on_the_stove_demo_processed.hdf5 \
         subgoal_encoder_k6.npz; do
    get "$f" "$D/libero/$f"
done

echo "=== PushT-Obs ==="
get sac_pusht_2000000_bc.npz "$D/pusht/sac_pusht_2000000_bc.npz"

echo "=== OGBench scene task4 ==="
if [ ! -s "$D/ogbench/scene_task4_bc/scene-play-v0.npz" ]; then
    get ogbench_scene_task4_bc.tar.gz "$D/ogbench/ogbench_scene_task4_bc.tar.gz"
    tar -xzf "$D/ogbench/ogbench_scene_task4_bc.tar.gz" -C "$D/ogbench" && rm "$D/ogbench/ogbench_scene_task4_bc.tar.gz"
fi

echo "=== done ==="
du -sh "$D"/* 2>/dev/null || true
