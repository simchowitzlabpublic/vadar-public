#!/bin/bash
# Build the uv environments for VaDaR.
#
#   bash scripts/setup_env.sh            # both environments
#   bash scripts/setup_env.sh main       # robomimic + RoboCasa + OGBench + PushT
#   bash scripts/setup_env.sh libero     # LIBERO (needs robosuite 1.4, so it is separate)
#
# Environment variables:
#   CUDA=cuda12|cuda13      JAX CUDA wheels (default cuda12)
#   VADAR_VENV              main environment      (default <repo>/.venv)
#   VADAR_VENV_LIBERO       LIBERO environment    (default <repo>/.venv-libero)
#   ROBOCASA_ASSETS         existing robocasa models/assets directory to link instead of
#                           downloading the ~20 GB kitchen assets
#   SKIP_ROBOCASA_ASSETS=1  skip the RoboCasa asset step entirely
#   UV_CACHE_DIR            put the uv cache on scratch storage (it gets large)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET=${1:-all}
CUDA=${CUDA:-cuda12}
VADAR_VENV=${VADAR_VENV:-$ROOT/.venv}
VADAR_VENV_LIBERO=${VADAR_VENV_LIBERO:-$ROOT/.venv-libero}
# LIBERO commit pinned in uv.lock; its packaging is incomplete, so the checkout is added to the env via a .pth file.
LIBERO_REV=8f1084e3132a39270c3a13ebe37270a43ece2a01
LIBERO_CLONE=${LIBERO_CLONE:-$ROOT/third_party/LIBERO}

command -v uv >/dev/null || { echo "uv not found: https://docs.astral.sh/uv/getting-started/installation/" >&2; exit 1; }
cd "$ROOT"

setup_main() {
    echo "=== main environment -> $VADAR_VENV ($CUDA, robomimic, robocasa) ==="
    UV_PROJECT_ENVIRONMENT="$VADAR_VENV" uv sync --frozen --extra "$CUDA" --extra robomimic --extra robocasa --group test
    local py="$VADAR_VENV/bin/python"
    local assets
    assets=$("$py" -c "import os, robocasa; print(os.path.join(os.path.dirname(robocasa.__file__), 'models', 'assets'))" 2>/dev/null | tail -1)
    "$py" -m robocasa.scripts.setup_macros >/dev/null 2>&1 || true
    if [ "${SKIP_ROBOCASA_ASSETS:-0}" != "1" ]; then
        if [ -n "${ROBOCASA_ASSETS:-}" ]; then
            echo "linking RoboCasa assets: $assets -> $ROBOCASA_ASSETS"
            rm -rf "$assets" && ln -s "$ROBOCASA_ASSETS" "$assets"
        elif [ ! -d "$assets/arenas" ]; then
            echo "downloading RoboCasa kitchen assets (~20 GB, needs internet) ..."
            "$py" -m robocasa.scripts.download_kitchen_assets
        fi
    fi
    MUJOCO_GL=egl PYTHONPATH="$ROOT" "$py" - <<'PY'
import jax, flax, mujoco, robosuite, robocasa, robomimic, ogbench
print('jax', jax.__version__, '| flax', flax.__version__, '| mujoco', mujoco.__version__,
      '| robosuite', robosuite.__version__)
import vadar.main  # noqa: F401  (all agents and env factories import)
print('main environment OK')
PY
}

setup_libero() {
    echo "=== LIBERO environment -> $VADAR_VENV_LIBERO ($CUDA, libero) ==="
    UV_PROJECT_ENVIRONMENT="$VADAR_VENV_LIBERO" uv sync --frozen --extra "$CUDA" --extra libero --group test
    if [ ! -d "$LIBERO_CLONE/.git" ]; then
        git clone -q https://github.com/Lifelong-Robot-Learning/LIBERO.git "$LIBERO_CLONE"
    fi
    git -C "$LIBERO_CLONE" checkout -q "$LIBERO_REV"
    # The libero wheel built from git does not include the package itself; put the clone on
    # the environment's sys.path instead.
    local site
    site=$("$VADAR_VENV_LIBERO/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
    echo "$LIBERO_CLONE" > "$site/libero_clone.pth"
    # LIBERO asks for its paths interactively on first import; write its config instead.
    mkdir -p "$HOME/.libero"
    if [ ! -f "$HOME/.libero/config.yaml" ]; then
        local base="$LIBERO_CLONE/libero/libero"
        cat > "$HOME/.libero/config.yaml" <<YAML
benchmark_root: $base
bddl_files: $base/bddl_files
init_states: $base/init_files
datasets: $LIBERO_CLONE/datasets
assets: $base/assets
YAML
    fi
    MUJOCO_GL=egl PYTHONPATH="$ROOT" "$VADAR_VENV_LIBERO/bin/python" - <<'PY'
import jax, mujoco, robosuite, bddl, libero
from libero.libero.envs import TASK_MAPPING
print('jax', jax.__version__, '| mujoco', mujoco.__version__, '| robosuite', robosuite.__version__,
      '| LIBERO tasks', len(TASK_MAPPING))
import vadar.main  # noqa: F401
print('LIBERO environment OK')
PY
}

case "$TARGET" in
    main) setup_main ;;
    libero) setup_libero ;;
    all) setup_main; setup_libero ;;
    *) echo "usage: $0 [main|libero|all]" >&2; exit 1 ;;
esac
echo "Done. Datasets: bash scripts/fetch_datasets.sh"
