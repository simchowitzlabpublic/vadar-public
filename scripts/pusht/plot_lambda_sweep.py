"""Plot the PushT-Obs lambda_safety sweep: VaDaR vs aggSAC success rate per lambda.

Reads the per-seed training logs, keeps the online phase (everything after
"BC Training Complete"), parses `[EVAL ODE] step=N | success=V` lines, and draws
the mean over seeds with a min/max band, one panel per lambda (paper Fig.
"pusht_panel"). Port of the QIPM fork's scripts/plot_pushT_lambda_sweep.py.

Log file names: <prefix>_lam<TAG>_seed<S>_<anything>.out, where TAG is the
lambda with '.' -> 'p' (0p1, 5, 10, ...) and <prefix> identifies the method:
    sac_vadar_pusht / sac_qipm_avoid_pusht   -> VaDaR
    sac_aggsac_pusht / sac_dense_parity_pusht -> aggSAC
(the second names are the fork's, so the paper's original logs plot as is).

The paper's lambda = 1 curves came from separate wandb runs; pass
--lam1_wandb to pull them (ids below are the paper's runs), otherwise lambda = 1
is read from logs like every other lambda. The paper excluded lambda in {0.5, 50}.

Usage:
    python scripts/pusht/plot_lambda_sweep.py --logs_dir logs --out_dir figs/pusht
"""
import argparse
import glob
import os
import pickle
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

METHOD_OF_PREFIX = {
    "sac_vadar_pusht": "vadar",
    "sac_qipm_avoid_pusht": "vadar",
    "sac_aggsac_pusht": "aggsac",
    "sac_dense_parity_pusht": "aggsac",
}
LOG_RE = re.compile(r"^(?P<prefix>.+?)_lam(?P<lam>[0-9p]+)_seed(?P<seed>\d+)(?:_.*)?\.out$")
EVAL_RE = re.compile(r"\[EVAL ODE\] step=\s*(\d+)\s*\|\s*success=([0-9.]+)")
BC_DONE_RE = re.compile(r"BC Training Complete")

# The paper's lambda = 1 runs (wandb entity/project of the QIPM fork).
WANDB_ENTITY = "pluralistic-goal-conditioning"
WANDB_PROJECT = "pushT-paper"
LAM1_WANDB = {
    "aggsac": ["zrzkfbm2", "lyrzlj6x", "fe7sv3ak"],
    "vadar": ["e5snzbf3", "0lmk7nnb", "dwblp6zr"],
}
WANDB_METRIC = "eval/success"

COLOR = {"vadar": "#B30000", "aggsac": "#2A52BE"}
LABEL = {"vadar": "VaDaR", "aggsac": "aggSAC"}
LINEWIDTH = {"vadar": 5, "aggsac": 3}
METHODS = ("vadar", "aggsac")


def lam_from_tag(tag):
    return float(tag.replace("p", "."))


def parse_log(path):
    """(steps, success) arrays for the online phase of one log."""
    steps, success, online = [], [], False
    with open(path, "r", errors="ignore") as fh:
        for line in fh:
            if not online:
                online = bool(BC_DONE_RE.search(line))
                continue
            m = EVAL_RE.search(line)
            if m:
                steps.append(int(m.group(1)))
                success.append(float(m.group(2)))
    return np.asarray(steps, dtype=np.int64), np.asarray(success, dtype=np.float32)


def fetch_wandb_runs(run_ids, x_max, cache_path=None, refetch=False):
    """{method: [(steps, success), ...]} from wandb; the last x_max steps of each run, rebased to 0."""
    if cache_path and not refetch and os.path.isfile(cache_path):
        with open(cache_path, "rb") as fh:
            return pickle.load(fh)
    import wandb
    api = wandb.Api()
    out = {}
    for method, ids in run_ids.items():
        out[method] = []
        for rid in ids:
            h = api.run(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{rid}").history(keys=["_step", WANDB_METRIC]).dropna()
            steps = h["_step"].to_numpy().astype(np.int64)
            succ = h[WANDB_METRIC].to_numpy().astype(np.float32)
            offset = max(steps.max() - x_max, 0)   # drop the BC phase (and anything beyond x_max online steps)
            keep = steps >= offset
            out[method].append((steps[keep] - offset, succ[keep]))
    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        with open(cache_path, "wb") as fh:
            pickle.dump(out, fh)
    return out


def ewma(x, w):
    if w == 0.0 or len(x) == 0:
        return x
    y = np.copy(x)
    for i in range(1, len(x)):
        if not np.isnan(x[i]) and not np.isnan(y[i - 1]):
            y[i] = w * y[i - 1] + (1 - w) * x[i]
    return y


def on_grid(runs, num_points=200):
    runs = [(s, v) for s, v in runs if len(s) > 0]
    if not runs:
        return None, None
    grid = np.linspace(min(s.min() for s, _ in runs), max(s.max() for s, _ in runs), num_points)
    rows = []
    for s, v in runs:
        row = np.full(num_points, np.nan)
        inside = (grid >= s.min()) & (grid <= s.max())
        row[inside] = np.interp(grid[inside], s, v)
        rows.append(row)
    return grid, np.asarray(rows)


def draw(ax, runs, method, smoothing):
    grid, rows = on_grid(runs)
    if grid is None:
        return
    rows = np.array([ewma(r, smoothing) for r in rows])
    mean, lo, hi = np.nanmean(rows, 0), np.nanmin(rows, 0), np.nanmax(rows, 0)
    x = grid - grid.min()
    ax.plot(x, mean, color=COLOR[method], linewidth=LINEWIDTH[method])
    ax.fill_between(x, np.clip(lo, 0, 1), np.clip(hi, 0, 1), color=COLOR[method], alpha=0.1)


def style():
    plt.rcParams.update({
        "font.family": "monospace",
        "font.monospace": ["Courier New", "Courier", "DejaVu Sans Mono", "Nimbus Mono PS", "monospace"],
        "font.weight": "bold", "font.size": 18, "axes.labelweight": "bold", "axes.titleweight": "bold",
        "axes.labelsize": 20, "axes.titlesize": 22, "legend.fontsize": 16,
        "xtick.labelsize": 16, "ytick.labelsize": 16,
    })


def format_axis(ax, x_max, ylabel=True):
    ax.set_xlabel("Steps")
    if ylabel:
        ax.set_ylabel("Success Rate")
    ax.set_ylim(0, 1)
    ax.set_xlim(0, x_max)
    ax.ticklabel_format(style="scientific", axis="x", scilimits=(0, 0))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--logs_dir", default="logs")
    p.add_argument("--out_dir", default="figs/pusht")
    p.add_argument("--smoothing", type=float, default=0.5)
    p.add_argument("--x_max", type=int, default=800_000, help="online steps shown (the paper: 800k)")
    p.add_argument("--exclude", type=str, default="0.5,50", help="comma-separated lambdas to drop")
    p.add_argument("--lam1_wandb", action="store_true", help="use the paper's wandb runs for lambda = 1")
    p.add_argument("--refetch", action="store_true", help="ignore the wandb cache")
    args = p.parse_args()

    runs = {}
    for f in sorted(glob.glob(os.path.join(args.logs_dir, "*_lam*_seed*.out"))):
        m = LOG_RE.match(os.path.basename(f))
        if not m or m.group("prefix") not in METHOD_OF_PREFIX:
            continue
        key = (METHOD_OF_PREFIX[m.group("prefix")], lam_from_tag(m.group("lam")))
        runs.setdefault(key, []).append(parse_log(f))
    if args.lam1_wandb:
        cache = os.path.join(args.out_dir, ".lam1_wandb_cache.pkl")
        for method, method_runs in fetch_wandb_runs(LAM1_WANDB, args.x_max, cache, args.refetch).items():
            runs[(method, 1.0)] = list(method_runs)

    exclude = {float(x) for x in args.exclude.split(",") if x.strip()}
    lambdas = sorted({lam for _, lam in runs} - exclude)
    if not lambdas:
        raise SystemExit(f"No sweep logs found in {args.logs_dir}")
    for (method, lam), r in sorted(runs.items()):
        print(f"  {LABEL[method]:7s} lambda={lam:<5g} seeds={len(r)} "
              f"last_step={max((s[-1] if len(s) else -1) for s, _ in r)}")

    style()
    os.makedirs(args.out_dir, exist_ok=True)
    w, h = 9, 6

    fig = plt.figure(figsize=(w * len(lambdas), h))
    for i, lam in enumerate(lambdas):
        ax = fig.add_subplot(1, len(lambdas), i + 1)
        for method in METHODS:
            if (method, lam) in runs:
                draw(ax, runs[(method, lam)], method, args.smoothing)
        ax.set_title(f"λ = {lam:g}")
        format_axis(ax, args.x_max, ylabel=(i == 0))
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(args.out_dir, f"lambda_sweep.{ext}"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    for lam in lambdas:
        fig, ax = plt.subplots(figsize=(w, h))
        for method in METHODS:
            if (method, lam) in runs:
                draw(ax, runs[(method, lam)], method, args.smoothing)
        format_axis(ax, args.x_max)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(args.out_dir, f"lam{str(lam).replace('.', 'p')}.{ext}"),
                        dpi=150, bbox_inches="tight")
        plt.close(fig)

    fig = plt.figure(figsize=(8, 0.6))
    fig.legend([plt.Line2D([0], [0], color=COLOR[m], linewidth=LINEWIDTH[m]) for m in METHODS],
               [LABEL[m] for m in METHODS], loc="center", ncol=len(METHODS), frameon=False, fontsize=18)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(args.out_dir, f"legend.{ext}"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] figures in {args.out_dir}")


if __name__ == "__main__":
    main()
