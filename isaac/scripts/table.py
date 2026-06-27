"""
make_results_table.py
─────────────────────────────────────────────────────────────────────────────
Loads all trajectory .npz files from a results folder, computes per-episode
metrics, aggregates per (variant, encoder, tighten) group, and prints:
  1. A clean terminal table
  2. A LaTeX table ready to paste into your thesis

Metrics computed per episode
─────────────────────────────
  timesteps          : number of steps taken
  goal               : 1 if success==True, else 0
  constraint_goal    : 1 if success AND violations==0, else 0
  violations         : number of timesteps inside any obstacle footprint

Obstacle collision check (XY footprint)
  box      : axis-aligned square  |x-cx| < 0.10  AND  |y-cy| < 0.10
  cylinder : circle               (x-cx)^2 + (y-cy)^2 < 0.06^2

Usage
─────
    python make_results_table.py --results_dir path/to/npz/folder
    python make_results_table.py --results_dir . --out results_table.tex

Flags
─────
    --results_dir   folder containing *.npz files  (default: current dir)
    --out           write LaTeX to this file        (default: print only)
    --box_size      box half-width for collision    (default: 0.10  -> 0.20m box)
    --cyl_radius    cylinder radius for collision   (default: 0.06)
"""

import argparse
import os
import glob
import numpy as np
from collections import defaultdict


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

GATE_X      = 4.0   # x position of the goal gate
DRONE_RADIUS = 0.08  # matches crazyflie_env.py drone_radius default

def count_violations(xyz, boxes, cylinders, box_half, cyl_radius, drone_radius=DRONE_RADIUS):
    """Count timesteps where the drone body overlaps any obstacle footprint.
    Uses cyl_radius + drone_radius to match the collision check in crazyflie_env.py."""
    cyl_thresh_sq = (cyl_radius + drone_radius) ** 2
    box_thresh    = box_half + drone_radius
    violations = 0
    for t in range(len(xyz)):
        x, y = float(xyz[t, 0]), float(xyz[t, 1])
        hit = False
        for bx, by in boxes:
            if abs(x - bx) < box_thresh and abs(y - by) < box_thresh:
                hit = True
                break
        if not hit:
            for cx, cy in cylinders:
                if (x - cx) ** 2 + (y - cy) ** 2 < cyl_thresh_sq:
                    hit = True
                    break
        if hit:
            violations += 1
    return violations


def load_episode(path, box_half, cyl_radius, drone_radius=DRONE_RADIUS):
    """Return a dict of scalar metrics for one .npz file."""
    d = np.load(path, allow_pickle=True)

    xyz       = d["xyz"]                          # (T, 3)
    actions   = d["actions"]                      # (T-1, 3)
    success   = bool(d["success"])
    fell      = bool(d["fell"])
    variant   = str(d["variant"])
    encoder   = str(d["encoder"])
    tighten   = float(d["tighten"])
    latent    = int(d["latent_dim"]) if "latent_dim" in d else -1

    boxes     = d["boxes"]      if "boxes"     in d else np.zeros((0, 2))
    cylinders = d["cylinders"]  if "cylinders" in d else np.zeros((0, 2))

    violations    = count_violations(xyz, boxes, cylinders, box_half, cyl_radius, drone_radius)
    path_length   = float(np.linalg.norm(actions, axis=1).sum())
    final_x       = float(xyz[-1, 0])
    final_y       = float(xyz[-1, 1])
    final_z       = float(xyz[-1, 2])
    max_x         = float(xyz[:, 0].max())
    # progress rate: how far along the corridor the drone got (0.0 – 1.0)
    # e.g. reaches x=2 before failing → 2/4 = 0.50
    progress_rate = float(min(max_x, GATE_X) / GATE_X)

    return dict(
        variant       = variant,
        encoder       = encoder,
        tighten       = tighten,
        latent        = latent,
        timesteps     = len(xyz),
        success       = 1 if success else 0,
        fell          = 1 if fell else 0,
        violations    = violations,
        progress_rate = progress_rate,
        path_length   = path_length,
        final_x       = final_x,
        final_y       = final_y,
        final_z       = final_z,
        max_x         = max_x,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation
# ─────────────────────────────────────────────────────────────────────────────

def aggregate(episodes):
    """
    episodes: list of metric dicts (same variant/encoder/tighten group)
    Returns mean +- std strings for each metric.
    """
    n   = len(episodes)
    ts  = np.array([e["timesteps"]   for e in episodes], dtype=float)
    sr  = np.array([e["success"]     for e in episodes], dtype=float)
    v   = np.array([e["violations"]  for e in episodes], dtype=float)
    pl  = np.array([e["path_length"] for e in episodes], dtype=float)
    fx  = np.array([e["final_x"]     for e in episodes], dtype=float)
    fy  = np.array([e["final_y"]     for e in episodes], dtype=float)
    fz  = np.array([e["final_z"]     for e in episodes], dtype=float)
    mx  = np.array([e["max_x"]       for e in episodes], dtype=float)

    pr  = np.array([e["progress_rate"] for e in episodes], dtype=float)

    def ms(arr, fmt=".2f"):
        if n == 1:
            return f"{arr.mean():{fmt}}"
        return f"{arr.mean():{fmt}} \\pm {arr.std():{fmt}}"

    def rate_str(arr):
        s = f"{arr.mean():.2f}"
        if n > 1:
            s += f" \\pm {arr.std():.2f}"
        return s

    return dict(
        n             = n,
        timesteps     = ms(ts,  ".0f"),
        success       = rate_str(sr),
        progress_rate = rate_str(pr),
        violations    = ms(v,   ".1f"),
        path_length   = ms(pl,  ".2f"),
        final_x       = ms(fx,  ".3f"),
        final_y       = ms(fy,  ".3f"),
        final_z       = ms(fz,  ".3f"),
        max_x         = ms(mx,  ".3f"),
        # raw means for bolding
        _sr_mean      = sr.mean(),
        _pr_mean      = pr.mean(),
        _v_mean       = v.mean(),
        _ts_mean      = ts.mean(),
        _pl_mean      = pl.mean(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Display order (matches SDPC paper Table 1 style)
# ─────────────────────────────────────────────────────────────────────────────

VARIANT_ORDER = [
    "diffuser",
    "gradient", "gradient-tightened",
    "post_processing", "post_processing-tightened",
    "model_free", "model_free-tightened",
    "sdpc-r", "sdpc-r-tightened",
    "sdpc-c", "sdpc-c-tightened",
    "sdpc-t", "sdpc-t-tightened",
    # dt sweep
    "sdpc-c-tightened-dt0p25",
    "sdpc-c-tightened-dt0p5",
    "sdpc-c-tightened-dt2p0",
    "sdpc-c-tightened-dt4p0",
]

VARIANT_LABEL = {
    "diffuser"                  : "Diffuser",
    "gradient"                  : "Guidance",
    "gradient-tightened"        : "Guidance (tightened)",
    "post_processing"           : "Post-Processing",
    "post_processing-tightened" : "Post-Processing (tightened)",
    "model_free"                : "Model-Free",
    "model_free-tightened"      : "Model-Free (tightened)",
    "sdpc-r"                    : "SDPC-R",
    "sdpc-r-tightened"          : "SDPC-R (tightened)",
    "sdpc-c"                    : "SDPC-C",
    "sdpc-c-tightened"          : "SDPC-C (tightened)",
    "sdpc-t"                    : "SDPC-T",
    "sdpc-t-tightened"          : "SDPC-T (tightened)",
    "sdpc-c-tightened-dt0p25"   : r"SDPC-C (dt=0.25)",
    "sdpc-c-tightened-dt0p5"    : r"SDPC-C (dt=0.5)",
    "sdpc-c-tightened-dt2p0"    : r"SDPC-C (dt=2.0)",
    "sdpc-c-tightened-dt4p0"    : r"SDPC-C (dt=4.0)",
}


# ─────────────────────────────────────────────────────────────────────────────
# Formatting helpers
# ─────────────────────────────────────────────────────────────────────────────

def bold(s):
    return f"\\textbf{{{s}}}"


def make_terminal_table(rows, encoders):
    """rows: list of (variant, encoder, tighten, agg_dict)"""
    col_w = [30, 8, 10, 7, 8, 10, 8, 8, 8, 8]
    header = (
        f"{'Variant':<{col_w[0]}} {'Encoder':<{col_w[1]}}"
        f" {'Steps':>{col_w[2]}} {'SR':>{col_w[3]}} {'Progress':>{col_w[4]}}"
        f" {'Violations':>{col_w[5]}} {'PathLen':>{col_w[6]}}"
        f" {'Final-x':>{col_w[7]}} {'Final-y':>{col_w[8]}} {'Final-z':>{col_w[9]}}"
    )
    sep = "-" * len(header)
    lines = [sep, header, sep]
    for variant, encoder, tighten, agg in rows:
        label = VARIANT_LABEL.get(variant, variant)
        lines.append(
            f"{label:<{col_w[0]}} {encoder:<{col_w[1]}}"
            f" {agg['timesteps']:>{col_w[2]}} {agg['success']:>{col_w[3]}}"
            f" {agg['progress_rate']:>{col_w[4]}}"
            f" {agg['violations']:>{col_w[5]}} {agg['path_length']:>{col_w[6]}}"
            f" {agg['final_x']:>{col_w[7]}} {agg['final_y']:>{col_w[8]}} {agg['final_z']:>{col_w[9]}}"
            f"  (n={agg['n']})"
        )
    lines.append(sep)
    return "\n".join(lines)


def make_latex_table(rows, caption="", label="tab:results"):
    """
    Booktabs LaTeX table.
    Columns: Variant | Encoder | Timesteps | SR | Progress | Violations | Path Length | Final x | Final y | Final z
    Best value per column is bolded automatically.
    """
    best_ts = min(r[3]["_ts_mean"] for r in rows)
    best_sr = max(r[3]["_sr_mean"] for r in rows)
    best_pr = max(r[3]["_pr_mean"] for r in rows)
    best_v  = min(r[3]["_v_mean"]  for r in rows)
    best_pl = min(r[3]["_pl_mean"] for r in rows)

    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"  \centering")
    lines.append(r"  \small")
    if caption:
        lines.append(f"  \\caption{{{caption}}}")
    lines.append(f"  \\label{{{label}}}")
    lines.append(r"  \begin{tabular}{llrrrrrrrr}")
    lines.append(r"  \toprule")
    lines.append(
        r"  \textbf{Variant} & \textbf{Enc.} & \textbf{Steps}"
        r" & \textbf{SR} & \textbf{Progress}"
        r" & \textbf{Violations}"
        r" & \textbf{Path (m)} & \textbf{$x_f$} & \textbf{$y_f$} & \textbf{$z_f$} \\"
    )
    lines.append(r"  \midrule")

    prev_variant = None
    for variant, encoder, tighten, agg in rows:
        label_str = VARIANT_LABEL.get(variant, variant)
        if prev_variant is not None and variant != prev_variant:
            lines.append(r"  \addlinespace[2pt]")
        prev_variant = variant

        ts_str = agg["timesteps"]
        sr_str = agg["success"]
        pr_str = agg["progress_rate"]
        v_str  = agg["violations"]
        pl_str = agg["path_length"]
        fx_str = agg["final_x"]
        fy_str = agg["final_y"]
        fz_str = agg["final_z"]

        if abs(agg["_ts_mean"] - best_ts) < 1e-3: ts_str = bold(ts_str)
        if abs(agg["_sr_mean"] - best_sr) < 1e-6: sr_str = bold(sr_str)
        if abs(agg["_pr_mean"] - best_pr) < 1e-6: pr_str = bold(pr_str)
        if abs(agg["_v_mean"]  - best_v)  < 1e-3: v_str  = bold(v_str)
        if abs(agg["_pl_mean"] - best_pl) < 1e-3: pl_str = bold(pl_str)

        lines.append(
            f"  {label_str} & {encoder}"
            f" & ${ts_str}$ & ${sr_str}$ & ${pr_str}$"
            f" & ${v_str}$ & ${pl_str}$"
            f" & ${fx_str}$ & ${fy_str}$ & ${fz_str}$ \\\\"
        )

    lines.append(r"  \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default=".",
                        help="Folder containing *.npz result files")
    parser.add_argument("--out",         default=None,
                        help="Write LaTeX table to this .tex file")
    parser.add_argument("--box_size",    type=float, default=0.20,
                        help="Box side length in metres (default 0.20)")
    parser.add_argument("--cyl_radius",  type=float, default=0.06,
                        help="Cylinder radius in metres (default 0.06)")
    parser.add_argument("--drone_radius", type=float, default=DRONE_RADIUS,
                        help=f"Drone body radius for Minkowski collision (default {DRONE_RADIUS})")
    parser.add_argument("--caption",     default=(
                        "Comparison of safety variants across encoders. "
                        "Mean $\\pm$ std over episodes."))
    parser.add_argument("--label",       default="tab:results")
    args = parser.parse_args()

    box_half     = args.box_size / 2.0
    cyl_radius   = args.cyl_radius
    drone_radius = args.drone_radius

    # ── Load all npz files ───────────────────────────────────────────────────
    paths = sorted(glob.glob(os.path.join(args.results_dir, "*.npz")))
    if not paths:
        print(f"[ERROR] No .npz files found in: {args.results_dir}")
        return

    print(f"Found {len(paths)} npz files in {args.results_dir}\n")

    # group: (variant, encoder, tighten) -> list of episode dicts
    groups = defaultdict(list)
    encoders_seen = set()

    for p in paths:
        try:
            ep = load_episode(p, box_half, cyl_radius, drone_radius)
            key = (ep["variant"], ep["encoder"], ep["tighten"])
            groups[key].append(ep)
            encoders_seen.add(ep["encoder"])
        except Exception as e:
            print(f"  [WARN] skipping {os.path.basename(p)}: {e}")

    # ── Aggregate ────────────────────────────────────────────────────────────
    agg_rows = {}   # key -> agg dict
    for key, eps in groups.items():
        agg_rows[key] = aggregate(eps)

    # ── Sort into display order ──────────────────────────────────────────────
    def sort_key(key):
        variant, encoder, tighten = key
        v_idx = VARIANT_ORDER.index(variant) if variant in VARIANT_ORDER else 999
        return (v_idx, encoder, tighten)

    sorted_keys = sorted(agg_rows.keys(), key=sort_key)
    rows = [(v, enc, t, agg_rows[(v, enc, t)]) for v, enc, t in sorted_keys]

    # ── Terminal output ──────────────────────────────────────────────────────
    print(make_terminal_table(rows, encoders_seen))
    print()

    # Per-encoder sub-tables if multiple encoders present
    if len(encoders_seen) > 1:
        for enc in sorted(encoders_seen):
            enc_rows = [r for r in rows if r[1] == enc]
            if enc_rows:
                print(f"\n=== Encoder: {enc} ===")
                print(make_terminal_table(enc_rows, {enc}))

    # ── LaTeX output ─────────────────────────────────────────────────────────
    latex = make_latex_table(rows, caption=args.caption, label=args.label)
    print("\n" + "=" * 60)
    print("LaTeX table:")
    print("=" * 60)
    print(latex)

    out_path = args.out if args.out else os.path.join(args.results_dir, "results_table.tex")
    with open(out_path, "w") as f:
        f.write(latex + "\n")
    print(f"\nLaTeX saved -> {out_path}")


if __name__ == "__main__":
    main()