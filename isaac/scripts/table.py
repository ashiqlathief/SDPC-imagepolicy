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

Obstacle collision check (XY footprint) -- the drone is modeled as a disc of
radius drone_radius, matching eval_crazieflie.py's build_obstacle_constraint_list
("sphere_outside" constraints), NOT an axis-aligned square/point check:
  box      : circle   (x-cx)^2 + (y-cy)^2 < (0.15 + drone_radius)^2
  cylinder : circle   (x-cx)^2 + (y-cy)^2 < (0.06 + drone_radius)^2

Usage
─────
    python make_results_table.py --dir path/to/npz/folder
    python make_results_table.py --dir . --out results_table.tex

Flags
─────
    --dir   folder containing *.npz files  (default: current dir)
    --out           write LaTeX to this file        (default: print only)
    --box_radius    box exclusion radius for collision (default: 0.15, matches eval_crazieflie.py)
    --cyl_radius    cylinder radius for collision   (default: 0.06)

Note: the LaTeX table zebra-stripes rows via rowcolors, which needs
xcolor loaded with the "table" option (not plain \\usepackage{xcolor})
in whatever document includes it.
"""

import argparse
import os
import glob
import subprocess
import numpy as np
from collections import defaultdict


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

GATE_X      = 4.0   # x position of the goal gate
DRONE_RADIUS = 0.08  # matches crazyflie_env.py drone_radius default

# ── graded adjusted-success penalties ─────────────────────────────────────────
# altitude: no penalty inside [ALT_LOW, ALT_HIGH]; outside it, the multiplier
# drops linearly with distance from the nearest band edge, e.g. with the
# defaults below z=0.9 is 0.2m above the band -> 0.5*0.2 = 0.10 penalty.
ALT_LOW           = 0.6
ALT_HIGH          = 0.7
ALT_PENALTY_PER_M = 1.0


def dir_birthtime(path):
    """True directory creation time via `stat`, if the filesystem tracks it.
    Unlike mtime (which advances every time a new file is added inside the
    directory -- i.e. every episode save bumps it), birth time is fixed at
    creation, right before the first episode's rollout starts. Needed as the
    wall-clock reference point for the FIRST variant saved in a run, which has
    no earlier sibling npz to diff mtimes against. Returns None if unsupported
    (e.g. non-Linux, or a filesystem without birth-time support) -- callers
    should treat that as "no estimate available", same as before this existed.
    """
    try:
        out = subprocess.run(["stat", "-c", "%W", path], capture_output=True, text=True, timeout=5)
        val = float(out.stdout.strip())
        return val if val > 0 else None
    except Exception:
        return None

def point_hits_obstacle(x, y, boxes, cylinders, box_thresh_sq, cyl_thresh_sq):
    """True if a disc-modeled drone centered at (x, y) overlaps any box or
    cylinder obstacle. Boxes are treated as circular exclusion zones too,
    matching eval_crazieflie.py's build_obstacle_constraint_list, which uses
    a "sphere_outside" (disc) constraint for boxes -- not an axis-aligned
    square -- so this is a point-vs-disc check for both obstacle types."""
    for bx, by in boxes:
        if (x - bx) ** 2 + (y - by) ** 2 < box_thresh_sq:
            return True
    for cx, cy in cylinders:
        if (x - cx) ** 2 + (y - cy) ** 2 < cyl_thresh_sq:
            return True
    return False


def point_violates_halfspace(x, y, halfspaces):
    """True if (x, y) is on the infeasible side of any halfspace constraint.
    Vertical boundaries are skipped (matching plotall.py's rendering, which
    draws them as unenforced reference lines, not shaded infeasible regions)."""
    for p1, p2, side in halfspaces:
        x1, y1 = float(p1[0]), float(p1[1])
        x2, y2 = float(p2[0]), float(p2[1])
        if abs(x2 - x1) < 1e-8:
            continue
        m = (y2 - y1) / (x2 - x1)
        line_y = m * x + (y1 - m * x1)
        # matches plotall.py's shading: side=="above" shades (marks infeasible)
        # BELOW the line, side=="below" shades ABOVE it -- these labels name
        # the feasible side, not the infeasible one.
        if side == "above" and y < line_y:
            return True
        if side == "below" and y > line_y:
            return True
    return False


def count_violations(xyz, boxes, cylinders, box_radius, cyl_radius, drone_radius=DRONE_RADIUS,
                      halfspaces=None):
    """Count timesteps where the drone body overlaps any obstacle footprint
    or crosses a halfspace constraint. Uses box_radius/cyl_radius + drone_radius
    to match the collision check in eval_crazieflie.py (disc-vs-disc for both
    obstacle types)."""
    box_thresh_sq = (box_radius + drone_radius) ** 2
    cyl_thresh_sq = (cyl_radius + drone_radius) ** 2
    halfspaces = halfspaces or []
    violations = 0
    for t in range(len(xyz)):
        x, y = float(xyz[t, 0]), float(xyz[t, 1])
        if (point_hits_obstacle(x, y, boxes, cylinders, box_thresh_sq, cyl_thresh_sq)
                or point_violates_halfspace(x, y, halfspaces)):
            violations += 1
    return violations


def count_horizon_violations(cand_traj_xy, snap_chosen, boxes, cylinders, box_radius, cyl_radius,
                              drone_radius=DRONE_RADIUS, halfspaces=None):
    """Among the CHOSEN candidate's planned horizon at each replanning step
    (not the realized/executed trajectory), count how many of those horizons
    cross an obstacle or halfspace at ANY point along their look-ahead --
    i.e. how often the policy committed to a plan that was unsafe somewhere
    in its future, even if replanning/tracking meant it was never actually
    executed all the way through."""
    if cand_traj_xy is None or np.asarray(cand_traj_xy).size == 0:
        return 0
    box_thresh_sq = (box_radius + drone_radius) ** 2
    cyl_thresh_sq = (cyl_radius + drone_radius) ** 2
    halfspaces = halfspaces or []
    horizon_violations = 0
    for si in range(cand_traj_xy.shape[0]):
        chosen_k = int(snap_chosen[si])
        rollout = cand_traj_xy[si, chosen_k]   # (H+1, 2) or (H+1, 3) if z is present
        for x, y in rollout[:, :2]:
            x, y = float(x), float(y)
            if (point_hits_obstacle(x, y, boxes, cylinders, box_thresh_sq, cyl_thresh_sq)
                    or point_violates_halfspace(x, y, halfspaces)):
                horizon_violations += 1
                break
    return horizon_violations


def altitude_score(xyz, alt_low=ALT_LOW, alt_high=ALT_HIGH, penalty_per_m=ALT_PENALTY_PER_M):
    """1.0 if the FINAL altitude landed inside [alt_low, alt_high]; otherwise
    penalized by penalty_per_m times the final z's distance outside the band,
    clipped to [0, 1]. Only the end-of-episode altitude matters -- mid-flight
    excursions (e.g. during takeoff) are not penalized."""
    z_final = float(xyz[-1, 2])
    deviation = max(0.0, alt_low - z_final, z_final - alt_high)
    return float(np.clip(1.0 - penalty_per_m * deviation, 0.0, 1.0))


def load_episode(path, box_radius, cyl_radius, drone_radius=DRONE_RADIUS, wall_time_fallback_sec=None,
                  alt_low=ALT_LOW, alt_high=ALT_HIGH, alt_penalty_per_m=ALT_PENALTY_PER_M):
    """Return a dict of scalar metrics for one .npz file.

    wall_time_fallback_sec: mtime-diff estimate for this file (see main()),
    used only if the npz predates the episode_wall_time_sec instrumentation
    added to eval_crazieflie.py. Real timing (when present) always wins.
    """
    d = np.load(path, allow_pickle=True)

    xyz       = d["xyz"]                          # (T, 3)
    actions   = d["actions"]                      # (T-1, 3)
    success   = bool(d["success"])
    fell      = bool(d["fell"])
    variant   = str(d["variant"])
    if variant.startswith("dpcc-"):
        # "dpcc" is this project's old name for the "sdpc" variant family --
        # older npz files still carry it in their saved variant field.
        variant = "sdpc-" + variant[len("dpcc-"):]
    encoder   = str(d["encoder"])
    tighten   = float(d["tighten"])
    latent    = int(d["latent_dim"]) if "latent_dim" in d else -1

    boxes     = d["boxes"]      if "boxes"     in d else np.zeros((0, 2))
    cylinders = d["cylinders"]  if "cylinders" in d else np.zeros((0, 2))

    if "halfspaces" in d and len(d["halfspaces"]) > 0:
        hs_pts   = d["halfspaces"].tolist()
        hs_sides = d["hs_sides"].tolist() if "hs_sides" in d else ["below"] * len(hs_pts)
        halfspaces = [[pts[0], pts[1], side] for pts, side in zip(hs_pts, hs_sides)]
    else:
        halfspaces = []

    cand_traj_xy = np.asarray(d["cand_traj_xy"]) if "cand_traj_xy" in d else np.zeros((0, 0, 0, 2))
    snap_chosen  = np.asarray(d["snap_chosen"])  if "snap_chosen"  in d else np.zeros((0,), dtype=int)

    executed_violations = count_violations(xyz, boxes, cylinders, box_radius, cyl_radius, drone_radius,
                                            halfspaces=halfspaces)
    horizon_violations  = count_horizon_violations(cand_traj_xy, snap_chosen, boxes, cylinders,
                                                    box_radius, cyl_radius, drone_radius,
                                                    halfspaces=halfspaces)
    violations  = executed_violations + horizon_violations
    alt_score   = altitude_score(xyz, alt_low, alt_high, alt_penalty_per_m)
    adjusted_success = alt_score if success else 0.0
    path_length   = float(np.linalg.norm(actions, axis=1).sum())
    final_x       = float(xyz[-1, 0])
    final_y       = float(xyz[-1, 1])
    final_z       = float(xyz[-1, 2])
    max_x         = float(xyz[:, 0].max())
    # progress rate: how far along the corridor the drone got (0.0 – 1.0)
    # e.g. reaches x=2 before failing → 2/4 = 0.50
    progress_rate = float(min(max_x, GATE_X) / GATE_X)

    if "episode_wall_time_sec" in d:
        wall_time_sec  = float(d["episode_wall_time_sec"])
        wall_time_real = True
    else:
        wall_time_sec  = wall_time_fallback_sec   # None if unavailable (e.g. first file in a run)
        wall_time_real = False

    return dict(
        variant       = variant,
        encoder       = encoder,
        tighten       = tighten,
        latent        = latent,
        timesteps     = len(xyz),
        success       = 1 if success else 0,
        fell          = 1 if fell else 0,
        violations    = violations,
        altitude_score   = alt_score,
        adjusted_success = adjusted_success,
        progress_rate = progress_rate,
        path_length   = path_length,
        final_x       = final_x,
        final_y       = final_y,
        final_z       = final_z,
        max_x         = max_x,
        wall_time_sec  = wall_time_sec,
        wall_time_real = wall_time_real,
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
    sr  = np.array([e["adjusted_success"] for e in episodes], dtype=float)
    v   = np.array([e["violations"]  for e in episodes], dtype=float)
    pl  = np.array([e["path_length"] for e in episodes], dtype=float)
    fx  = np.array([e["final_x"]     for e in episodes], dtype=float)
    fy  = np.array([e["final_y"]     for e in episodes], dtype=float)
    fz  = np.array([e["final_z"]     for e in episodes], dtype=float)
    mx  = np.array([e["max_x"]       for e in episodes], dtype=float)

    pr  = np.array([e["progress_rate"] for e in episodes], dtype=float)

    # wall-clock time: some episodes may have no value (first file in a run,
    # no preceding mtime to diff against) -- drop those rather than treat as 0.
    wt_secs = [e["wall_time_sec"] for e in episodes if e["wall_time_sec"] is not None]
    wt_min  = np.array(wt_secs, dtype=float) / 60.0
    any_estimate = any((not e["wall_time_real"]) for e in episodes if e["wall_time_sec"] is not None)

    def ms(arr, fmt=".2f"):
        if n == 1:
            return f"{arr.mean():{fmt}}"
        return f"{arr.mean():{fmt}} \\pm {arr.std():{fmt}}"

    def rate_str(arr):
        s = f"{arr.mean():.2f}"
        if n > 1:
            s += f" \\pm {arr.std():.2f}"
        return s

    def wall_time_str():
        # "*" flags that at least one contributing episode has no real
        # episode_wall_time_sec (old npz, predates the timing instrumentation
        # in eval_crazieflie.py) and fell back to an mtime-diff estimate,
        # which is wrong if that run was ever paused/resumed between episodes.
        if len(wt_min) == 0:
            return "--"   # matches the "not evaluated" convention used elsewhere
        s = f"{wt_min.mean():.1f}"
        if len(wt_min) > 1:
            s += f" \\pm {wt_min.std():.1f}"
        if any_estimate:
            s += "*"
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
        wall_time     = wall_time_str(),
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

def rank_best_second(means, higher_is_better, eps=1e-3):
    """Best and second-best *distinct* values from a list of column means.
    Ties at best are all marked green by fmt_cell (matches the reference
    table style where multiple tied cells in a column are all highlighted).
    second is None if every value ties with best."""
    uniq = sorted({round(float(v), 6) for v in means}, reverse=higher_is_better)
    best = uniq[0]
    second = next((u for u in uniq[1:] if abs(u - best) > eps), None)
    return best, second


def fmt_cell(raw_str, mean_val, best, second, eps=1e-3):
    """Wrap a formatted 'mean \\pm std' string in $...$, adding a cellcolor +
    bold for the column's best value (ties included) or cellcolor alone for
    the second-best. \\cellcolor must sit outside the $...$ math shell --
    \\boldsymbol (not \\textbf) is used inside it since these strings can
    contain \\pm, which is math-mode-only and breaks inside \\textbf's
    text-mode argument."""
    if abs(mean_val - best) < eps:
        return f"\\cellcolor{{bestgreen}}$\\boldsymbol{{{raw_str}}}$"
    if second is not None and abs(mean_val - second) < eps:
        return f"\\cellcolor{{secondblue}}${raw_str}$"
    return f"${raw_str}$"


def make_terminal_table(rows, encoders):
    """rows: list of (variant, encoder, tighten, agg_dict)"""
    col_w = [30, 8, 10, 7, 8, 10, 8, 8, 8, 8, 12]
    header = (
        f"{'Variant':<{col_w[0]}} {'Encoder':<{col_w[1]}}"
        f" {'Steps':>{col_w[2]}} {'SR':>{col_w[3]}} {'Progress':>{col_w[4]}}"
        f" {'Violations':>{col_w[5]}} {'PathLen':>{col_w[6]}}"
        f" {'Final-x':>{col_w[7]}} {'Final-y':>{col_w[8]}} {'Final-z':>{col_w[9]}}"
        f" {'Time(min)':>{col_w[10]}}"
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
            f" {agg['wall_time']:>{col_w[10]}}"
            f"  (n={agg['n']})"
        )
    lines.append(sep)
    return "\n".join(lines)


def make_latex_table(rows, caption="", label="tab:results"):
    """
    Booktabs LaTeX table, styled after the ParetoDiffMPC paper's Table I:
    shaded+centered header, green/blue cellcolor for best/second-best per
    column (ties included), grey/white zebra-striped body rows.
    Columns: Variant | Encoder | Timesteps | SR | Progress | Violations | Path Length | Final x | Final y | Final z | Wall time

    "Violations" combines executed-trajectory obstacle/halfspace overlaps with
    "horizon violations" -- timesteps where the CHOSEN candidate's planned
    look-ahead crossed an obstacle/halfspace, even if never actually executed
    (see count_violations / count_horizon_violations in load_episode).

    Wall time is informational, not ranked/colored -- a "*" suffix means at
    least one contributing episode predates the episode_wall_time_sec
    instrumentation and used an mtime-diff estimate instead (see load_episode).
    """
    best_ts, second_ts = rank_best_second([r[3]["_ts_mean"] for r in rows], higher_is_better=False)
    best_sr, second_sr = rank_best_second([r[3]["_sr_mean"] for r in rows], higher_is_better=True)
    best_pr, second_pr = rank_best_second([r[3]["_pr_mean"] for r in rows], higher_is_better=True)
    best_v,  second_v  = rank_best_second([r[3]["_v_mean"]  for r in rows], higher_is_better=False)
    best_pl, second_pl = rank_best_second([r[3]["_pl_mean"] for r in rows], higher_is_better=False)

    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"  \centering")
    lines.append(r"  \small")
    if caption:
        lines.append(f"  \\caption{{{caption}}}")
    lines.append(f"  \\label{{{label}}}")
    # Colors: bestgreen/secondblue match the names+values already defined at
    # the top of Results.tex (redefining with identical RGB is harmless if
    # both end up \input in the same document); headershade is new, a pale
    # blue-grey matching the reference table's header row.
    lines.append(r"  \definecolor{bestgreen}{RGB}{198,239,206}")
    lines.append(r"  \definecolor{secondblue}{RGB}{204,229,255}")
    lines.append(r"  \definecolor{headershade}{RGB}{222,235,247}")
    lines.append(r"  \rowcolors{2}{gray!10}{white}")
    lines.append(r"  \begin{tabular}{llrrrrrrrrr}")
    lines.append(r"  \toprule")
    lines.append(r"  \rowcolor{headershade}")
    lines.append(
        r"  \multicolumn{1}{c}{\textbf{Variant}} & \multicolumn{1}{c}{\textbf{Enc.}}"
        r" & \multicolumn{1}{c}{\textbf{Steps}} & \multicolumn{1}{c}{\textbf{SR}}"
        r" & \multicolumn{1}{c}{\textbf{Progress}} & \multicolumn{1}{c}{\textbf{Violations}}"
        r" & \multicolumn{1}{c}{\textbf{Path (m)}} & \multicolumn{1}{c}{\textbf{$x_f$}}"
        r" & \multicolumn{1}{c}{\textbf{$y_f$}} & \multicolumn{1}{c}{\textbf{$z_f$}}"
        r" & \multicolumn{1}{c}{\textbf{Time (min)}} \\"
    )
    lines.append(r"  \midrule")

    prev_variant = None
    for variant, encoder, tighten, agg in rows:
        label_str = VARIANT_LABEL.get(variant, variant)
        if prev_variant is not None and variant != prev_variant:
            lines.append(r"  \addlinespace[2pt]")
        prev_variant = variant

        ts_cell = fmt_cell(agg["timesteps"],      agg["_ts_mean"], best_ts, second_ts)
        sr_cell = fmt_cell(agg["success"],        agg["_sr_mean"], best_sr, second_sr)
        pr_cell = fmt_cell(agg["progress_rate"],  agg["_pr_mean"], best_pr, second_pr)
        v_cell  = fmt_cell(agg["violations"],     agg["_v_mean"],  best_v,  second_v)
        pl_cell = fmt_cell(agg["path_length"],    agg["_pl_mean"], best_pl, second_pl)

        # wall_time is text ("--") when unavailable -- only wrap in math mode
        # when it actually has \pm content, so "--" renders as a plain en-dash
        # (matching the other "not evaluated" cells) instead of two math minuses.
        wt_str  = agg["wall_time"]
        wt_cell = wt_str if wt_str == "--" else f"${wt_str}$"

        lines.append(
            f"  {label_str} & {encoder}"
            f" & {ts_cell} & {sr_cell} & {pr_cell}"
            f" & {v_cell} & {pl_cell}"
            f" & ${agg['final_x']}$ & ${agg['final_y']}$ & ${agg['final_z']}$"
            f" & {wt_cell} \\\\"
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
    parser.add_argument("--dir", nargs="+", default=["."],
                        help="One or more folders containing *.npz result files "
                             "(e.g. one trajectories/ dir per seed -- episodes across "
                             "all given dirs are pooled per (variant, encoder, tighten) group)")
    parser.add_argument("--out",         default=None,
                        help="Write LaTeX table to this .tex file")
    parser.add_argument("--box_radius",  type=float, default=0.15,
                        help="Box exclusion radius in metres for collision -- boxes are "
                             "treated as discs, matching eval_crazieflie.py's "
                             "build_obstacle_constraint_list (default 0.15)")
    parser.add_argument("--cyl_radius",  type=float, default=0.06,
                        help="Cylinder radius in metres (default 0.06)")
    parser.add_argument("--drone_radius", type=float, default=DRONE_RADIUS,
                        help=f"Drone body radius for Minkowski collision (default {DRONE_RADIUS})")
    parser.add_argument("--alt_low",     type=float, default=ALT_LOW,
                        help=f"Lower edge of the no-penalty altitude band, metres (default {ALT_LOW})")
    parser.add_argument("--alt_high",    type=float, default=ALT_HIGH,
                        help=f"Upper edge of the no-penalty altitude band, metres (default {ALT_HIGH})")
    parser.add_argument("--alt_penalty", type=float, default=ALT_PENALTY_PER_M,
                        help="Adjusted-SR penalty per metre of final-altitude deviation outside "
                             f"the altitude band (default {ALT_PENALTY_PER_M})")
    parser.add_argument("--caption",     default=(
                        "Comparison of safety variants across encoders. "
                        "Mean $\\pm$ std over episodes."))
    parser.add_argument("--label",       default="tab:results")
    args = parser.parse_args()

    box_radius   = args.box_radius
    cyl_radius   = args.cyl_radius
    drone_radius = args.drone_radius

    # ── Load all npz files (pooled across every --dir given) ─────────
    paths = sorted(
        p for d in args.dir for p in glob.glob(os.path.join(d, "*.npz"))
    )
    if not paths:
        print(f"[ERROR] No .npz files found in: {args.dir}")
        return

    print(f"Found {len(paths)} npz files across {len(args.dir)} dir(s): {args.dir}\n")

    # ── Wall-clock fallback: for npz files that predate episode_wall_time_sec,
    # estimate per-episode time from the gap between consecutive file mtimes
    # within the SAME directory (files are saved in projection_variants
    # execution order, so mtime order == run order). Only valid within one
    # directory/run -- never diff mtimes across different --dir.
    # The first file in each directory has no preceding npz to diff against;
    # use the directory's own birth time instead (its mtime is NOT usable
    # here -- mtime advances every time a new file is added inside it, so by
    # the time this script runs it just equals the LAST file's timestamp).
    mtime_fallback = {}
    for d in args.dir:
        dir_files = sorted(glob.glob(os.path.join(d, "*.npz")), key=os.path.getmtime)
        prev_mtime = dir_birthtime(d)   # None if unsupported -- first file stays unestimated
        for f in dir_files:
            mt = os.path.getmtime(f)
            mtime_fallback[f] = (mt - prev_mtime) if prev_mtime is not None else None
            prev_mtime = mt

    # group: (variant, encoder, tighten) -> list of episode dicts
    groups = defaultdict(list)
    encoders_seen = set()

    for p in paths:
        try:
            ep = load_episode(p, box_radius, cyl_radius, drone_radius,
                               wall_time_fallback_sec=mtime_fallback.get(p),
                               alt_low=args.alt_low, alt_high=args.alt_high,
                               alt_penalty_per_m=args.alt_penalty)
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

    out_path = args.out if args.out else os.path.join(args.dir[0], "results_table.tex")
    with open(out_path, "w") as f:
        f.write(latex + "\n")
    print(f"\nLaTeX saved -> {out_path}")


if __name__ == "__main__":
    main()