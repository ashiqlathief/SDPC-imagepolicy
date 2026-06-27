"""
view_traj.py  —  trajectory plot + metrics table for one trajectories* folder
Usage:
    python view_traj.py --dir path/to/trajectories8
    python view_traj.py --dir path/to/trajectories8 --out fig.pdf
"""
import argparse, glob, importlib, os, re, sys, types
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle, Polygon

# ── load config (lightweight — mock heavy deps) ───────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
_du = types.ModuleType("diffuser.utils")
_du.watch = lambda *a, **kw: (lambda fn: fn)
sys.modules.setdefault("diffuser",       types.ModuleType("diffuser"))
sys.modules.setdefault("diffuser.utils", _du)
_cfg = importlib.import_module("config.avoiding-crazyflie")
CONFIG_HALFSPACES = list(getattr(_cfg, "CORRIDOR_HALFSPACES", []))

# ── variant colours ──────────────────────────────────────────────────────────
STYLE = {
    "sdpc-c":             dict(color="tab:green",   lw=1.8, ls="-",  label="SDPC-C"),
    "sdpc-r":             dict(color="tab:red",     lw=1.8, ls="-",  label="SDPC-R"),
    "sdpc-r-tightened":   dict(color="hotpink",     lw=1.8, ls="--", label="SDPC-R (tightened)"),
    "sdpc-t":             dict(color="tab:blue",    lw=1.8, ls="-",  label="SDPC-T"),
    "sdpc-t-tightened":   dict(color="steelblue",   lw=1.8, ls="--", label="SDPC-T (tightened)"),
    "sdpc-c-tightened":   dict(color="limegreen",   lw=1.8, ls="--", label="SDPC-C (tightened)"),
    "diffuser":           dict(color="darkorange",  lw=1.8, ls="-",  label="Diffuser"),
    "gradient":           dict(color="purple",      lw=1.8, ls="-",  label="Guidance"),
    "gradient-tightened": dict(color="violet",      lw=1.8, ls="--", label="Guidance (tightened)"),
    "post_processing":    dict(color="saddlebrown", lw=1.8, ls="-",  label="Post-Proc."),
    "model_free":         dict(color="gray",        lw=1.8, ls="-",  label="Model-Free"),
}

CYL_R   = 0.06
DRONE_R = 0.08
WALL_Y  = 1.00
WALL_W  = 0.10


# ─────────────────────────────────────────────────────────────────────────────
# LaTeX table parser
# ─────────────────────────────────────────────────────────────────────────────

def _strip_latex(s):
    s = re.sub(r'\\textbf\{([^}]*)\}', r'\1', s)   # \textbf{x} → x
    s = re.sub(r'\$([^$]*)\$', r'\1', s)            # $x$ → x
    s = re.sub(r'\\pm', '±', s)
    return s.strip()


def parse_tex_table(tex_path):
    """Return (headers: list[str], rows: list[list[str]]) from a booktabs table."""
    with open(tex_path) as f:
        content = f.read()

    headers, rows, in_tab, past_mid = [], [], False, False
    for line in content.splitlines():
        line = line.strip()
        if r'\begin{tabular}' in line:
            in_tab = True;  continue
        if r'\end{tabular}' in line:
            break
        if not in_tab or '&' not in line:
            if r'\midrule' in line:
                past_mid = True
            continue
        if r'\addlinespace' in line or r'\bottomrule' in line:
            continue

        cols = [_strip_latex(c) for c in line.rstrip('\\').split('&')]
        if not past_mid:
            headers = cols
        else:
            rows.append(cols)

    return headers, rows


# ─────────────────────────────────────────────────────────────────────────────
# Oscillation band helper
# ─────────────────────────────────────────────────────────────────────────────

def oscillation_band(x0, y0, amp, axis, cyl_r, alpha=0.15):
    excl = cyl_r
    if axis == "y":
        return mpatches.Rectangle(
            (x0 - excl, y0 - amp - excl), 2*excl, 2*amp + 2*excl,
            facecolor="tab:orange", alpha=alpha, zorder=1, linewidth=0)
    if axis == "x":
        return mpatches.Rectangle(
            (x0 - amp - excl, y0 - excl), 2*amp + 2*excl, 2*excl,
            facecolor="tab:orange", alpha=alpha, zorder=1, linewidth=0)
    # "xy" — 45° diagonal strip
    p1 = np.array([x0 - amp, y0 - amp])
    p2 = np.array([x0 + amp, y0 + amp])
    seg = p2 - p1
    d   = seg / (np.linalg.norm(seg) + 1e-8)
    n   = np.array([-d[1], d[0]])
    corners = np.array([p1-excl*n, p1+excl*n, p2+excl*n, p2-excl*n])
    return Polygon(corners, facecolor="tab:orange", alpha=alpha, zorder=1, linewidth=0)


# ─────────────────────────────────────────────────────────────────────────────
# Halfspace constraint overlay
# ─────────────────────────────────────────────────────────────────────────────

def plot_halfspace_constraints_xy(ax, halfspaces, xlim, ylim, alpha=0.10):
    """
    halfspaces: list of [p1, p2, side] where p1/p2 are [x,y] and side in {'above','below'}
    Draws the boundary line and shades the infeasible half-plane.
    """
    xmin, xmax = xlim
    ymin, ymax = ylim

    for hs in halfspaces:
        p1, p2, side = hs
        x1, y1 = p1
        x2, y2 = p2

        if abs(x2 - x1) < 1e-8:
            ax.axvline(x1, color="royalblue", linewidth=1.8, alpha=0.7, linestyle="--")
            continue

        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        xs = np.array([xmin, xmax], dtype=np.float64)
        ys = m * xs + b
        ax.plot(xs, ys, color="royalblue", linewidth=1.8, alpha=0.7, linestyle="--")
        if side == "above":
            ax.fill_between(xs, ys, ymin, color="royalblue", alpha=alpha)
        else:
            ax.fill_between(xs, ys, ymax, color="royalblue", alpha=alpha)


# ─────────────────────────────────────────────────────────────────────────────
# NPZ loader
# ─────────────────────────────────────────────────────────────────────────────

def load_npz(path):
    d = np.load(path, allow_pickle=True)
    # corridor halfspaces: list of [p1, p2] pairs, plus sides list
    if "halfspaces" in d and len(d["halfspaces"]) > 0:
        hs_pts  = d["halfspaces"].tolist()   # list of [[x1,y1],[x2,y2]]
        hs_sides = d["hs_sides"].tolist() if "hs_sides" in d else ["below"] * len(hs_pts)
        halfspaces = [[pts[0], pts[1], side] for pts, side in zip(hs_pts, hs_sides)]
    else:
        halfspaces = []
    return dict(
        xyz        = d["xyz"],
        variant    = str(d["variant"]),
        success    = bool(d["success"]),
        fell       = bool(d["fell"]),
        cylinders  = d["cylinders"] if "cylinders" in d else np.zeros((0, 2)),
        dyn_idx    = d["dynamic_cyl_indices"].tolist() if "dynamic_cyl_indices" in d else [],
        amp        = float(d["obs_amplitude"]) if "obs_amplitude" in d else 0.0,
        axes       = list(d["obs_axes"])       if "obs_axes"      in d else [],
        halfspaces = halfspaces,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main figure builder
# ─────────────────────────────────────────────────────────────────────────────

def make_figure(folder, out=None):
    paths    = sorted(glob.glob(os.path.join(folder, "*.npz")))
    tex_path = os.path.join(folder, "results_table.tex")
    has_tex  = os.path.isfile(tex_path)

    if not paths:
        print(f"No .npz files found in {folder}")
        return

    episodes = [load_npz(p) for p in paths]

    # ── layout: trajectory on top, table below (if tex exists) ───────────────
    if has_tex:
        headers, tbl_rows = parse_tex_table(tex_path)
        n_rows = len(tbl_rows)
        row_h  = 0.38                              # height per table row (inches)
        tbl_h  = (n_rows + 1) * row_h + 0.3       # +1 header, +0.3 padding
        fig = plt.figure(figsize=(13, 4.5 + tbl_h))
        gs  = gridspec.GridSpec(2, 1,
                                height_ratios=[4.5, tbl_h],
                                hspace=0.35)
        ax_traj = fig.add_subplot(gs[0])
        ax_tbl  = fig.add_subplot(gs[1])
    else:
        fig, ax_traj = plt.subplots(figsize=(11, 4))
        ax_tbl = None

    # ── trajectory panel ─────────────────────────────────────────────────────
    ax = ax_traj

    # walls
    for sign in (1, -1):
        ax.add_patch(mpatches.Rectangle(
            (-0.5, sign * WALL_Y), 5.0, sign * WALL_W,
            facecolor="#cccccc", zorder=0, linewidth=0))

    # obstacles (same in every episode)
    ep0    = episodes[0]
    cyls   = ep0["cylinders"]
    dyn    = [int(i) for i in ep0["dyn_idx"]]
    amp    = ep0["amp"]
    axes_v = ep0["axes"]

    def get_axis(k):
        return axes_v[0] if len(axes_v) == 1 else (axes_v[k] if k < len(axes_v) else axes_v[0])

    for k, idx in enumerate(dyn):
        ax.add_patch(oscillation_band(
            float(cyls[idx, 0]), float(cyls[idx, 1]), amp, get_axis(k), CYL_R))

    for cx, cy in cyls:
        ax.add_patch(Circle((cx, cy), CYL_R,
                            facecolor="tab:orange", edgecolor="k", lw=0.5, zorder=2))
        ax.add_patch(Circle((cx, cy), CYL_R + DRONE_R,
                            facecolor="none", edgecolor="gray", lw=0.8, ls=":", zorder=2))

    ax.axvline(4.0, color="limegreen", lw=2, zorder=3)

    # halfspace constraints: only plot what was active during the recorded run
    halfspaces = ep0.get("halfspaces", [])
    hs_handle = []
    if halfspaces:
        plot_halfspace_constraints_xy(ax, halfspaces, xlim=(-0.5, 4.5), ylim=(-1.25, 1.25))
        hs_handle = [mpatches.Patch(facecolor="royalblue", alpha=0.3, label="halfspace constraint")]

    # trajectories
    legend_handles = []
    for ep in episodes:
        v  = ep["variant"]
        st = STYLE.get(v, dict(color="black", lw=1.5, ls="-", label=v))
        ok = ep["success"]
        xyz = ep["xyz"]

        ax.plot(xyz[:, 0], xyz[:, 1], color=st["color"], lw=st["lw"], ls=st["ls"], zorder=4)
        ax.plot(xyz[0, 0],  xyz[0, 1],  "o", color=st["color"], ms=5, zorder=5)
        ax.plot(xyz[-1, 0], xyz[-1, 1], "^" if ok else "x",
                color=st["color"], ms=8, mew=2, zorder=5)

        mark = "✓" if ok else "✗ fell" if ep["fell"] else "✗"
        legend_handles.append(mpatches.Patch(facecolor=st["color"],
                                             label=f"{st['label']}  {mark}"))

    legend_handles += [
        mpatches.Patch(facecolor="tab:orange", alpha=0.4, label=f"Oscillation (±{amp} m)"),
        mpatches.Patch(facecolor="none", edgecolor="gray", linestyle=":",
                       label=f"Constraint margin ({DRONE_R} m)"),
    ] + hs_handle

    ax.legend(handles=legend_handles, fontsize=8,
              bbox_to_anchor=(1.01, 1), loc="upper left", borderaxespad=0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(-1.25, 1.25)
    ax.set_xlabel("x  (m)")
    ax.set_ylabel("y  (m)")

    latent = str(paths[0]).split("_L")[-1].split("_")[0] if "_L" in paths[0] else ""
    ax.set_title(f"Trajectories — latent dim {latent}  [Top-down XY]", fontsize=11)

    # ── table panel ───────────────────────────────────────────────────────────
    if ax_tbl is not None and tbl_rows:
        ax_tbl.axis("off")

        tbl = ax_tbl.table(
            cellText=tbl_rows,
            colLabels=headers,
            cellLoc="center",
            loc="upper center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8.5)
        tbl.auto_set_column_width(list(range(len(headers))))

        # style header row
        for col in range(len(headers)):
            cell = tbl[0, col]
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")

        # alternating row shading + colour-match variant column
        variant_col_idx = 0
        for row_i, row_data in enumerate(tbl_rows, start=1):
            bg = "#f0f4f8" if row_i % 2 == 0 else "white"
            for col in range(len(headers)):
                cell = tbl[row_i, col]
                cell.set_facecolor(bg)
                cell.set_edgecolor("#dddddd")

            # tint the variant cell with its trajectory colour
            v_key = row_data[variant_col_idx].lower().replace(" ", "-").replace("(", "").replace(")", "")
            if v_key in STYLE:
                tbl[row_i, variant_col_idx].set_facecolor(STYLE[v_key]["color"])
                tbl[row_i, variant_col_idx].set_text_props(color="white", fontweight="bold")

        # # place title just above the table, not at top of blank axes
        # ax_tbl.text(0.0, 1.02, "Metrics  (from results_table.tex)",
        #             transform=ax_tbl.transAxes, fontsize=9,
        #             va="bottom", ha="left")

    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    print(f"Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True,
                        help="trajectories* folder containing *.npz (and optionally results_table.tex)")
    parser.add_argument("--out", default=None,
                        help="Output file (e.g. view.pdf). Defaults to <dir>/view_traj.pdf.")
    args = parser.parse_args()
    out = args.out or os.path.join(args.dir, "view_traj.pdf")
    make_figure(args.dir, out)
