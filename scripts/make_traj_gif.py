"""
make_traj_gif.py  —  animate drone + dynamic obstacle motion from a saved .npz trajectory.

Usage:
    python scripts/make_traj_gif.py path/to/traj_*.npz
    python scripts/make_traj_gif.py trajectories/          # whole directory
    python scripts/make_traj_gif.py traj.npz --fps 15 --dpi 120 --figsize 12 5 --out run.gif
    python scripts/make_traj_gif.py traj.npz --plane xz    # side view (x vs z / altitude)
    python scripts/make_traj_gif.py traj.npz --plane xyz   # 3D view
"""
import argparse
import glob
import importlib
import os
import sys
import types
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle, Rectangle, FancyArrowPatch
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ── load config (mock heavy deps so this script works standalone) ─────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
_du = types.ModuleType("diffuser.utils")
_du.watch = lambda *a, **kw: (lambda fn: fn)
sys.modules.setdefault("diffuser",       types.ModuleType("diffuser"))
sys.modules.setdefault("diffuser.utils", _du)
_cfg = importlib.import_module("config.avoiding-crazyflie")
CONFIG_HALFSPACES = list(getattr(_cfg, "CORRIDOR_HALFSPACES", []))


def _draw_halfspaces_xy(ax, halfspaces, xlim, ylim, alpha=0.12):
    """Draw halfspace boundary lines + infeasible shading on an XY axes."""
    xmin, xmax = xlim
    ymin, ymax = ylim
    for hs in halfspaces:
        p1, p2, side = hs
        x1, y1 = float(p1[0]), float(p1[1])
        x2, y2 = float(p2[0]), float(p2[1])
        if abs(x2 - x1) < 1e-8:
            ax.axvline(x1, color="royalblue", linewidth=1.8,
                       linestyle="--", alpha=0.75, zorder=2)
            continue
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        xs = np.array([xmin, xmax], dtype=np.float64)
        ys = m * xs + b
        ax.plot(xs, ys, color="royalblue", linewidth=1.8,
                linestyle="--", alpha=0.80, zorder=4)
        if side == "above":
            ax.fill_between(xs, ys, ymin, color="royalblue", alpha=alpha, zorder=2)
        else:
            ax.fill_between(xs, ys, ymax, color="royalblue", alpha=alpha, zorder=2)


def _make_gif_3d(
    npz_path: str,
    fps: int,
    out_path: str,
    figsize: tuple,
    dpi: int,
) -> str:
    """3D animated view of the corridor, obstacles and drone trajectory."""
    data = np.load(npz_path, allow_pickle=True)

    xyz         = data["xyz"]
    cylinders   = data["cylinders"]
    dynamic     = bool(data.get("dynamic_obstacles", False))
    cyl_xy_traj = data.get("cyl_xy_traj", None)
    dyn_indices = data.get("dynamic_cyl_indices", np.array([], dtype=int))

    has_spheres  = bool(data.get("floating_spheres", False))
    sph_pos_rest = np.array(data["sphere_positions"], dtype=np.float32) \
                   if "sphere_positions" in data else np.zeros((0, 3), np.float32)
    sph_radius   = float(data["sphere_radius"]) if "sphere_radius" in data else 0.10
    sph_xyz_traj = np.array(data["sph_xyz_traj"], dtype=np.float32) \
                   if "sph_xyz_traj" in data else np.zeros((0, 0, 3), np.float32)

    T = len(xyz)
    has_cyl_traj = (dynamic and cyl_xy_traj is not None
                    and cyl_xy_traj.shape[0] > 0)
    has_sph_traj = (has_spheres and sph_xyz_traj.shape[0] > 0
                    and sph_xyz_traj.shape[1] > 0)

    # ── figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax  = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_xlim(-0.2, 4.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_zlim(0.0, 1.2)
    ax.view_init(elev=25, azim=-60)

    # ── corridor wireframe ────────────────────────────────────────────────────
    cx0, cx1 = 0.0, 4.1
    cy0, cy1 = -1.0, 1.0
    cz0, cz1 = 0.0, 1.0
    wire_kw = dict(color="#888888", linewidth=0.8, alpha=0.35)
    # floor
    for ys, ye in [(cy0, cy0), (cy1, cy1)]:
        ax.plot([cx0, cx1], [ys, ye], [cz0, cz0], **wire_kw)
    for xs, xe in [(cx0, cx0), (cx1, cx1)]:
        ax.plot([xs, xe], [cy0, cy1], [cz0, cz0], **wire_kw)
    # ceiling
    for ys, ye in [(cy0, cy0), (cy1, cy1)]:
        ax.plot([cx0, cx1], [ys, ye], [cz1, cz1], **wire_kw)
    for xs, xe in [(cx0, cx0), (cx1, cx1)]:
        ax.plot([xs, xe], [cy0, cy1], [cz1, cz1], **wire_kw)
    # vertical corner edges
    for y in [cy0, cy1]:
        for x in [cx0, cx1]:
            ax.plot([x, x], [y, y], [cz0, cz1], **wire_kw)
    # goal plane outline
    ax.plot([4.0, 4.0], [cy0, cy1], [cz0, cz0], color="#00aa00", linewidth=1.5, alpha=0.7)
    ax.plot([4.0, 4.0], [cy0, cy1], [cz1, cz1], color="#00aa00", linewidth=1.5, alpha=0.7)
    ax.plot([4.0, 4.0], [cy0, cy0], [cz0, cz1], color="#00aa00", linewidth=1.5, alpha=0.7)
    ax.plot([4.0, 4.0], [cy1, cy1], [cz0, cz1], color="#00aa00", linewidth=1.5, alpha=0.7)

    # ── helper: draw a cylinder outline in 3D ─────────────────────────────────
    _th = np.linspace(0, 2 * np.pi, 32)
    def _draw_cyl_3d(ocx, ocy, color="#e87020", alpha=0.5, lw=1.0):
        xc = ocx + 0.06 * np.cos(_th)
        yc = ocy + 0.06 * np.sin(_th)
        ax.plot(xc, yc, np.zeros(32), color=color, linewidth=lw, alpha=alpha)
        ax.plot(xc, yc, np.ones(32),  color=color, linewidth=lw, alpha=alpha)
        ax.plot([ocx, ocx], [ocy, ocy], [0.0, 1.0], color=color,
                linewidth=lw * 2, alpha=alpha)

    # ── static cylinders ──────────────────────────────────────────────────────
    dyn_set = set(int(i) for i in dyn_indices)
    for i, c in enumerate(cylinders):
        if i not in dyn_set:
            _draw_cyl_3d(float(c[0]), float(c[1]))

    # ── dynamic cylinder artists (Line3D, updated each frame) ─────────────────
    dyn_cyl_artists = []
    for i in sorted(dyn_set):
        ocx, ocy = float(cylinders[i][0]), float(cylinders[i][1])
        xc = ocx + 0.06 * np.cos(_th)
        yc = ocy + 0.06 * np.sin(_th)
        bot, = ax.plot(xc, yc, np.zeros(32), color="#e87020", linewidth=1.0, alpha=0.85)
        top, = ax.plot(xc, yc, np.ones(32),  color="#e87020", linewidth=1.0, alpha=0.85)
        stem,= ax.plot([ocx, ocx], [ocy, ocy], [0.0, 1.0], color="#e87020",
                       linewidth=2.5, alpha=0.85)
        dyn_cyl_artists.append((i, bot, top, stem))

    # ── static sphere ghosts (wireframe drawn once) ───────────────────────────
    _pu = np.linspace(0, 2 * np.pi, 20)
    _pv = np.linspace(0, np.pi, 12)
    for spos in sph_pos_rest:
        sx, sy, sz = float(spos[0]), float(spos[1]), float(spos[2])
        xs = sx + sph_radius * np.outer(np.cos(_pu), np.sin(_pv))
        ys = sy + sph_radius * np.outer(np.sin(_pu), np.sin(_pv))
        zs = sz + sph_radius * np.outer(np.ones(20), np.cos(_pv))
        ax.plot_wireframe(xs, ys, zs, color="#4a9de0", alpha=0.12,
                          linewidth=0.4, rstride=2, cstride=2)

    # ── animated sphere dots ──────────────────────────────────────────────────
    sph_dots = []
    for spos in sph_pos_rest:
        sx, sy, sz = float(spos[0]), float(spos[1]), float(spos[2])
        dot, = ax.plot([sx], [sy], [sz], "o",
                       color="#4a9de0", markersize=max(4, int(sph_radius * 80)),
                       alpha=0.90)
        sph_dots.append(dot)

    # ── drone artists ─────────────────────────────────────────────────────────
    xs_, ys_, zs_ = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    trail, = ax.plot([], [], [], linewidth=2.0, color="steelblue", alpha=0.75)
    dot,   = ax.plot([], [], [], "o", markersize=8, color="steelblue")
    ax.plot([xs_[0]], [ys_[0]], [zs_[0]], "o", markersize=7, color="green")
    step_text = ax.text2D(0.02, 0.95, "", transform=ax.transAxes,
                          fontsize=8, verticalalignment="top")

    # ── animation ─────────────────────────────────────────────────────────────
    def _cyl_snap(t):
        if not has_cyl_traj:
            return None
        return min(t, cyl_xy_traj.shape[0] - 1)

    def _sph_snap(t):
        if not has_sph_traj:
            return None
        return min(t, sph_xyz_traj.shape[0] - 1)

    def update(t):
        trail.set_data_3d(xs_[:t + 1], ys_[:t + 1], zs_[:t + 1])
        dot.set_data_3d([xs_[t]], [ys_[t]], [zs_[t]])
        step_text.set_text(f"step {t}/{T - 1}")

        ci = _cyl_snap(t)
        if ci is not None:
            for orig_i, bot, top, stem in dyn_cyl_artists:
                ncx, ncy = cyl_xy_traj[ci, orig_i]
                ncx, ncy = float(ncx), float(ncy)
                xc = ncx + 0.06 * np.cos(_th)
                yc = ncy + 0.06 * np.sin(_th)
                bot.set_data_3d(xc, yc, np.zeros(32))
                top.set_data_3d(xc, yc, np.ones(32))
                stem.set_data_3d([ncx, ncx], [ncy, ncy], [0.0, 1.0])

        si = _sph_snap(t)
        if si is not None:
            for k, dot_k in enumerate(sph_dots):
                sx = float(sph_xyz_traj[si, k, 0])
                sy = float(sph_xyz_traj[si, k, 1])
                sz = float(sph_xyz_traj[si, k, 2])
                dot_k.set_data_3d([sx], [sy], [sz])

    anim = FuncAnimation(fig, update, frames=T, interval=1000 // fps, blit=False)
    anim.save(out_path, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    print(f"[GIF] saved: {out_path}  ({T} frames @ {fps}fps, plane=xyz)")
    return out_path


def make_gif(
    npz_path: str,
    fps: int = 10,
    out_path: Optional[str] = None,
    figsize: tuple = (9, 4),
    dpi: int = 100,
    plane: str = "xy",     # "xy" (top-down), "xz" (side/altitude), "xyz" (3D)
) -> str:
    if out_path is None:
        out_path = npz_path.replace(".npz", f"_anim_{plane}.gif")

    if plane == "xyz":
        return _make_gif_3d(npz_path, fps, out_path, figsize, dpi)

    # ── load trajectory data ─────────────────────────────────────────────────
    data = np.load(npz_path, allow_pickle=True)

    xyz         = data["xyz"]           # (T, 3)
    boxes       = data["boxes"]         # (N, 2)
    cylinders   = data["cylinders"]     # (M, 2)
    dynamic     = bool(data.get("dynamic_obstacles", False))
    cyl_xy_traj = data.get("cyl_xy_traj", None)   # (T_snap, M, 2)
    dyn_indices = data.get("dynamic_cyl_indices", np.array([], dtype=int))

    # floating sphere data
    has_spheres   = bool(data.get("floating_spheres", False))
    sph_pos_rest  = np.array(data["sphere_positions"], dtype=np.float32) \
                    if "sphere_positions" in data else np.zeros((0, 3), np.float32)
    sph_radius    = float(data["sphere_radius"]) if "sphere_radius" in data else 0.10
    sph_xyz_traj  = np.array(data["sph_xyz_traj"], dtype=np.float32) \
                    if "sph_xyz_traj" in data else np.zeros((0, 0, 3), np.float32)

    # candidate rollouts (chosen + unchosen), one snapshot per env step.
    # snap i corresponds to xyz[i+1] (xyz[0] is the pre-loop start position).
    cand_traj_xy = data.get("cand_traj_xy", None)   # (N_snap, K, H+1, 2), absolute xy
    snap_chosen  = data.get("snap_chosen",  None)   # (N_snap,)
    has_candidates = (
        plane == "xy"
        and cand_traj_xy is not None
        and np.asarray(cand_traj_xy).size > 0
    )
    if has_candidates:
        cand_traj_xy = np.asarray(cand_traj_xy)
        snap_chosen  = np.asarray(snap_chosen)
        # persist each candidate fan on screen for its own horizon length (H frames)
        # instead of just the 1 frame it was sampled on, so you can see whether the
        # drone's actual path over the next H steps tracked the chosen candidate.
        persist_gens = max(cand_traj_xy.shape[2] - 1, 1)   # H = (H+1 waypoints) - 1

    # ── halfspaces: only plot what was active during the recorded run ─────────
    if "halfspaces" in data and len(data["halfspaces"]) > 0:
        hs_pts   = data["halfspaces"].tolist()
        hs_sides = data["hs_sides"].tolist() if "hs_sides" in data \
                   else ["below"] * len(hs_pts)
        halfspaces = [[pts[0], pts[1], side]
                      for pts, side in zip(hs_pts, hs_sides)]
    else:
        halfspaces = []

    T = len(xyz)

    has_cyl_traj = (
        dynamic
        and cyl_xy_traj is not None
        and len(cyl_xy_traj) > 0
        and cyl_xy_traj.shape[0] > 0
    )
    has_sph_traj = (
        has_spheres
        and sph_xyz_traj.shape[0] > 0
        and sph_xyz_traj.shape[1] > 0
    )

    # ── axis helpers: pick which columns of xyz to use ───────────────────────
    if plane == "xz":
        h_idx, v_idx = 0, 2
        h_label, v_label = "x (m)", "z (m)"
        h_lim = (-0.2, 4.2)
        v_lim = (-0.05, 1.3)
    else:                          # "xy" — default top-down
        h_idx, v_idx = 0, 1
        h_label, v_label = "x (m)", "y (m)"
        h_lim = (-0.2, 4.2)
        v_lim = (-1.2, 1.2)

    corridor_end = 4.1
    wall_kw = dict(linewidth=1.5, edgecolor="#555555", facecolor="#cccccc",
                   alpha=0.50, zorder=1)

    # ── figure & axes ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_xlim(*h_lim)
    ax.set_ylim(*v_lim)
    ax.set_aspect("equal")
    ax.set_xlabel(h_label)
    ax.set_ylabel(v_label)
    ax.grid(True, alpha=0.25, linewidth=0.6)

    # ── static scene geometry ─────────────────────────────────────────────────
    if plane == "xy":
        ax.add_patch(Rectangle((0.0,  1.00), corridor_end, 0.10, **wall_kw))
        ax.add_patch(Rectangle((0.0, -1.10), corridor_end, 0.10, **wall_kw))
        ax.add_patch(Rectangle((corridor_end, -1.10), 0.10, 2.20, **wall_kw))
        ax.plot([4.0, 4.0], [-1.0, 1.0], color="#00aa00", linewidth=2.2,
                linestyle="-", alpha=0.85, zorder=5)
        if halfspaces:
            _draw_halfspaces_xy(ax, halfspaces, xlim=h_lim, ylim=v_lim)
    else:  # xz — side view: draw floor, ceiling, goal post
        ax.axhline(0.0, color="#555555", linewidth=1.2, linestyle="--", alpha=0.6)
        ax.axhline(1.0, color="#555555", linewidth=1.2, linestyle="--", alpha=0.6)
        ax.plot([4.0, 4.0], [0.0, 1.0], color="#00aa00", linewidth=2.2,
                linestyle="-", alpha=0.85, zorder=5)

    # ── static box obstacles ─────────────────────────────────────────────────
    for b in boxes:
        bx, by = float(b[0]), float(b[1])
        if plane == "xy":
            ax.add_patch(Rectangle(
                (bx - 0.10, by - 0.10), 0.20, 0.20,
                linewidth=0.8, edgecolor="black", facecolor="#cc3030",
                alpha=0.35, zorder=2,
            ))
        else:  # xz: boxes span z=0..1 (floor to ceiling)
            ax.add_patch(Rectangle(
                (bx - 0.10, 0.0), 0.20, 1.0,
                linewidth=0.8, edgecolor="black", facecolor="#cc3030",
                alpha=0.22, zorder=2,
            ))

    # ── static cylinder obstacles ─────────────────────────────────────────────
    dyn_set = set(int(i) for i in dyn_indices)
    for i, c in enumerate(cylinders):
        if i not in dyn_set:
            cx, cy = float(c[0]), float(c[1])
            if plane == "xy":
                ax.add_patch(Circle(
                    (cx, cy), 0.06,
                    linewidth=0.8, edgecolor="black", facecolor="#e87020",
                    alpha=0.40, zorder=2,
                ))
            else:  # xz: cylinders are full-height columns
                ax.add_patch(Rectangle(
                    (cx - 0.06, 0.0), 0.12, 1.0,
                    linewidth=0.8, edgecolor="black", facecolor="#e87020",
                    alpha=0.25, zorder=2,
                ))

    # ── dynamic cylinder patches ─────────────────────────────────────────────
    dyn_cyl_patches = []
    for i in sorted(dyn_set):
        cx, cy = float(cylinders[i][0]), float(cylinders[i][1])
        if plane == "xy":
            p = Circle(
                (cx, cy), 0.06,
                facecolor="#e87020", edgecolor="darkorange",
                linewidth=1.0, alpha=0.85, zorder=4,
            )
        else:  # xz: dynamic cylinder shown as a tall rectangle
            p = Rectangle(
                (cx - 0.06, 0.0), 0.12, 1.0,
                facecolor="#e87020", edgecolor="darkorange",
                linewidth=1.0, alpha=0.55, zorder=4,
            )
        ax.add_patch(p)
        dyn_cyl_patches.append((i, p))

    # ── dynamic cylinder direction arrows (instantaneous velocity) ───────────
    dyn_cyl_arrows = []
    if plane == "xy":
        for i in sorted(dyn_set):
            cx, cy = float(cylinders[i][0]), float(cylinders[i][1])
            arrow = FancyArrowPatch(
                (cx, cy), (cx, cy), arrowstyle="-|>", mutation_scale=10,
                color="darkorange", linewidth=1.5, alpha=0.9, zorder=6,
            )
            ax.add_patch(arrow)
            dyn_cyl_arrows.append((i, arrow))

    # ── candidate rollouts: unchosen (faint) + chosen (bold), fading with age ────
    # gen_cand_lines[g] / gen_chosen_lines[g] hold the fan sampled `g` steps ago;
    # g=0 is the freshest snapshot, g=persist_gens-1 is about to expire.
    if has_candidates:
        K = cand_traj_xy.shape[1]
        gen_cand_lines, gen_chosen_lines = [], []
        for g in range(persist_gens):
            age_frac = g / max(persist_gens - 1, 1)          # 0 (fresh) .. 1 (oldest)
            alpha_unchosen = 0.30 * (1.0 - age_frac) + 0.02
            alpha_chosen   = 0.90 * (1.0 - age_frac) + 0.05
            lines_k = [
                ax.plot([], [], linewidth=1.0, color="#999999", alpha=alpha_unchosen, zorder=3)[0]
                for _ in range(K)
            ]
            chosen_l, = ax.plot([], [], linewidth=2.0, color="#d62728",
                                alpha=alpha_chosen, zorder=5,
                                label="chosen candidate" if g == 0 else None)
            gen_cand_lines.append(lines_k)
            gen_chosen_lines.append(chosen_l)
        cand_lines = [line for gen in gen_cand_lines for line in gen] + gen_chosen_lines
    else:
        gen_cand_lines, gen_chosen_lines = [], []
        cand_lines = []

    # ── static sphere footprints ──────────────────────────────────────────────
    for k, spos in enumerate(sph_pos_rest):
        sx, sy, sz = float(spos[0]), float(spos[1]), float(spos[2])
        if plane == "xy":
            ax.add_patch(Circle(
                (sx, sy), sph_radius,
                linewidth=0.7, edgecolor="#1f6fbf", facecolor="#4a9de0",
                alpha=0.22, zorder=1, linestyle=":",
            ))
        else:  # xz
            ax.add_patch(Circle(
                (sx, sz), sph_radius,
                linewidth=0.7, edgecolor="#1f6fbf", facecolor="#4a9de0",
                alpha=0.22, zorder=1, linestyle=":",
            ))

    # ── dynamic sphere patches (repositioned each frame) ─────────────────────
    sph_patches = []
    for k, spos in enumerate(sph_pos_rest):
        sx, sy, sz = float(spos[0]), float(spos[1]), float(spos[2])
        center = (sx, sy) if plane == "xy" else (sx, sz)
        p = Circle(
            center, sph_radius,
            facecolor="#4a9de0", edgecolor="#1f6fbf",
            linewidth=1.2, alpha=0.80, zorder=4,
        )
        ax.add_patch(p)
        sph_patches.append((k, p))

    # ── drone artists ─────────────────────────────────────────────────────────
    h0 = xyz[:, h_idx]
    v0 = xyz[:, v_idx]
    trail_line, = ax.plot([], [], linewidth=2.0, color="steelblue", alpha=0.7, zorder=6)
    drone_dot,  = ax.plot([], [], "o", markersize=8, color="steelblue", zorder=7)
    ax.plot(h0[0], v0[0], "o", markersize=7, color="green", zorder=8)
    step_text = ax.text(0.02, 0.95, "", transform=ax.transAxes,
                        fontsize=8, verticalalignment="top")

    fig.tight_layout()

    # ── animation helpers ─────────────────────────────────────────────────────
    def _cyl_snap_idx(t):
        if not has_cyl_traj:
            return None
        return min(t, cyl_xy_traj.shape[0] - 1)

    def _sph_snap_idx(t):
        if not has_sph_traj:
            return None
        return min(t, sph_xyz_traj.shape[0] - 1)

    all_patches = ([p for _, p in dyn_cyl_patches] + [p for _, p in sph_patches]
                   + [a for _, a in dyn_cyl_arrows] + cand_lines)

    def init():
        trail_line.set_data([], [])
        drone_dot.set_data([], [])
        step_text.set_text("")
        return [trail_line, drone_dot, step_text] + all_patches

    def update(t):
        trail_line.set_data(h0[:t + 1], v0[:t + 1])
        drone_dot.set_data([h0[t]], [v0[t]])
        step_text.set_text(f"step {t}/{T - 1}")

        # move dynamic cylinders
        ci = _cyl_snap_idx(t)
        if ci is not None:
            for orig_i, patch in dyn_cyl_patches:
                cx, cy = cyl_xy_traj[ci, orig_i]
                if plane == "xy":
                    patch.center = (float(cx), float(cy))
                else:
                    patch.set_x(float(cx) - 0.06)

            # direction arrow: finite-difference velocity vs. previous snapshot
            if plane == "xy" and ci > 0:
                for orig_i, arrow in dyn_cyl_arrows:
                    cx, cy = (float(v) for v in cyl_xy_traj[ci, orig_i])
                    px, py = (float(v) for v in cyl_xy_traj[ci - 1, orig_i])
                    vx, vy = cx - px, cy - py
                    speed = (vx ** 2 + vy ** 2) ** 0.5
                    if speed > 1e-6:
                        ux, uy = vx / speed, vy / speed
                        arrow_len = 0.15
                        arrow.set_positions((cx, cy), (cx + ux * arrow_len, cy + uy * arrow_len))

        # move floating spheres
        si = _sph_snap_idx(t)
        if si is not None:
            for k, patch in sph_patches:
                sx = float(sph_xyz_traj[si, k, 0])
                sy = float(sph_xyz_traj[si, k, 1])
                sz = float(sph_xyz_traj[si, k, 2])
                patch.center = (sx, sy) if plane == "xy" else (sx, sz)

        # candidate rollouts: snap i -> xyz[i+1], so frame t's freshest snap is t-1.
        # each generation g back in time (t-1-g) stays visible until it ages out
        # of the persist_gens window, fading as it does (see artist alpha setup).
        if has_candidates:
            si_latest = t - 1
            for g in range(persist_gens):
                si_cand = si_latest - g
                if 0 <= si_cand < cand_traj_xy.shape[0]:
                    rollouts = cand_traj_xy[si_cand]          # (K, H+1, 2)
                    chosen_k = int(snap_chosen[si_cand])
                    for k, line in enumerate(gen_cand_lines[g]):
                        if k == chosen_k:
                            line.set_data([], [])             # drawn separately, bold
                        else:
                            line.set_data(rollouts[k, :, 0], rollouts[k, :, 1])
                    gen_chosen_lines[g].set_data(rollouts[chosen_k, :, 0], rollouts[chosen_k, :, 1])
                else:
                    for line in gen_cand_lines[g]:
                        line.set_data([], [])
                    gen_chosen_lines[g].set_data([], [])

        return [trail_line, drone_dot, step_text] + all_patches

    # ── render & save ─────────────────────────────────────────────────────────
    anim = FuncAnimation(fig, update, frames=T, init_func=init,
                         interval=1000 // fps, blit=True)
    anim.save(out_path, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    print(f"[GIF] saved: {out_path}  "
          f"({figsize[0]*dpi:.0f}x{figsize[1]*dpi:.0f}px @ {fps}fps, plane={plane})")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Animate drone trajectory from a saved .npz file.")
    parser.add_argument("npz", nargs="+",
                        help=".npz file(s) or directories containing .npz files")
    parser.add_argument("--fps",     type=int,   default=20)
    parser.add_argument("--dpi",     type=int,   default=100)
    parser.add_argument("--figsize", type=float, nargs=2, default=[9, 4],
                        metavar=("W", "H"))
    parser.add_argument("--out",     type=str,   default=None,
                        help="Output GIF path (ignored for multiple files)")
    parser.add_argument("--plane",   type=str,   default="xy",
                        choices=["xy", "xz", "xyz"],
                        help="View: 'xy' top-down (default), 'xz' side/altitude, "
                             "'xyz' full 3D")
    args = parser.parse_args()

    resolved = []
    for p in args.npz:
        if os.path.isdir(p):
            found = sorted(glob.glob(os.path.join(p, "*.npz")))
            if not found:
                print(f"[WARN] no .npz files found in {p}")
            resolved.extend(found)
        else:
            resolved.append(p)

    if args.out and len(resolved) > 1:
        print("[WARN] --out ignored for multiple files")
        args.out = None

    for npz_path in resolved:
        make_gif(npz_path, fps=args.fps, out_path=args.out,
                 figsize=tuple(args.figsize), dpi=args.dpi, plane=args.plane)


if __name__ == "__main__":
    main()
