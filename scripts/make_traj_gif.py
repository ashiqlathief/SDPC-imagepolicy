"""
make_traj_gif.py  —  animate drone + dynamic obstacle motion from a saved .npz trajectory.

Usage:
    python scripts/make_traj_gif.py path/to/traj_*.npz
    python scripts/make_traj_gif.py trajectories/          # whole directory
    python scripts/make_traj_gif.py traj.npz --fps 15 --dpi 120 --figsize 12 5 --out run.gif
"""
import argparse
import glob
import os
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle, Rectangle
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D


def make_gif(
    npz_path: str,
    fps: int = 10,
    out_path: Optional[str] = None,
    figsize: tuple = (9, 4),   # figure size in inches — controls pixel size together with dpi
    dpi: int = 100,            # dots per inch; pixel dims = figsize * dpi (e.g. 9x4 @ 100dpi → 900x400px)
) -> str:
    # ── load trajectory data ─────────────────────────────────────────────────
    data = np.load(npz_path, allow_pickle=True)

    xyz         = data["xyz"]           # (T, 3) drone positions at every action step
    boxes       = data["boxes"]         # (N, 2) box obstacle centre positions
    cylinders   = data["cylinders"]     # (M, 2) cylinder centre positions (rest positions)
    dynamic     = bool(data.get("dynamic_obstacles", False))
    cyl_xy_traj = data.get("cyl_xy_traj", None)  # (T_snap, M, 2) per-step cylinder positions
    dyn_indices = data.get("dynamic_cyl_indices", np.array([], dtype=int))

    T = len(xyz)

    # cyl_xy_traj is recorded once per action step alongside the drone position,
    # but may be shorter than T if snapshots were capped during eval
    has_cyl_traj = (
        dynamic
        and cyl_xy_traj is not None
        and len(cyl_xy_traj) > 0
        and cyl_xy_traj.shape[0] > 0
    )

    if out_path is None:
        out_path = npz_path.replace(".npz", "_anim.gif")

    # ── static scene geometry ────────────────────────────────────────────────
    corridor_end = 4.1   # x position of the end-cap wall (matches crazyflie_env_cfg.py)

    # wall rectangle style — matches plotz.py
    wall_kw = dict(linewidth=1.5, edgecolor="#555555", facecolor="#cccccc",
                   alpha=0.50, zorder=1)

    # ── figure & axes ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_xlim(-0.2, corridor_end + 0.1)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect("equal")   # keeps data units square so corridor doesn't look stretched
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(True, alpha=0.25, linewidth=0.6)

    # corridor walls: top strip, bottom strip, end cap
    ax.add_patch(Rectangle((0.0,  1.00), corridor_end, 0.10, **wall_kw))
    ax.add_patch(Rectangle((0.0, -1.10), corridor_end, 0.10, **wall_kw))
    ax.add_patch(Rectangle((corridor_end, -1.10), 0.10, 2.20, **wall_kw))

    # goal line at x=4 — viewer can see when drone crosses it
    ax.plot([4.0, 4.0], [-1.0, 1.0], color="#00aa00", linewidth=2.2,
            linestyle="-", alpha=0.85, zorder=5)
    # ax.text(4.0, 1.01, "goal", color="#00aa00", fontsize=7, ha="center", va="bottom")

    # ── static obstacles ─────────────────────────────────────────────────────
    # box obstacles — red filled rectangles (half-size 0.10 m each side)
    for b in boxes:
        bx, by = float(b[0]), float(b[1])
        ax.add_patch(Rectangle(
            (bx - 0.10, by - 0.10), 0.20, 0.20,
            linewidth=0.8, edgecolor="black", facecolor="#cc3030", alpha=0.35, zorder=2,
        ))

    # identify which cylinder indices are dynamic so static ones can be drawn differently
    dyn_set = set(int(i) for i in dyn_indices)

    # static cylinders — orange, same style as plotz.py
    for i, c in enumerate(cylinders):
        if i not in dyn_set:
            ax.add_patch(Circle(
                (float(c[0]), float(c[1])), 0.06,
                linewidth=0.8, edgecolor="black", facecolor="#e87020", alpha=0.40, zorder=2,
            ))

    # ── dynamic cylinder patches ─────────────────────────────────────────────
    # these patches are repositioned every frame to their recorded position
    dyn_cyl_patches = []
    for i in sorted(dyn_set):
        p = Circle(
            (float(cylinders[i][0]), float(cylinders[i][1])), 0.06,
            facecolor="#e87020", edgecolor="darkorange",
            linewidth=1.0, alpha=0.85, zorder=4,
        )
        ax.add_patch(p)
        dyn_cyl_patches.append((i, p))  # store original index so cyl_xy_traj can be indexed

    # ── drone artists (updated every frame) ──────────────────────────────────
    trail_line, = ax.plot([], [], linewidth=2.0, color="steelblue", alpha=0.7, zorder=6)
    drone_dot,  = ax.plot([], [], "o", markersize=8, color="steelblue", zorder=7)

    # fixed start marker
    ax.plot(xyz[0, 0], xyz[0, 1], "o", markersize=7, color="green", zorder=8)

    # # step counter shown in top-left corner of the axes
    step_text = ax.text(0.02, 0.95, "", transform=ax.transAxes,
                        fontsize=8, verticalalignment="top")

    # ── legend ───────────────────────────────────────────────────────────────
    # to remove a legend entry, just delete that line from the list below
    # legend_handles = [
    #     mpatches.Patch(facecolor="#cc3030", edgecolor="black", linewidth=0.8,
    #                    alpha=0.35, label="box obstacle"),
    #     mpatches.Patch(facecolor="#e87020", edgecolor="black", linewidth=0.8,
    #                    alpha=0.40, label="cylinder (static)"),
    #     mpatches.Patch(facecolor="#e87020", edgecolor="darkorange",
    #                    linewidth=1.0, alpha=0.85, label="cylinder (dynamic)"),
    #     Line2D([0], [0], color="steelblue", linewidth=2.0, label="drone path"),
    #     mpatches.Patch(color="green",    label="start"),
    #     Line2D([0], [0], color="#00aa00", linewidth=2.2, label="goal (x=4 m)"),
    # ]
    # ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.01, 1.0),
    #           borderaxespad=0, fontsize=7, framealpha=0.9)

    fig.tight_layout()

    # ── animation callbacks ───────────────────────────────────────────────────
    def _cyl_snap_idx(t):
        """Clamp drone step t to a valid cyl_xy_traj row index."""
        if not has_cyl_traj:
            return None
        return min(t, cyl_xy_traj.shape[0] - 1)

    def init():
        trail_line.set_data([], [])
        drone_dot.set_data([], [])
        step_text.set_text("")
        return [trail_line, drone_dot, step_text] + [p for _, p in dyn_cyl_patches]

    def update(t):
        # grow the trail up to the current step
        trail_line.set_data(xyz[:t + 1, 0], xyz[:t + 1, 1])
        drone_dot.set_data([xyz[t, 0]], [xyz[t, 1]])
        step_text.set_text(f"step {t}/{T - 1}")

        # move dynamic cylinder patches to their recorded positions
        ci = _cyl_snap_idx(t)
        if ci is not None:
            for orig_i, patch in dyn_cyl_patches:
                cx, cy = cyl_xy_traj[ci, orig_i]
                patch.center = (float(cx), float(cy))

        return [trail_line, drone_dot, step_text] + [p for _, p in dyn_cyl_patches]

    # ── render & save ─────────────────────────────────────────────────────────
    anim = FuncAnimation(fig, update, frames=T, init_func=init,
                         interval=1000 // fps, blit=True)
    anim.save(out_path, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    print(f"[GIF] saved: {out_path}  ({figsize[0]*dpi:.0f}x{figsize[1]*dpi:.0f}px @ {fps}fps)")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Animate drone trajectory from a saved .npz file.")
    parser.add_argument("npz", nargs="+",
                        help=".npz file(s) or directories containing .npz files")
    parser.add_argument("--fps",     type=int,   default=20,
                        help="Frames per second (default: 10)")
    parser.add_argument("--dpi",     type=int,   default=100,
                        help="Dots per inch — higher = larger file & sharper image (default: 100)")
    parser.add_argument("--figsize", type=float, nargs=2, default=[9, 4], metavar=("W", "H"),
                        help="Figure size in inches, e.g. --figsize 12 5 (default: 9 4)")
    parser.add_argument("--out",     type=str,   default=None,
                        help="Output GIF path (auto-named next to .npz if omitted; "
                             "ignored when multiple files are processed)")
    args = parser.parse_args()

    # expand any directory arguments into the .npz files inside them
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
        print("[WARN] --out ignored for multiple files; each GIF is saved alongside its .npz")
        args.out = None

    for npz_path in resolved:
        make_gif(npz_path, fps=args.fps, out_path=args.out,
                 figsize=tuple(args.figsize), dpi=args.dpi)


if __name__ == "__main__":
    main()
