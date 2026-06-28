"""
make_dataset_video.py
─────────────────────────────────────────────────────────────────────────────
Creates a presentation-quality video from the Crazyflie dataset showing:
  - Left panel : FPV camera feed (drone's eye view)
  - Right panel: top-down XY map with the drone's live position + trajectory tail

Usage
─────
    python make_dataset_video.py \
        --zarr   /path/to/env_000.zarr \
        --out    dataset_fpv.mp4 \
        --fps    30 \
        --ep     0          # episode index (0-based); omit to concatenate all

Optional flags
──────────────
    --ep     INT        single episode to render (default: all)
    --fps    INT        output fps (default: 30)
    --speed  FLOAT      playback multiplier, e.g. 2.0 = 2× faster (default: 1.0)
    --tail   INT        how many past positions to draw as trajectory tail (default: 60)
    --out    PATH       output file path (default: dataset_fpv.mp4)
"""

import argparse
import os
import zarr
import numpy as np
import cv2
import matplotlib
from isaac.dataset.sim_path import sim_framework_path
matplotlib.use("Agg")          # headless
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle, Circle
from io import BytesIO
from PIL import Image

# ─────────────────────────────────────────────────────────────────────────────
# Corridor / obstacle layout  (must match your Isaac Sim scene)
# ─────────────────────────────────────────────────────────────────────────────
WALL_HEIGHT = 1.0
BOXES = [
    # (3.5,  0.2, WALL_HEIGHT / 2.0),
    # (2.2, -0.2, WALL_HEIGHT / 2.0),
    # (1.6,  0.2, WALL_HEIGHT / 2.0),
    # (3.7, -0.3, WALL_HEIGHT / 2.0),
    # (2.8, -0.2, WALL_HEIGHT / 2.0),
    # (1.0, -0.3, WALL_HEIGHT / 2.0),
]
CYLINDERS = [
    # (1.4, -0.2, WALL_HEIGHT / 2.0),
    # (2.8,  0.3, WALL_HEIGHT / 2.0),
    # (1.2,  0.4, WALL_HEIGHT / 2.0),
    # (3.5,  0.4, WALL_HEIGHT / 2.0),
    # (2.0,  0.4, WALL_HEIGHT / 2.0),
    # (2.5,  0.3, WALL_HEIGHT / 2.0),
]
BOX_SIZE   = 0.20
CYL_RADIUS = 0.06

# ─────────────────────────────────────────────────────────────────────────────
# Video layout constants
# ─────────────────────────────────────────────────────────────────────────────
FPV_SIZE   = 480          # FPV panel: FPV_SIZE × FPV_SIZE px
MAP_W      = 640          # map panel width
MAP_H      = 480          # map panel height
BORDER     = 8            # dark border between panels
HEADER_H   = 48           # top header bar height

VIDEO_W = FPV_SIZE + BORDER + MAP_W
VIDEO_H = HEADER_H + max(FPV_SIZE, MAP_H)

BG_COLOR      = (15,  15,  20)   # near-black background
HEADER_COLOR  = (25,  25,  35)
ACCENT_COLOR  = (0,  200, 255)   # cyan accent
TEXT_COLOR    = (220, 220, 220)
TAIL_COLOR    = (0,  200, 255)
DOT_COLOR     = (255, 80,  80)   # current position dot


def draw_map_frame(ax, pos_history, pos_now, episode_len):
    """Render the top-down corridor map into a matplotlib axes."""
    ax.set_facecolor("#0d0d14")
    ax.set_xlim(-0.3, 4.3)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect("equal")
    ax.tick_params(colors="#555")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")

    # Corridor walls
    for y_wall in [1.0, -1.0]:
        ax.axhline(y_wall, color="#444", linewidth=2.5, linestyle="--", alpha=0.6)

    # Boxes
    for x, y, _ in BOXES:
        ax.add_patch(Rectangle(
            (x - BOX_SIZE/2, y - BOX_SIZE/2), BOX_SIZE, BOX_SIZE,
            facecolor="#c0392b", edgecolor="#e74c3c", alpha=0.75, linewidth=1.2,
        ))
    # Cylinders
    for x, y, _ in CYLINDERS:
        ax.add_patch(Circle(
            (x, y), CYL_RADIUS,
            facecolor="#e67e22", edgecolor="#f39c12", alpha=0.80, linewidth=1.2,
        ))

    # Goal zone
    ax.axvline(4.0, color="#2ecc71", linewidth=2.0, linestyle=":", alpha=0.8,
               label="goal line")

    # Trajectory tail  (matplotlib needs 0-1 float RGB)
    TAIL_COLOR_MPL = (0.0, 0.78, 1.0)   # cyan  #00C8FF
    if len(pos_history) > 1:
        hist = np.array(pos_history)
        n = len(hist)
        for i in range(1, n):
            alpha = 0.15 + 0.75 * (i / n)
            ax.plot(hist[i-1:i+1, 0], hist[i-1:i+1, 1],
                    color=TAIL_COLOR_MPL,
                    linewidth=2.0, alpha=alpha,
                    solid_capstyle="round")

    # Current position
    if pos_now is not None:
        ax.scatter(*pos_now[:2], s=120, color="#ff5050",
                   zorder=10, edgecolors="white", linewidths=1.2)
        # small heading arrow (just forward in x)
        ax.annotate("", xy=(pos_now[0]+0.12, pos_now[1]),
                    xytext=(pos_now[0], pos_now[1]),
                    arrowprops=dict(arrowstyle="->", color="white", lw=1.5))

    # Start marker
    if pos_history:
        p0 = pos_history[0]
        ax.scatter(p0[0], p0[1], s=80, marker="o", color="#2ecc71",
                   zorder=9, edgecolors="white", linewidths=1.0)

    # Progress bar along x-axis
    frac = min(1.0, pos_now[0] / 4.0) if pos_now is not None else 0.0
    ax.annotate(f"Progress: {frac*100:.0f}%",
                xy=(0.02, 0.05), xycoords="axes fraction",
                color="#aaa", fontsize=8)

    ax.set_xlabel("x (m)", color="#888", fontsize=9)
    ax.set_ylabel("y (m)", color="#888", fontsize=9)
    ax.set_title("Top-down map", color="#ccc", fontsize=10, pad=4)


def fig_to_rgb(fig, w, h):
    """Render matplotlib figure to an (H, W, 3) uint8 numpy array."""
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100,
                facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0)
    buf.seek(0)
    img = np.array(Image.open(buf).convert("RGB"))
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    return img


def build_frame(fpv_rgb, map_rgb, step, episode_len, ep_idx, n_episodes, pos_now):
    """Composite FPV + map + HUD into one VIDEO_W × VIDEO_H BGR frame."""
    frame = np.full((VIDEO_H, VIDEO_W, 3), BG_COLOR[::-1], dtype=np.uint8)

    # ── Header bar ──────────────────────────────────────────────────────────
    cv2.rectangle(frame, (0, 0), (VIDEO_W, HEADER_H), HEADER_COLOR[::-1], -1)
    title = "Crazyflie FPV Dataset  |  Safe 3D Drone Navigation"
    cv2.putText(frame, title, (12, 30),
                cv2.FONT_HERSHEY_DUPLEX, 0.65, ACCENT_COLOR[::-1], 1, cv2.LINE_AA)
    ep_text = f"Ep {ep_idx+1}/{n_episodes}   step {step+1}/{episode_len}"
    cv2.putText(frame, ep_text, (VIDEO_W - 280, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, TEXT_COLOR[::-1], 1, cv2.LINE_AA)

    # ── FPV panel ────────────────────────────────────────────────────────────
    fpv = cv2.resize(fpv_rgb, (FPV_SIZE, FPV_SIZE))
    fpv_bgr = cv2.cvtColor(fpv, cv2.COLOR_RGB2BGR)
    frame[HEADER_H:HEADER_H+FPV_SIZE, 0:FPV_SIZE] = fpv_bgr

    # FPV label + crosshair overlay
    cx, cy = FPV_SIZE // 2, HEADER_H + FPV_SIZE // 2
    cv2.line(frame, (cx-18, cy), (cx+18, cy), ACCENT_COLOR[::-1], 1)
    cv2.line(frame, (cx, cy-18), (cx, cy+18), ACCENT_COLOR[::-1], 1)
    cv2.circle(frame, (cx, cy), 6, ACCENT_COLOR[::-1], 1)
    cv2.putText(frame, "FPV CAMERA", (6, HEADER_H + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, ACCENT_COLOR[::-1], 1, cv2.LINE_AA)

    # ── Map panel ────────────────────────────────────────────────────────────
    map_x = FPV_SIZE + BORDER
    map_bgr = cv2.cvtColor(map_rgb, cv2.COLOR_RGB2BGR)
    frame[HEADER_H:HEADER_H+MAP_H, map_x:map_x+MAP_W] = map_bgr

    # Altitude mini-bar on the map panel (right edge)
    if pos_now is not None:
        z_frac = np.clip(pos_now[2] / 1.5, 0, 1)
        bar_h  = int(MAP_H * 0.6)
        bar_x  = map_x + MAP_W - 18
        bar_y0 = HEADER_H + MAP_H - 20
        cv2.rectangle(frame, (bar_x, bar_y0 - bar_h), (bar_x+10, bar_y0),
                      (50, 50, 60), -1)
        fill_h = int(bar_h * z_frac)
        cv2.rectangle(frame, (bar_x, bar_y0 - fill_h), (bar_x+10, bar_y0),
                      ACCENT_COLOR[::-1], -1)
        cv2.putText(frame, "Z", (bar_x, bar_y0 - bar_h - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, TEXT_COLOR[::-1], 1, cv2.LINE_AA)
        cv2.putText(frame, f"{pos_now[2]:.2f}m",
                    (bar_x - 8, bar_y0 + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, TEXT_COLOR[::-1], 1, cv2.LINE_AA)

    return frame


def render_episode(g, ep_start, ep_end, ep_idx, n_episodes, tail_len,
                   writer, fps, speed_factor):
    """Render one episode and write frames to the VideoWriter."""
    states = g["states"][ep_start:ep_end+1]
    pos    = states[:, :3]
    ep_len = ep_end - ep_start + 1

    # Decide which frames to actually write (speed-up = skip frames)
    step_indices = list(range(ep_len))
    skip = max(1, int(speed_factor))
    step_indices = step_indices[::skip]

    pos_history = []

    for frame_no, t in enumerate(step_indices):
        # ── RGB from zarr ────────────────────────────────────────────────────
        rgb_frame = g["rgb"][ep_start + t]          # (H, W, 3) uint8
        if rgb_frame.dtype != np.uint8:
            rgb_frame = (rgb_frame * 255).clip(0, 255).astype(np.uint8)

        pos_now = pos[t]
        pos_history.append(pos_now.copy())
        if len(pos_history) > tail_len:
            pos_history.pop(0)

        # ── Render map ───────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(MAP_W/100, MAP_H/100),
                               facecolor="#0d0d14")
        draw_map_frame(ax, pos_history, pos_now, ep_len)
        fig.subplots_adjust(left=0.09, right=0.97, bottom=0.08, top=0.92)
        map_rgb = fig_to_rgb(fig, MAP_W, MAP_H)
        plt.close(fig)

        # ── Composite ────────────────────────────────────────────────────────
        out_frame = build_frame(
            fpv_rgb   = rgb_frame,
            map_rgb   = map_rgb,
            step      = frame_no,
            episode_len = len(step_indices),
            ep_idx    = ep_idx,
            n_episodes = n_episodes,
            pos_now   = pos_now,
        )

        writer.write(out_frame)

    # Brief black pause between episodes (0.4 s)
    pause_frames = max(1, int(fps * 0.4))
    black = np.zeros_like(out_frame)
    for _ in range(pause_frames):
        writer.write(black)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zarr",  default=sim_framework_path("isaac", "dataset", "avoiding_crazyflie", "data", "zarr", "env_000.zarr"))
    parser.add_argument("--out",   default="dataset_fpv.mp4")
    parser.add_argument("--fps",   type=int,   default=30)
    parser.add_argument("--speed", type=float, default=2.0,
                        help="Playback speed multiplier (e.g. 2.0 = 2x faster)")
    parser.add_argument("--tail",  type=int,   default=60,
                        help="Number of past positions shown as trajectory tail")
    parser.add_argument("--ep",    type=int,   default=0,
                        help="Single episode index to render (default: all)")
    parser.add_argument("--max_eps", type=int, default=None,
                        help="Max number of episodes to render (default: all)")
    args = parser.parse_args()

    g       = zarr.open_group(args.zarr, mode="r")
    term    = g["terminals"][:].astype(np.uint8)
    ends    = np.where(term == 1)[0]
    starts  = np.r_[0, ends[:-1] + 1]

    # Episode selection
    if args.ep is not None:
        ep_range = [(args.ep, starts[args.ep], ends[args.ep])]
    else:
        max_e = args.max_eps if args.max_eps else len(starts)
        ep_range = [(i, int(starts[i]), int(ends[i])) for i in range(max_e)]

    n_episodes = len(ep_range)
    print(f"Rendering {n_episodes} episode(s) → {args.out}")
    print(f"  FPS={args.fps}  speed={args.speed}x  tail={args.tail}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out, fourcc, args.fps, (VIDEO_W, VIDEO_H))

    for render_idx, (ep_idx, ep_start, ep_end) in enumerate(ep_range):
        ep_len = ep_end - ep_start + 1
        print(f"  Episode {ep_idx}  ({ep_len} steps) …", end="", flush=True)
        render_episode(
            g           = g,
            ep_start    = ep_start,
            ep_end      = ep_end,
            ep_idx      = render_idx,
            n_episodes  = n_episodes,
            tail_len    = args.tail,
            writer      = writer,
            fps         = args.fps,
            speed_factor = args.speed,
        )
        print(" done")

    writer.release()
    print(f"\nVideo saved → {args.out}  ({VIDEO_W}×{VIDEO_H})")


if __name__ == "__main__":
    main()