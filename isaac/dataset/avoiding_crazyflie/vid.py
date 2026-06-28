import argparse
import os
import zarr
import numpy as np
import cv2
from isaac.dataset.sim_path import sim_framework_path

# BGR colours
C_BG       = ( 14,  14,  20)
C_FOOTER   = ( 18,  18,  28)
C_ACCENT   = (  0, 200, 255)   # cyan
C_FILL     = (  0, 160, 200)   # progress fill (darker cyan)
C_TRACK    = ( 55,  55,  75)   # progress track
C_TEXT     = (210, 210, 210)
C_DIM      = (110, 110, 130)

FONT       = cv2.FONT_HERSHEY_DUPLEX
FONT_S     = cv2.FONT_HERSHEY_SIMPLEX

POSE_H  = 40
PROG_H  = 50
FOOTER_H = POSE_H + PROG_H


def make_layout(video_size):
    W = video_size
    H = video_size + FOOTER_H
    return dict(
        W=W, H=H,
        fpv_y0=0,
        fpv_y1=video_size,
        pose_y0=video_size,
        pose_y1=video_size + POSE_H,
        prog_y0=video_size + POSE_H,
        prog_y1=video_size + POSE_H + PROG_H,
    )


def build_frame(fpv_rgb, pos_now, step, n_steps, ep_idx, n_episodes, L):
    W, H = L["W"], L["H"]
    frame = np.full((H, W, 3), C_BG, dtype=np.uint8)

    # FPV image
    fpv_bgr = cv2.cvtColor(fpv_rgb, cv2.COLOR_RGB2BGR)
    fpv_up  = cv2.resize(fpv_bgr, (W, W), interpolation=cv2.INTER_NEAREST)
    frame[L["fpv_y0"]:L["fpv_y1"], 0:W] = fpv_up

    # Separator line
    cv2.line(frame, (0, L["fpv_y1"]), (W, L["fpv_y1"]), C_ACCENT, 1)

    # Pose row
    cv2.rectangle(frame, (0, L["pose_y0"]), (W, L["pose_y1"]), C_FOOTER, -1)
    x, y, z = float(pos_now[0]), float(pos_now[1]), float(pos_now[2])
    pose_str = f"x  {x:+.3f} m        y  {y:+.3f} m        z  {z:+.3f} m"
    cv2.putText(frame, pose_str, (14, L["pose_y0"] + 26),
                FONT_S, 0.52, C_TEXT, 1, cv2.LINE_AA)

    # Progress bar row
    cv2.rectangle(frame, (0, L["prog_y0"]), (W, L["prog_y1"]), C_FOOTER, -1)
    cv2.line(frame, (0, L["prog_y0"]), (W, L["prog_y0"]), (38, 38, 52), 1)

    margin   = 14
    bar_h_px = 10
    bar_mid  = L["prog_y0"] + PROG_H // 2
    bar_x0   = margin
    bar_x1   = W - margin - 130

    frac   = step / max(n_steps - 1, 1)
    fill_x = bar_x0 + int((bar_x1 - bar_x0) * frac)

    cv2.rectangle(frame,
                  (bar_x0, bar_mid - bar_h_px // 2),
                  (bar_x1, bar_mid + bar_h_px // 2),
                  C_TRACK, -1)
    if fill_x > bar_x0:
        cv2.rectangle(frame,
                      (bar_x0, bar_mid - bar_h_px // 2),
                      (fill_x, bar_mid + bar_h_px // 2),
                      C_FILL, -1)
    cv2.circle(frame, (fill_x, bar_mid), bar_h_px - 1, C_ACCENT, -1, cv2.LINE_AA)

    # Ep label left of bar
    cv2.putText(frame, f"Ep {ep_idx+1}/{n_episodes}",
                (bar_x0, bar_mid - bar_h_px - 4),
                FONT_S, 0.40, C_DIM, 1, cv2.LINE_AA)

    # Percentage above knob
    pct_x = max(bar_x0, fill_x - 14)
    cv2.putText(frame, f"{frac*100:.0f}%",
                (pct_x, bar_mid - bar_h_px - 4),
                FONT_S, 0.38, C_ACCENT, 1, cv2.LINE_AA)

    # Step counter right of bar
    cv2.putText(frame, f"Step {step+1}",
                (bar_x1 + 10, bar_mid - 2),
                FONT_S, 0.50, C_TEXT, 1, cv2.LINE_AA)
    cv2.putText(frame, f"/ {n_steps}",
                (bar_x1 + 10, bar_mid + 16),
                FONT_S, 0.40, C_DIM, 1, cv2.LINE_AA)

    return frame


def render_episode(g, ep_start, ep_end, ep_idx, n_episodes,
                   writer, fps, speed_factor, layout):
    states = g["states"][ep_start:ep_end + 1]
    pos    = states[:, :3]
    ep_len = ep_end - ep_start + 1

    skip         = max(1, int(speed_factor))
    step_indices = list(range(ep_len))[::skip]
    n_steps      = len(step_indices)

    last_frame = None
    for frame_no, t in enumerate(step_indices):
        rgb_raw = g["rgb"][ep_start + t]
        if rgb_raw.dtype != np.uint8:
            rgb_raw = (rgb_raw * 255).clip(0, 255).astype(np.uint8)


        out = build_frame(
            fpv_rgb    = rgb_raw,
            pos_now    = pos[t],
            step       = frame_no,
            n_steps    = n_steps,
            ep_idx     = ep_idx,
            n_episodes = n_episodes,
            L          = layout,
        )
        writer.write(out)
        last_frame = out

    # 0.5 s black pause between episodes
    if last_frame is not None:
        black = np.zeros_like(last_frame)
        for _ in range(max(1, fps // 2)):
            writer.write(black)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zarr", default=sim_framework_path("isaac", "dataset", "avoiding_crazyflie", "data", "zarr", "env_000.zarr"))
    parser.add_argument("--out",     default="dataset_fpv.mp4")
    parser.add_argument("--fps",     type=int,   default=30)
    parser.add_argument("--speed",   type=float, default=2.0)
    parser.add_argument("--ep",      type=int,   default=0)
    parser.add_argument("--max_eps", type=int,   default=None)
    parser.add_argument("--size",    type=int,   default=256)
    args = parser.parse_args()

    g      = zarr.open_group(args.zarr, mode="r")
    term   = g["terminals"][:].astype(np.uint8)
    ends   = np.where(term == 1)[0]
    starts = np.r_[0, ends[:-1] + 1]

    if args.ep is not None:
        ep_range = [(args.ep, int(starts[args.ep]), int(ends[args.ep]))]
    else:
        max_e    = args.max_eps if args.max_eps else len(starts)
        ep_range = [(i, int(starts[i]), int(ends[i])) for i in range(max_e)]

    layout     = make_layout(args.size)
    n_episodes = len(ep_range)
    W, H       = layout["W"], layout["H"]

    print(f"Rendering {n_episodes} episode(s) -> {args.out}")
    print(f"  FPS={args.fps}  speed={args.speed}x  frame={W}x{H}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out, fourcc, args.fps, (W, H))

    for render_idx, (ep_idx, ep_start, ep_end) in enumerate(ep_range):
        ep_len = ep_end - ep_start + 1
        print(f"  Episode {ep_idx}  ({ep_len} steps) ...", end="", flush=True)
        render_episode(
            g            = g,
            ep_start     = ep_start,
            ep_end       = ep_end,
            ep_idx       = render_idx,
            n_episodes   = n_episodes,
            writer       = writer,
            fps          = args.fps,
            speed_factor = args.speed,
            layout       = layout,
        )
        print(" done")

    writer.release()
    print(f"\nVideo saved -> {args.out}  ({W}x{H})")


if __name__ == "__main__":
    main()