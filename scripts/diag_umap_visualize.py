"""Visualize detect_obstacles_umap()'s detections directly on a real depth+color frame
pulled from a ROS2 bag -- draws each surviving contour's bounding box (with its measured
depth labeled) on both the raw depth image and the color image, so you can look at a frame
(or play a whole video of them) and judge for yourself whether a detection is a real object
or noise, instead of just staring at printed world/cam coordinates.

Reads the bag directly via the `rosbags` library (pure Python, no live ROS2/rclpy needed --
`pip install rosbags` if missing) rather than replaying it, so it's fast and can jump to
any frame index instantly.

Usage:
    # single frame -> two PNGs
    python scripts/diag_umap_visualize.py <bag_dir> --frame 1030 --min_pixel_count 40

    # frame range -> two MP4s (depth-viz + color), one frame of each per bag message
    python scripts/diag_umap_visualize.py <bag_dir> --frame_range 800 900 --min_pixel_count 40
    python scripts/diag_umap_visualize.py <bag_dir> --frame_range 0 -1 --min_pixel_count 8   # whole bag, noisy baseline

See [[umap_obstacle_detector_bugs]] (project memory) for the investigation this came from --
mpc=8 (the shipped default) produces 100+ spurious detections per real frame; mpc=40 was
found to filter most of that out while still catching real objects, across two different
camera resolutions (848x480 and 424x240) -- but that was checked against a handful of bags,
not empirically retuned at scale. Use this script to keep checking it against more real data.
"""
from __future__ import annotations
import argparse
import os
import sys

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from depth_obstacle_estimator import _umap_contours, _contour_to_camera_detection

# Same real RealSense calibration used by eval_crazieflieros2.py / depth_camera_live_test.py.
DEPTH_FX = DEPTH_FY = 430.64617919921875
DEPTH_CX, DEPTH_CY = 427.7652587890625, 242.2167510986328
UMAP_MAX_RANGE, UMAP_BIN_SIZE = 5.0, 100
UMAP_T_POI, UMAP_T_THO = 500.0, 1800.0
EDGE_CROP_FRAC = 0.12
BOX_COLORS = [(0, 255, 0), (0, 165, 255), (255, 0, 255), (255, 255, 0), (0, 0, 255), (255, 0, 0)]


def crop_edges(depth, frac=EDGE_CROP_FRAC):
    """Same edge-noise crop as depth_camera_live_test.py -- see EDGE_CROP_FRAC there."""
    margin = int(depth.shape[1] * frac)
    if margin <= 0:
        return depth
    depth = depth.copy()
    depth[:, :margin] = 0.0
    depth[:, -margin:] = 0.0
    return depth


def load_messages(bag_dir, depth_topic, color_topic):
    """One bag pass: returns (depth_msgs, color_msgs), each a list of (timestamp, conn, rawdata)."""
    from rosbags.highlevel import AnyReader
    from rosbags.typesys import Stores, get_typestore
    from pathlib import Path

    typestore = get_typestore(Stores.ROS2_HUMBLE)
    with AnyReader([Path(bag_dir)], default_typestore=typestore) as reader:
        depth_conns = [c for c in reader.connections if c.topic == depth_topic]
        color_conns = [c for c in reader.connections if c.topic == color_topic]
        depth_msgs = [(ts, conn, raw) for conn, ts, raw in reader.messages(connections=depth_conns)]
        color_msgs = [(ts, conn, raw) for conn, ts, raw in reader.messages(connections=color_conns)]
    return depth_msgs, color_msgs


def decode_frame(reader, depth_msgs, color_msgs, frame_idx, color_ts_arr):
    """Returns (depth (H,W) metres, color (H,W,3) RGB uint8 or None, time_delta_ms)."""
    ts_d, conn_d, raw_d = depth_msgs[frame_idx]
    msg_d = reader.deserialize(raw_d, conn_d.msgtype)
    depth = (np.frombuffer(bytes(msg_d.data), dtype=np.uint16)
             .reshape(msg_d.height, msg_d.width).astype(np.float32) * 0.001)

    color, dt_ms = None, None
    if color_msgs:
        ci = min(max(np.searchsorted(color_ts_arr, ts_d), 0), len(color_msgs) - 1)
        ts_c, conn_c, raw_c = color_msgs[ci]
        msg_c = reader.deserialize(raw_c, conn_c.msgtype)
        color = (np.frombuffer(bytes(msg_c.data), dtype=np.uint8)
                 .reshape(msg_c.height, msg_c.width, 3).copy())
        dt_ms = abs(ts_c - ts_d) / 1e6
    return depth, color, dt_ms


def render(depth_cropped, min_pixel_count, base_img, max_range, bin_size, scale_x=1.0):
    """Draws boxes for every surviving detection onto base_img (BGR, mutated in place).
    Returns (n_raw_contours, n_valid_detections)."""
    contours, areas, depth_mm, mm_per_bin, W = _umap_contours(
        depth_cropped, DEPTH_FX, bin_size, max_range, UMAP_T_POI, UMAP_T_THO, min_pixel_count
    )
    order = np.argsort(areas)[::-1]
    n_drawn = 0
    for i in order:
        x, _y, w, _h = cv2.boundingRect(contours[int(i)])
        det = _contour_to_camera_detection(contours[int(i)], depth_mm, DEPTH_FX, DEPTH_FY, DEPTH_CX, DEPTH_CY, mm_per_bin, W)
        if det is None:
            continue
        pos_cam, half_w, _half_h = det
        col = BOX_COLORS[n_drawn % len(BOX_COLORS)]
        x0, x1 = int(x * scale_x), int((x + w) * scale_x)
        cv2.rectangle(base_img, (x0, 0), (x1, base_img.shape[0] - 1), col, 2)
        cv2.putText(base_img, f"z={pos_cam[2]:.2f}m r={half_w:.2f}",
                    (x0, 15 + 14 * n_drawn), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)
        n_drawn += 1
    return len(contours), n_drawn


def make_depth_vis(depth_cropped, max_range):
    """near=bright, far=dark (matches how a person naturally reads distance); invalid/
    cropped-out pixels shown black, not 'near'."""
    vis = np.clip(depth_cropped / max_range, 0, 1)
    vis = (255 * (1 - vis)).astype(np.uint8)
    vis[depth_cropped <= 0] = 0
    return cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("bag_dir", help="path to a rosbag2 directory (containing metadata.yaml + .db3)")
    p.add_argument("--frame", type=int, default=None, help="single depth message index to visualize (0-based) -> PNGs")
    p.add_argument("--frame_range", type=int, nargs=2, metavar=("START", "END"), default=None,
                    help="depth message index range [START, END) -> MP4s. END=-1 means to the end of the bag.")
    p.add_argument("--fps", type=float, default=10.0, help="video output frame rate (video mode only)")
    p.add_argument("--min_pixel_count", type=int, default=50, help="threshold to test (default 8, the shipped value)")
    p.add_argument("--max_range", type=float, default=UMAP_MAX_RANGE, help="depth range (m) covered by the U-map histogram")
    p.add_argument("--bin_size", type=int, default=UMAP_BIN_SIZE, help="number of depth bins across --max_range")
    p.add_argument("--depth_topic", type=str, default="/camera/camera/depth/image_rect_raw")
    p.add_argument("--color_topic", type=str, default="/camera/camera/color/image_raw")
    p.add_argument("--out_prefix", type=str, default=None, help="output filename prefix (default: derived from bag_dir)")
    args = p.parse_args()

    if (args.frame is None) == (args.frame_range is None):
        p.error("pass exactly one of --frame (single PNG pair) or --frame_range START END (video)")

    from rosbags.highlevel import AnyReader
    from rosbags.typesys import Stores, get_typestore
    from pathlib import Path

    depth_msgs, color_msgs = load_messages(args.bag_dir, args.depth_topic, args.color_topic)
    color_ts_arr = np.array([c[0] for c in color_msgs]) if color_msgs else np.array([])
    prefix = args.out_prefix or os.path.basename(os.path.normpath(args.bag_dir))

    typestore = get_typestore(Stores.ROS2_HUMBLE)
    with AnyReader([Path(args.bag_dir)], default_typestore=typestore) as reader:
        if args.frame is not None:
            if not (0 <= args.frame < len(depth_msgs)):
                p.error(f"--frame {args.frame} out of range: bag has {len(depth_msgs)} depth messages")
            _run_single(reader, depth_msgs, color_msgs, color_ts_arr, args, prefix)
        else:
            start, end = args.frame_range
            end = len(depth_msgs) if end < 0 else end
            if not (0 <= start < end <= len(depth_msgs)):
                p.error(f"--frame_range {start} {end} invalid: bag has {len(depth_msgs)} depth messages")
            _run_video(reader, depth_msgs, color_msgs, color_ts_arr, args, prefix, start, end)


def _run_single(reader, depth_msgs, color_msgs, color_ts_arr, args, prefix):
    depth, color, dt_ms = decode_frame(reader, depth_msgs, color_msgs, args.frame, color_ts_arr)
    if dt_ms is not None:
        print(f"depth/color time delta: {dt_ms:.1f}ms")
    depth_cropped = crop_edges(depth)
    suffix = f"frame{args.frame}_mpc{args.min_pixel_count}_mr{args.max_range:g}_bs{args.bin_size}"

    depth_vis = make_depth_vis(depth_cropped, args.max_range)
    n_contours, n_drawn = render(depth_cropped, args.min_pixel_count, depth_vis, args.max_range, args.bin_size)
    print(f"min_pixel_count={args.min_pixel_count} max_range={args.max_range} bin_size={args.bin_size}: "
          f"{n_contours} raw contours, {n_drawn} valid detections")
    depth_vis = cv2.resize(depth_vis, (depth_vis.shape[1] * 2, depth_vis.shape[0] * 2), interpolation=cv2.INTER_NEAREST)
    depth_path = f"{prefix}_{suffix}_depth.png"
    cv2.imwrite(depth_path, depth_vis)
    print(f"saved {depth_path}")

    if color is not None:
        color_bgr = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
        scale_x = color.shape[1] / depth.shape[1]
        render(depth_cropped, args.min_pixel_count, color_bgr, args.max_range, args.bin_size, scale_x=scale_x)
        color_path = f"{prefix}_{suffix}_color.png"
        cv2.imwrite(color_path, color_bgr)
        print(f"saved {color_path}  (NOTE: box x-position approximated by scaling depth pixel "
              f"coords to color resolution -- depth/color sensors have a physical baseline "
              f"offset, so this has some parallax error, worse at close range)")
    else:
        print(f"no messages on --color_topic={args.color_topic!r}, skipping color overlay")


def _run_video(reader, depth_msgs, color_msgs, color_ts_arr, args, prefix, start, end):
    suffix = f"frames{start}-{end}_mpc{args.min_pixel_count}_mr{args.max_range:g}_bs{args.bin_size}"
    depth_writer = color_writer = None
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    n_frames = end - start
    total_contours = total_drawn = 0

    for k, idx in enumerate(range(start, end)):
        depth, color, _dt_ms = decode_frame(reader, depth_msgs, color_msgs, idx, color_ts_arr)
        depth_cropped = crop_edges(depth)

        depth_vis = make_depth_vis(depth_cropped, args.max_range)
        n_contours, n_drawn = render(depth_cropped, args.min_pixel_count, depth_vis, args.max_range, args.bin_size)
        total_contours += n_contours
        total_drawn += n_drawn
        depth_vis = cv2.resize(depth_vis, (depth_vis.shape[1] * 2, depth_vis.shape[0] * 2), interpolation=cv2.INTER_NEAREST)
        if depth_writer is None:
            depth_path = f"{prefix}_{suffix}_depth.mp4"
            depth_writer = cv2.VideoWriter(depth_path, fourcc, args.fps, (depth_vis.shape[1], depth_vis.shape[0]))
        depth_writer.write(depth_vis)

        if color is not None:
            color_bgr = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
            scale_x = color.shape[1] / depth.shape[1]
            render(depth_cropped, args.min_pixel_count, color_bgr, args.max_range, args.bin_size, scale_x=scale_x)
            if color_writer is None:
                color_path = f"{prefix}_{suffix}_color.mp4"
                color_writer = cv2.VideoWriter(color_path, fourcc, args.fps, (color_bgr.shape[1], color_bgr.shape[0]))
            color_writer.write(color_bgr)

        if (k + 1) % 50 == 0 or k + 1 == n_frames:
            print(f"  {k + 1}/{n_frames} frames rendered", flush=True)

    if depth_writer is not None:
        depth_writer.release()
        print(f"saved {prefix}_{suffix}_depth.mp4")
    if color_writer is not None:
        color_writer.release()
        print(f"saved {prefix}_{suffix}_color.mp4  (NOTE: box x-position approximated by scaling "
              f"depth pixel coords to color resolution -- has some parallax error, worse at close range)")
    print(f"min_pixel_count={args.min_pixel_count} max_range={args.max_range} bin_size={args.bin_size}: "
          f"avg {total_contours / n_frames:.1f} raw contours/frame, avg {total_drawn / n_frames:.1f} valid detections/frame")


if __name__ == "__main__":
    main()
