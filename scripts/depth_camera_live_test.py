"""Standalone test harness for the depth -> obstacle pipeline in
depth_obstacle_estimator.py, isolated from IsaacLab so it can run against a
real depth camera (once we have one) or, for now, two stand-ins:

  --source synthetic   a fabricated depth frame with one obstacle at a known
                        distance, so we can check the pipeline recovers the
                        right (x,y,z) before ever touching real hardware.
  --source zarr         replays real recorded depth frames from the existing
                        Isaac dataset, at roughly real-time rate, as a smoke
                        test on real depth-shaped imagery/noise.
  --source realsense    live capture via pyrealsense2 (lazy-imported, only
                        needed once the camera is actually plugged in).

Output is in CAMERA frame, not world frame: there's no pose source when
you're holding the camera by hand, so (x,y,z) here means "relative to the
camera's own position/orientation right now," per the earlier discussion.

Reuses filter_points / cluster_points / ObstacleTracker / backproject_depth_
to_world unchanged from depth_obstacle_estimator.py -- only the frame source
and the (now-identity) camera pose are different from the sim path.
"""
from __future__ import annotations
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from depth_obstacle_estimator import (  # noqa: E402
    backproject_depth_to_world, filter_points, cluster_points, ObstacleTracker,
)

# Identity camera pose: output points are already "relative to the camera,"
# so no world transform is applied.
IDENTITY_POS = np.zeros(3, dtype=np.float32)
IDENTITY_QUAT = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # (w,x,y,z)

# Bounds crop in filter_points() is corridor-specific (sim flight envelope);
# for a free-hand/lab test there's no such box, so pass huge bounds to make
# that part of the crop a no-op and keep only its voxel/outlier filtering.
BIG = 1e3
NO_CROP_BOUNDS = (-BIG, BIG)


def process_frame(depth, fx, fy, cx, cy, args, tracker, dt):
    pts = backproject_depth_to_world(
        depth, fx, fy, cx, cy, IDENTITY_POS, IDENTITY_QUAT,
        max_range=args.max_range, stride=args.stride,
    )
    n_raw = len(pts)
    pts = filter_points(
        pts, NO_CROP_BOUNDS, NO_CROP_BOUNDS, NO_CROP_BOUNDS,
        voxel_size=args.voxel_size, outlier_radius=args.outlier_radius,
        outlier_min_neighbors=args.outlier_min_neighbors, wall_pad=0.0, z_margin=0.0,
    )
    n_filtered = len(pts)
    clusters = cluster_points(pts, eps=args.eps, min_samples=args.min_samples)
    tracks = tracker.update(clusters, dt)
    return tracks, n_raw, n_filtered


def print_tracks(tracks, n_raw, n_filtered, frame_idx, true_xyz=None):
    print(f"\n--- frame {frame_idx} (raw pts={n_raw}, filtered pts={n_filtered}, "
          f"tracks={len(tracks)}) ---")
    if not tracks:
        print("  (no obstacle detected)")
    for tid, tr in tracks:
        x, y, z = tr["centroid"]
        kind = "dynamic" if tr["is_dynamic"] else "static"
        print(f"  track {tid:>2} [{kind:7}]  (x,y,z) = ({x:+.3f}, {y:+.3f}, {z:+.3f}) m"
              f"   npts={len(tr['points']):3d}  vel={np.linalg.norm(tr['vel']):.3f} m/s")
    if true_xyz is not None:
        if tracks:
            best = min(tracks, key=lambda t: np.linalg.norm(t[1]["centroid"] - true_xyz))
            err = np.linalg.norm(best[1]["centroid"] - true_xyz)
            print(f"  [synthetic check] true=({true_xyz[0]:+.3f},{true_xyz[1]:+.3f},"
                  f"{true_xyz[2]:+.3f})  closest track error = {err:.3f} m")
        else:
            print(f"  [synthetic check] true=({true_xyz[0]:+.3f},{true_xyz[1]:+.3f},"
                  f"{true_xyz[2]:+.3f})  -- MISSED, no track detected")


# =============================================================================
# Source: synthetic
# =============================================================================

def run_synthetic(args, tracker):
    """One disc-shaped obstacle at a known distance in front of the camera,
    optionally drifting sideways across frames to also exercise the
    static/dynamic classifier. Background is a far plane (like an empty room)."""
    W = H = 96
    fx = fy = 24.0 * W / 20.955  # same intrinsics as the sim FPV camera, arbitrary otherwise
    cx, cy = W / 2.0, H / 2.0
    bg_depth = args.max_range * 0.9

    prev_t = time.time()
    for i in range(args.n_frames):
        depth = np.full((H, W), bg_depth, dtype=np.float32)
        # obstacle disc center drifts `synthetic_speed` px/frame to the right
        u0 = W / 2 + args.synthetic_offset_px + i * args.synthetic_speed
        v0 = H / 2.0
        uu, vv = np.meshgrid(np.arange(W), np.arange(H))
        mask = (uu - u0) ** 2 + (vv - v0) ** 2 <= args.synthetic_radius_px ** 2
        depth[mask] = args.synthetic_dist

        # ground truth: ray through the disc's center pixel * known distance
        ray = np.array([(u0 - cx) / fx, (v0 - cy) / fy, 1.0])
        ray /= np.linalg.norm(ray)
        true_xyz = ray * args.synthetic_dist

        now = time.time()
        dt = now - prev_t if i > 0 else 1.0 / args.rate
        prev_t = now
        tracks, n_raw, n_filtered = process_frame(depth, fx, fy, cx, cy, args, tracker, dt)
        print_tracks(tracks, n_raw, n_filtered, i, true_xyz=true_xyz)
        time.sleep(max(0.0, 1.0 / args.rate - (time.time() - now)))


# =============================================================================
# Source: zarr replay (real recorded depth frames, no live camera needed)
# =============================================================================

def run_zarr(args, tracker):
    import zarr
    z = zarr.open(args.zarr_path, mode="r")
    episode_id = z["episode_id"][:]
    idx = np.where(episode_id == args.episode)[0]
    if len(idx) == 0:
        print(f"[ERROR] no frames with episode_id == {args.episode} in {args.zarr_path}")
        return
    start, end = idx[0], idx[-1] + 1
    print(f"[INFO] replaying episode {args.episode}: frames {start}:{end} "
          f"({end - start} frames) from {args.zarr_path}")

    from depth_obstacle_estimator import camera_intrinsics, FPV_WIDTH, FPV_HEIGHT, \
        FPV_FOCAL_LENGTH, FPV_HORIZONTAL_APERTURE
    fx, fy, cx, cy = camera_intrinsics(FPV_WIDTH, FPV_HEIGHT, FPV_FOCAL_LENGTH,
                                        FPV_HORIZONTAL_APERTURE)

    prev_t = time.time()
    for i, frame_idx in enumerate(range(start, end, args.stride_frames)):
        depth = z["depth"][frame_idx]
        now = time.time()
        dt = now - prev_t if i > 0 else 1.0 / args.rate
        prev_t = now
        tracks, n_raw, n_filtered = process_frame(depth, fx, fy, cx, cy, args, tracker, dt)
        print_tracks(tracks, n_raw, n_filtered, frame_idx)
        time.sleep(max(0.0, 1.0 / args.rate - (time.time() - now)))


# =============================================================================
# Source: live RealSense (for when the camera is actually plugged in)
# =============================================================================

def run_realsense(args, tracker):
    try:
        import pyrealsense2 as rs
    except ImportError:
        print("[ERROR] pyrealsense2 not installed. Run: pip install pyrealsense2")
        return

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, args.rs_width, args.rs_height, rs.format.z16, args.rate_int)
    profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    intr = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
    print(f"[INFO] RealSense connected: {intr.width}x{intr.height}, "
          f"fx={intr.fx:.1f} fy={intr.fy:.1f} cx={intr.ppx:.1f} cy={intr.ppy:.1f}, "
          f"depth_scale={depth_scale}")

    prev_t = time.time()
    i = 0
    try:
        while True:
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            if not depth_frame:
                continue
            depth = np.asanyarray(depth_frame.get_data()).astype(np.float32) * depth_scale
            now = time.time()
            dt = now - prev_t if i > 0 else 1.0 / args.rate
            prev_t = now
            tracks, n_raw, n_filtered = process_frame(
                depth, intr.fx, intr.fy, intr.ppx, intr.ppy, args, tracker, dt)
            print_tracks(tracks, n_raw, n_filtered, i)
            i += 1
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()


# =============================================================================
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", choices=["synthetic", "zarr", "realsense"], default="synthetic")

    # shared pipeline params (same names/defaults as depth_obstacle_estimator.py's usage in eval_crazieflie1.py)
    p.add_argument("--max_range", type=float, default=5.0)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--voxel_size", type=float, default=0.05)
    p.add_argument("--outlier_radius", type=float, default=0.15)
    p.add_argument("--outlier_min_neighbors", type=int, default=4)
    p.add_argument("--eps", type=float, default=0.12)
    p.add_argument("--min_samples", type=int, default=4)
    p.add_argument("--rate", type=float, default=10.0, help="replay/print rate in Hz")

    # synthetic source
    p.add_argument("--n_frames", type=int, default=60)
    p.add_argument("--synthetic_dist", type=float, default=1.5, help="true obstacle distance, metres")
    p.add_argument("--synthetic_radius_px", type=float, default=8.0)
    p.add_argument("--synthetic_offset_px", type=float, default=-15.0)
    p.add_argument("--synthetic_speed", type=float, default=0.4, help="px/frame drift, to test dynamic classification")

    # zarr source
    p.add_argument("--zarr_path", type=str,
                    default="isaac/dataset/avoiding_crazyflie/data/zarr/env_000.zarr")
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--stride_frames", type=int, default=1, help="skip frames to slow down replay")

    # realsense source
    p.add_argument("--rs_width", type=int, default=424)
    p.add_argument("--rs_height", type=int, default=240)
    p.add_argument("--rate_int", type=int, default=30, help="RealSense stream fps (integer)")

    args = p.parse_args()
    tracker = ObstacleTracker()

    if args.source == "synthetic":
        run_synthetic(args, tracker)
    elif args.source == "zarr":
        run_zarr(args, tracker)
    elif args.source == "realsense":
        run_realsense(args, tracker)


if __name__ == "__main__":
    main()
