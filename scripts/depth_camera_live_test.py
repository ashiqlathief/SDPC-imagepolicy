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
  --source ros2          subscribes to an already-running ROS2 depth stream
                        (e.g. `ros2 launch realsense2_camera rs_launch.py
                        enable_depth:=true`) instead of opening the camera
                        directly -- lets this run alongside rviz2/other
                        consumers of the same topics, or on a different
                        machine than the one with the camera plugged in.

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
import matplotlib.pyplot as plt

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


def apply_radius_limit(points: np.ndarray, radius_limit: float | None) -> np.ndarray:
    """Spherical crop: keep only points within `radius_limit` metres of the
    camera origin (straight-line distance, not per-ray depth). Distinct from
    `max_range` in backproject_depth_to_world, which only bounds the radial
    distance *along each ray* before the point is even formed -- this instead
    trims the already-3D cloud, so it also affects points that were close in
    depth but far off-axis near the image edges."""
    if radius_limit is None or len(points) == 0:
        return points
    d = np.linalg.norm(points - IDENTITY_POS[None, :], axis=1)
    return points[d <= radius_limit]


def process_frame(depth, fx, fy, cx, cy, args, tracker, dt):
    pts_raw = backproject_depth_to_world(
        depth, fx, fy, cx, cy, IDENTITY_POS, IDENTITY_QUAT,
        max_range=args.max_range, stride=args.stride,
    )
    pts_raw = apply_radius_limit(pts_raw, args.radius_limit)
    n_raw = len(pts_raw)
    pts_filtered = filter_points(
        pts_raw, NO_CROP_BOUNDS, NO_CROP_BOUNDS, NO_CROP_BOUNDS,
        voxel_size=args.voxel_size, outlier_radius=args.outlier_radius,
        outlier_min_neighbors=args.outlier_min_neighbors, wall_pad=0.0, z_margin=0.0,
    )
    n_filtered = len(pts_filtered)
    clusters = cluster_points(pts_filtered, eps=args.eps, min_samples=args.min_samples)
    tracks = tracker.update(clusters, dt)
    return tracks, n_raw, n_filtered, pts_raw, pts_filtered


def print_tracks(tracks, n_raw, n_filtered, frame_idx, true_xyz=None,
                  min_reliable_npts: int = 6, min_reliable_age: int = 3):
    """min_reliable_npts / min_reliable_age flag a track as likely sensor
    noise rather than a real obstacle: a thin cluster (few points) or a
    freshly-spawned track (not yet matched across several frames) produces
    an unstable centroid/velocity that shouldn't be trusted -- e.g. a track
    with npts=4 and vel=1.7 m/s is almost certainly noise jitter, not
    something actually moving that fast. Sorted closest-first (distance
    from the camera), since that's what matters operationally -- track IDs
    are just creation-order labels (id 275 does NOT mean 275 points; see
    earlier confusion), not a ranking of anything useful to read top-down."""
    n_static = sum(1 for _, tr in tracks if not tr["is_dynamic"])
    n_dynamic = len(tracks) - n_static
    n_noisy = sum(1 for _, tr in tracks
                  if len(tr["points"]) < min_reliable_npts or tr["seen"] < min_reliable_age)

    print(f"\n--- frame {frame_idx}  (raw={n_raw}, filtered={n_filtered}, "
          f"tracks={len(tracks)}: {n_static} static / {n_dynamic} dynamic, "
          f"{n_noisy} flagged noisy) ---")

    if not tracks:
        print("  (no obstacle detected)")

    def _dist(item):
        return float(np.linalg.norm(item[1]["centroid"]))

    for tid, tr in sorted(tracks, key=_dist):
        x, y, z = tr["centroid"]
        dist = _dist((tid, tr))
        kind = "dynamic" if tr["is_dynamic"] else "static"
        npts = len(tr["points"])
        noisy = npts < min_reliable_npts or tr["seen"] < min_reliable_age
        flag = "  <- noisy?" if noisy else ""
        print(f"  track {tid:>3} [{kind:7}]  dist={dist:5.2f}m"
              f"  (x,y,z)=({x:+.3f},{y:+.3f},{z:+.3f})m"
              f"  npts={npts:3d}  age={tr['seen']:3d}"
              f"  vel={np.linalg.norm(tr['vel']):.3f} m/s{flag}")

    if true_xyz is not None:
        if tracks:
            best = min(tracks, key=lambda t: np.linalg.norm(t[1]["centroid"] - true_xyz))
            err = np.linalg.norm(best[1]["centroid"] - true_xyz)
            print(f"  [synthetic check] true=({true_xyz[0]:+.3f},{true_xyz[1]:+.3f},"
                  f"{true_xyz[2]:+.3f})  closest track error = {err:.3f} m")
        else:
            print(f"  [synthetic check] true=({true_xyz[0]:+.3f},{true_xyz[1]:+.3f},"
                  f"{true_xyz[2]:+.3f})  -- MISSED, no track detected")


class LivePlot:
    """Live-updating scatter of the point cloud, in camera frame (origin =
    camera, +z roughly forward -- see backproject_depth_to_world). Raw points
    (post radius-limit, pre voxel/outlier filter) are shown faint/gray so you
    can sanity-check the filter is actually discarding noise rather than real
    structure; filtered points (blue) are what's actually clustered and fed
    into the tracker. Deliberately does NOT plot track centroids -- those are
    a display-only summary (mean of a track's points) and are never what's
    passed to the projector; see tracks_to_constraints() in
    depth_obstacle_estimator.py, which builds one sphere constraint per
    (capped, subsampled) member point of tr["points"], not per centroid.

    mode="3d"  -- single rotatable 3D scatter (original view).
    mode="2d"  -- single flat panel, top-down / bird's-eye: X (right) vs Z
                  (forward/depth, camera optical frame -- see
                  backproject_depth_to_world). Camera sits at the near edge
                  since a camera physically can't see behind itself; the
                  cloud only ever fans out ahead of it, never surrounding
                  it. Y (vertical) is dropped, so height differences (e.g.
                  floor vs ceiling clutter) collapse onto one plane -- use
                  --plot_mode 3d if that distinction matters."""

    def __init__(self, radius_limit: float | None, mode: str = "3d", axis_limit: float = 3.0):
        plt.ion()
        self.mode = mode
        self.axis_limit = radius_limit if radius_limit is not None else axis_limit
        if mode == "3d":
            self.fig = plt.figure(figsize=(7, 7))
            self.ax = self.fig.add_subplot(111, projection="3d")
        elif mode == "2d":
            self.fig, self.ax2d = plt.subplots(figsize=(7, 7))
        else:
            raise ValueError(f"unknown plot mode '{mode}', expected '3d' or '2d'")

    def update(self, pts_raw, pts_filtered, tracks, frame_idx, true_xyz=None):
        if self.mode == "3d":
            self._update_3d(pts_raw, pts_filtered, tracks, frame_idx, true_xyz)
        else:
            self._update_2d(pts_raw, pts_filtered, tracks, frame_idx, true_xyz)

    def _update_3d(self, pts_raw, pts_filtered, tracks, frame_idx, true_xyz):
        ax = self.ax
        ax.cla()

        if len(pts_raw):
            ax.scatter(pts_raw[:, 0], pts_raw[:, 1], pts_raw[:, 2],
                       s=2, c="lightgray", alpha=0.35, label=f"raw ({len(pts_raw)})")
        if len(pts_filtered):
            ax.scatter(pts_filtered[:, 0], pts_filtered[:, 1], pts_filtered[:, 2],
                       s=8, c="tab:blue", label=f"filtered ({len(pts_filtered)})")

        if true_xyz is not None:
            ax.scatter([true_xyz[0]], [true_xyz[1]], [true_xyz[2]],
                       s=140, marker="*", c="gold", edgecolors="black", label="ground truth", zorder=6)

        ax.scatter([0], [0], [0], s=60, c="black", marker="^")  # camera origin

        lim = self.axis_limit
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-0.2, lim)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title(f"Frame {frame_idx}  |  tracks={len(tracks)}")
        ax.legend(loc="upper left", fontsize=8)
        plt.pause(0.001)

    def _update_2d(self, pts_raw, pts_filtered, tracks, frame_idx, true_xyz):
        """Single-panel top-down / bird's-eye view: X (right) vs Z (forward/
        depth), camera optical frame. Deliberately uses Z, not Y, as the
        vertical plot axis -- an RGBD camera only ever sees what's in front
        of the lens (for a D455, roughly an 87 deg horizontal FOV cone), so
        every real point has Z >= 0. Plotting against Z keeps that true: the
        camera sits at the near edge and the cloud only fans out ahead of
        it. An X-vs-Y plot (dropping Z) loses this entirely -- lateral/
        vertical offset alone puts points symmetrically all around the
        camera marker, which visually implies coverage on every side, as if
        the sensor could somehow see behind itself. It can't; that plot was
        just discarding the one axis that would have shown it."""
        ax = self.ax2d
        lim = self.axis_limit
        ax.cla()

        if len(pts_raw):
            ax.scatter(pts_raw[:, 0], pts_raw[:, 2],
                       s=2, c="lightgray", alpha=0.35, label=f"raw ({len(pts_raw)})")
        if len(pts_filtered):
            ax.scatter(pts_filtered[:, 0], pts_filtered[:, 2],
                       s=8, c="tab:blue", label=f"filtered ({len(pts_filtered)})")
        if true_xyz is not None:
            ax.scatter([true_xyz[0]], [true_xyz[2]],
                       s=140, marker="*", c="gold", edgecolors="black",
                       label="ground truth", zorder=6)
        ax.scatter([0], [0], s=60, c="black", marker="^")  # camera, at the
                                                             # near edge -- not
                                                             # the center, since
                                                             # nothing can be
                                                             # detected behind it

        ax.set_xlim(-lim, lim)
        ax.set_ylim(-0.2, lim)  # depth is forward-only; never symmetric
        ax.set_xlabel("X -- right (m)")
        ax.set_ylabel("Z -- forward/depth (m)")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=8)
        ax.set_title(f"Frame {frame_idx}  |  tracks={len(tracks)}  |  top-down (X vs Z)")

        self.fig.tight_layout()
        plt.pause(0.001)

    def close(self):
        plt.ioff()
        plt.close(self.fig)


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

    plot = LivePlot(args.radius_limit, mode=args.plot_mode) if args.viz else None
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
        tracks, n_raw, n_filtered, pts_raw, pts_filtered = process_frame(
            depth, fx, fy, cx, cy, args, tracker, dt)
        print_tracks(tracks, n_raw, n_filtered, i, true_xyz=true_xyz)
        if plot is not None:
            plot.update(pts_raw, pts_filtered, tracks, i, true_xyz=true_xyz)
        time.sleep(max(0.0, 1.0 / args.rate - (time.time() - now)))
    if plot is not None:
        plot.close()


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

    plot = LivePlot(args.radius_limit, mode=args.plot_mode) if args.viz else None
    prev_t = time.time()
    for i, frame_idx in enumerate(range(start, end, args.stride_frames)):
        depth = z["depth"][frame_idx]
        now = time.time()
        dt = now - prev_t if i > 0 else 1.0 / args.rate
        prev_t = now
        tracks, n_raw, n_filtered, pts_raw, pts_filtered = process_frame(
            depth, fx, fy, cx, cy, args, tracker, dt)
        print_tracks(tracks, n_raw, n_filtered, frame_idx)
        if plot is not None:
            plot.update(pts_raw, pts_filtered, tracks, frame_idx)
        time.sleep(max(0.0, 1.0 / args.rate - (time.time() - now)))
    if plot is not None:
        plot.close()


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

    plot = LivePlot(args.radius_limit, mode=args.plot_mode) if args.viz else None
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
            tracks, n_raw, n_filtered, pts_raw, pts_filtered = process_frame(
                depth, intr.fx, intr.fy, intr.ppx, intr.ppy, args, tracker, dt)
            print_tracks(tracks, n_raw, n_filtered, i)
            if plot is not None:
                plot.update(pts_raw, pts_filtered, tracks, i)
            i += 1
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()
        if plot is not None:
            plot.close()


# =============================================================================
# Source: ROS2 subscriber (subscribes to an already-running depth publisher,
# e.g. the realsense2_camera node -- doesn't open the camera itself, so it can
# run alongside rviz2 or on a different machine than the camera)
# =============================================================================

def run_ros2(args, tracker):
    try:
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import Image, CameraInfo
    except ImportError:
        print("[ERROR] rclpy/sensor_msgs not importable. Source your ROS2 install first, e.g.:\n"
              "        source /opt/ros/humble/setup.bash")
        return

    plot = LivePlot(args.radius_limit, mode=args.plot_mode) if args.viz else None

    class DepthObstacleNode(Node):
        """Caches the latest CameraInfo (for intrinsics) and runs the full
        pipeline once per depth frame received. No polling loop needed --
        rclpy.spin() drives everything via the two subscription callbacks."""

        def __init__(self):
            super().__init__("depth_obstacle_estimator")
            self.fx = self.fy = self.cx = self.cy = None
            self.frame_idx = 0
            self.prev_t = None
            self.create_subscription(CameraInfo, args.ros2_camera_info_topic, self._info_cb, 10)
            self.create_subscription(Image, args.ros2_depth_topic, self._depth_cb, 10)
            self.get_logger().info(
                f"Subscribed: depth={args.ros2_depth_topic}  "
                f"camera_info={args.ros2_camera_info_topic}"
            )

        def _info_cb(self, msg: "CameraInfo"):
            # msg.k is the row-major 3x3 intrinsic matrix [fx 0 cx; 0 fy cy; 0 0 1]
            self.fx, self.fy = msg.k[0], msg.k[4]
            self.cx, self.cy = msg.k[2], msg.k[5]

        def _depth_cb(self, msg: "Image"):
            if self.fx is None:
                return  # CameraInfo hasn't arrived yet; skip until intrinsics are known

            if msg.encoding == "16UC1":
                depth = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
                depth = depth.astype(np.float32) * 0.001  # RealSense: raw mm -> metres
            elif msg.encoding == "32FC1":
                depth = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
            else:
                self.get_logger().warn(f"unsupported depth encoding '{msg.encoding}', skipping frame")
                return

            now = time.time()
            dt = now - self.prev_t if self.prev_t is not None else 1.0 / args.rate
            self.prev_t = now
            tracks, n_raw, n_filtered, pts_raw, pts_filtered = process_frame(
                depth, self.fx, self.fy, self.cx, self.cy, args, tracker, dt)
            print_tracks(tracks, n_raw, n_filtered, self.frame_idx)
            if plot is not None:
                plot.update(pts_raw, pts_filtered, tracks, self.frame_idx)
            self.frame_idx += 1

    rclpy.init()
    node = DepthObstacleNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        if plot is not None:
            plot.close()


# =============================================================================
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", choices=["synthetic", "zarr", "realsense", "ros2"], default="synthetic")

    # shared pipeline params (same names/defaults as depth_obstacle_estimator.py's usage in eval_crazieflie1.py)
    p.add_argument("--max_range", type=float, default=5.0)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--voxel_size", type=float, default=0.05)
    p.add_argument("--outlier_radius", type=float, default=0.15)
    p.add_argument("--outlier_min_neighbors", type=int, default=4)
    p.add_argument("--eps", type=float, default=0.12)
    p.add_argument("--min_samples", type=int, default=4)
    p.add_argument("--rate", type=float, default=10.0, help="replay/print rate in Hz")
    p.add_argument("--radius_limit", type=float, default=None,
                    help="spherical crop radius in metres from the camera origin, applied "
                         "to the raw backprojected cloud before filtering (in addition to "
                         "--max_range, which only bounds per-ray depth, not true 3D distance). "
                         "Also sets the plot axes when --viz is used. Default: no extra crop.")
    p.add_argument("--viz", action="store_true",
                    help="show a live-updating scatter plot of raw/filtered points and tracks")
    p.add_argument("--plot_mode", choices=["3d", "2d"], default="3d",
                    help="'3d': single rotatable 3D scatter. '2d': two flat panels stacked, "
                         "top-down X-Y above and side-on X-Z below -- easier to read corridor/"
                         "wall clearance without rotating a 3D view. Only used with --viz.")

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

    # realsense source (direct pyrealsense2 connection)
    p.add_argument("--rs_width", type=int, default=424)
    p.add_argument("--rs_height", type=int, default=240)
    p.add_argument("--rate_int", type=int, default=30, help="RealSense stream fps (integer)")

    # ros2 source (subscribes to an already-running publisher instead)
    p.add_argument("--ros2_depth_topic", type=str,
                    default="/camera/camera/aligned_depth_to_color/image_raw")
    p.add_argument("--ros2_camera_info_topic", type=str,
                    default="/camera/camera/aligned_depth_to_color/camera_info")

    args = p.parse_args()
    tracker = ObstacleTracker()

    if args.source == "synthetic":
        run_synthetic(args, tracker)
    elif args.source == "zarr":
        run_zarr(args, tracker)
    elif args.source == "realsense":
        run_realsense(args, tracker)
    elif args.source == "ros2":
        run_ros2(args, tracker)


if __name__ == "__main__":
    main()