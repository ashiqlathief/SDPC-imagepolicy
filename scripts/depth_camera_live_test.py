"""
python scripts/depth_camera_live_test.py --source isaac --viz --radius_limit 3.0 --plot_mode 3d
python scripts/depth_camera_live_test.py --source isaac --viz --radius_limit 3.0 --plot_mode 2d --plot_axes topdown
python scripts/depth_camera_live_test.py --source isaac --viz --radius_limit 3.0 --plot_mode 2d --plot_axes frontal
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
    crop_points_by_world_z,
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


def process_frame(depth, fx, fy, cx, cy, args, tracker, dt, world_crop=None):
    """world_crop: optional (pos_cam_w, quat_cam_w, z_lo, z_hi) -- when given,
    drops any point whose WORLD-frame z falls outside [z_lo, z_hi] before
    clustering, via depth_obstacle_estimator.crop_points_by_world_z().
    Only meaningful for a source that actually knows the scene's geometry
    (--source isaac knows the floor is at world z=0 and the ceiling at
    CEILING_HEIGHT); the handheld-camera sources have no such knowledge,
    hence why this is opt-in rather than baked into the crop pipeline
    everyone shares. Output stays camera-frame either way -- this only
    removes points, it doesn't change the convention downstream."""
    pts_raw = backproject_depth_to_world(
        depth, fx, fy, cx, cy, IDENTITY_POS, IDENTITY_QUAT,
        max_range=args.max_range, stride=args.stride,
    )
    pts_raw = apply_radius_limit(pts_raw, args.radius_limit)
    if world_crop is not None:
        pos_cam_w, quat_cam_w, z_lo, z_hi = world_crop
        pts_raw = crop_points_by_world_z(pts_raw, pos_cam_w, quat_cam_w, z_lo, z_hi)
    n_raw = len(pts_raw)
    pts_filtered = filter_points(
        pts_raw, NO_CROP_BOUNDS, NO_CROP_BOUNDS, NO_CROP_BOUNDS,
        voxel_size=args.voxel_size, outlier_radius=args.outlier_radius,
        outlier_min_neighbors=args.outlier_min_neighbors, wall_pad=0.0, z_margin=0.0,
        # output_2d intentionally NOT used here: pts_filtered stays full 3D so
        # the point-cloud display (both plot modes) is unaffected regardless
        # of --xy_only. Only the clustering input below is flattened when
        # requested -- see build_position_projector_from_points in
        # eval_crazieflie1.py, which already only reads (x, y) from track
        # points/centroids either way, so nothing downstream needs to care.
    )
    n_filtered = len(pts_filtered)
    cluster_input = pts_filtered[:, :2] if args.xy_only else pts_filtered
    clusters = cluster_points(cluster_input, eps=args.eps, min_samples=args.min_samples)
    tracks = tracker.update(clusters, dt)
    return tracks, n_raw, n_filtered, pts_raw, pts_filtered


def print_tracks(tracks, n_raw, n_filtered, frame_idx,
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
        centroid = tr["centroid"]
        dist = _dist((tid, tr))
        kind = "dynamic" if tr["is_dynamic"] else "static"
        npts = len(tr["points"])
        noisy = npts < min_reliable_npts or tr["seen"] < min_reliable_age
        flag = "  <- noisy?" if noisy else ""
        if len(centroid) == 3:
            x, y, z = centroid
            pos_str = f"(x,y,z)=({x:+.3f},{y:+.3f},{z:+.3f})m"
        else:  # --xy_only: clustering ran on (x, y) only, no z to show
            x, y = centroid
            pos_str = f"(x,y)  =({x:+.3f},{y:+.3f})m"
        print(f"  track {tid:>3} [{kind:7}]  dist={dist:5.2f}m  {pos_str}"
              f"  npts={npts:3d}  age={tr['seen']:3d}"
              f"  vel={np.linalg.norm(tr['vel']):.3f} m/s{flag}")


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
    mode="2d"  -- single flat panel; see the `axes` param for which pair of
                  camera axes is plotted.

    Axis labels are CAMERA-frame letters (x=right, y=down, z=forward/depth),
    NOT world-frame (CYLINDERS/corridor bounds use x=depth-down-the-corridor,
    y=lateral -- the opposite of this file's x/y). Same letters, different
    physical meanings; that mismatch is a real, repeated source of
    confusion (see conversation) even when the underlying math was
    correct, so keep it in mind when comparing a plotted (x,y,z) here
    against a world-frame (x,y) elsewhere in this codebase. The vertical
    axis in both plot modes is plotted as -y (camera-down negated), so
    it's labelled "-Y (m)", not "Y (m)" -- the sign is real, not a typo."""

    def __init__(self, radius_limit: float | None, mode: str = "3d",
                 axes: str = "topdown", axis_limit: float = 3.0):
        # axis_limit is only the *fallback* used when radius_limit is None --
        # callers should pass args.max_range so the view window actually
        # covers everything --max_range let through the backprojection,
        # instead of silently clipping anything beyond this default 3.0m.
        plt.ion()
        self.mode = mode
        self.axes = axes  # only used when mode == "2d": "topdown" or "frontal"
        self.axis_limit = radius_limit if radius_limit is not None else axis_limit
        if mode == "3d":
            self.fig = plt.figure(figsize=(7, 7))
            self.ax = self.fig.add_subplot(111, projection="3d")
        elif mode == "2d":
            self.fig, self.ax2d = plt.subplots(figsize=(7, 7))
        else:
            raise ValueError(f"unknown plot mode '{mode}', expected '3d' or '2d'")

    def update(self, pts_raw, pts_filtered, tracks, frame_idx):
        if self.mode == "3d":
            self._update_3d(pts_raw, pts_filtered, tracks, frame_idx)
        else:
            self._update_2d(pts_raw, pts_filtered, tracks, frame_idx)

    def _update_3d(self, pts_raw, pts_filtered, tracks, frame_idx):
        # Point-cloud data is in camera/ROS convention: x=right, y=DOWN,
        # z=forward/depth. matplotlib's 3D axes render their 3rd argument as
        # the vertical/up axis on screen -- plotting data-z (forward) there
        # made anything at a fixed depth look "up", and real vertical extent
        # (data-y) come out along a horizontal-ish screen axis instead, so a
        # flat, physically-horizontal structure rendered visibly tilted/
        # rotated. Remap so screen-vertical = real up (-y), and the two
        # screen-horizontal axes are right (x) and forward/depth (z) --
        # matches the "topdown" 2d mode's convention, just with height added.
        def _remap(pts):
            return pts[:, 0], pts[:, 2], -pts[:, 1]

        ax = self.ax
        ax.cla()

        if len(pts_raw):
            xs, ys, zs = _remap(pts_raw)
            ax.scatter(xs, ys, zs, s=2, c="lightgray", alpha=0.35, label=f"raw ({len(pts_raw)})")
        if len(pts_filtered):
            xs, ys, zs = _remap(pts_filtered)
            ax.scatter(xs, ys, zs, s=8, c="tab:blue", label=f"filtered ({len(pts_filtered)})")

        ax.scatter([0], [0], [0], s=60, c="black", marker="^")  # camera origin

        lim = self.axis_limit
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-0.2, lim)   # forward/depth: camera can't see behind itself
        ax.set_zlim(-lim, lim)  # up/down relative to the camera
        ax.set_xlabel("Y (m)")
        ax.set_ylabel("X (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title(f"Frame {frame_idx}  |  tracks={len(tracks)}")
        ax.legend(loc="upper left", fontsize=8)
        # Static 3D camera angle is a genuine tradeoff, not something with a
        # single "correct" fixed value: matplotlib's default (elev=30,
        # azim=-60) makes flat/level structure look like a tilted
        # parallelogram (pure viewing-angle illusion, verified against a
        # synthetic level grid). azim=-90 fixes that -- but at azim=-90 the
        # camera looks almost straight down the depth axis, so near/far
        # obstacles foreshorten together and depth becomes hard to read
        # (verified: two known-different-depth pillar groups nearly
        # overlapped at azim=-90). This is a middle ground, not a fix for
        # either extreme -- for a geometrically UNAMBIGUOUS read of what's
        # near vs far, use --plot_mode 2d --plot_axes topdown instead, which
        # has no perspective ambiguity at all. Still click-drag rotatable in a
        # real (non-Agg) window. ax.cla() above resets the view each frame,
        # so this has to be set every call, not just once in __init__.
        ax.view_init(elev=20, azim=-70)
        plt.pause(0.001)

    def _update_2d(self, pts_raw, pts_filtered, tracks, frame_idx):
        """axes="topdown" (default, recommended): bird's-eye view, right vs
        forward/depth. Camera sits at the near edge since a camera
        physically can't see behind itself -- the cloud only ever fans out
        ahead of it, never surrounding it.

        axes="frontal": front-on view, right vs up -- depth is dropped, so
        the camera marker sits in the MIDDLE with points on every side,
        which looks like 360 deg coverage even though the sensor only sees
        a forward FOV cone. Two real points at different depths but similar
        (right, up) will also overlap here. Useful only if you specifically
        want to check lateral/vertical spread irrespective of distance --
        for a geometrically honest view, use axes="topdown"."""
        ax = self.ax2d
        lim = self.axis_limit

        if self.axes == "topdown":
            def _cols(pts):
                return pts[:, 0], pts[:, 2]  # x (right), z (forward/depth), as-is
            xlabel, ylabel = "Y (m)", "X (m)"
            xlim, ylim = (-lim, lim), (-0.2, lim)  # depth is forward-only
            title_suffix = "top-down"
        elif self.axes == "frontal":
            def _cols(pts):
                return pts[:, 0], -pts[:, 1]  # x (right), -y (camera-y is down-positive, negate for up)
            xlabel, ylabel = "Y (m)", "Z (m)"
            xlim, ylim = (-lim, lim), (-lim, lim)
            title_suffix = "front-on"
        else:
            raise ValueError(f"unknown axes '{self.axes}', expected 'topdown' or 'frontal'")

        ax.cla()
        if len(pts_raw):
            xs, ys = _cols(pts_raw)
            ax.scatter(xs, ys, s=2, c="lightgray", alpha=0.35, label=f"raw ({len(pts_raw)})")
        if len(pts_filtered):
            xs, ys = _cols(pts_filtered)
            ax.scatter(xs, ys, s=8, c="tab:blue", label=f"filtered ({len(pts_filtered)})")
        ax.scatter([0], [0], s=60, c="black", marker="^")  # camera origin

        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=8)
        ax.set_title(f"Frame {frame_idx}  |  tracks={len(tracks)}  |  {title_suffix}")

        self.fig.tight_layout()
        plt.pause(0.001)

    def close(self):
        plt.ioff()
        plt.close(self.fig)


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

    plot = LivePlot(args.radius_limit, mode=args.plot_mode, axes=args.plot_axes, axis_limit=args.max_range) if args.viz else None

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
# Source: Isaac Sim (the drone's own FPV depth camera, live inside a running
# IsaacLab episode -- same camera/physics/obstacle geometry
# eval_crazieflie1.py --depth_obstacles uses, but held here at a fixed hover
# setpoint instead of flown by a loaded policy, so this stays a
# self-contained smoke test with no checkpoint required.
# =============================================================================

def run_isaac(args, tracker):
    import importlib
    os.environ["CRAZYFLIE_ENV_HEADLESS"] = "1"

    cfg = importlib.import_module("config.avoiding-crazyflie")
    cfg.USE_DEPTH = True  # must be set before crazyflie_env_cfg is first imported below

    from isaac.scripts.crazyflie_env import Crazyflie, CrazyflieEnvCfg
    from isaac.scripts.crazyflie_env_cfg import CEILING_HEIGHT  # real value, not a comment
    from depth_obstacle_estimator import (
        camera_intrinsics, camera_world_pose, quat_apply,
        FPV_WIDTH, FPV_HEIGHT, FPV_FOCAL_LENGTH, FPV_HORIZONTAL_APERTURE,
    )

    fx, fy, cx, cy = camera_intrinsics(FPV_WIDTH, FPV_HEIGHT, FPV_FOCAL_LENGTH, FPV_HORIZONTAL_APERTURE)

    env_cfg = CrazyflieEnvCfg(
        num_envs=1,
        device=args.isaac_device,
        dt=args.isaac_dt,
        dynamic_obstacles=args.isaac_dynamic_obstacles,
    )
    env = Crazyflie(env_cfg)
    env.reset()

    # fixed hover setpoint, held for the whole run -- not flown anywhere.
    hover_xyz = np.array([args.isaac_hover_xy[0], args.isaac_hover_xy[1],
                           args.isaac_altitude], dtype=np.float32)
    step_dt = args.isaac_dt * getattr(env, "count", 100)

    plot = LivePlot(args.radius_limit, mode=args.plot_mode, axes=args.plot_axes, axis_limit=args.max_range) if args.viz else None
    try:
        for i in range(args.n_frames):
            env.step(hover_xyz)

            depth = env.get_depth()
            root = env.robot.data.root_state_w[0].detach().cpu().numpy()
            pos_body_w, quat_body_w = root[0:3], root[3:7]
            pos_cam_w, quat_cam_w = camera_world_pose(pos_body_w, quat_body_w)

            # Unlike the handheld-camera sources, we actually know this
            # scene's geometry: floor at world z=0, ceiling at
            # CEILING_HEIGHT (real value imported above -- NOT the stale
            # "=1.0" comment in crazyflie_env_cfg.py, which is wrong since
            # WALL_HEIGHT was bumped to 2.0). Crop both out (with a margin)
            # before clustering, so a real floor/ceiling patch can't get
            # DBSCAN'd into a fake "obstacle" track alongside the real
            # cylinders.
            world_crop = (pos_cam_w, quat_cam_w,
                          args.isaac_surface_margin, CEILING_HEIGHT - args.isaac_surface_margin)
            tracks, n_raw, n_filtered, pts_raw, pts_filtered = process_frame(
                depth, fx, fy, cx, cy, args, tracker, step_dt, world_crop=world_crop)
            print_tracks(tracks, n_raw, n_filtered, i)
            # print_tracks' (x,y,z) is CAMERA frame (x=right, y=down,
            # z=forward) -- do NOT compare that x against CYLINDERS' world
            # x (distance down the corridor, ~2.0-2.5m here); those are
            # different physical directions that just share the letter "x".
            # A cylinder at world x=2 shows up as camera-frame z~=2, not x.
            # This converts each track back to world frame (we have the
            # real pose here, unlike the handheld-camera sources) so it's
            # directly comparable to CYLINDERS without doing that mental
            # translation by hand.
            for tid, tr in tracks:
                c = tr["centroid"]
                if len(c) < 3:
                    print(f"    world track {tid:>3}: unavailable -- --xy_only dropped "
                          f"forward/depth (camera-z) before clustering, so there's no "
                          f"z left to reconstruct a world position from.")
                    continue
                world_c = pos_cam_w + quat_apply(quat_cam_w, np.asarray(c, dtype=np.float32))
                print(f"    world track {tid:>3}: (x,y,z)=({world_c[0]:+.3f},{world_c[1]:+.3f},"
                      f"{world_c[2]:+.3f})m")
            if len(pts_filtered):
                # World-frame height of every detected point (post floor/
                # ceiling crop above, so this should now mostly reflect
                # real obstacles) -- settles floor (world z~=0) vs ceiling
                # (world z~=CEILING_HEIGHT) vs real obstacle with a number,
                # since eyeballing height off the oblique 3D plot isn't
                # reliable (even the camera marker, truly at up=0, doesn't
                # render at the "0" tick -- see conversation).
                world_pts = pos_cam_w[None, :] + quat_apply(quat_cam_w, pts_filtered)
                wz = world_pts[:, 2]
                print(f"  [height check] world z: min={wz.min():.3f} max={wz.max():.3f} "
                      f"mean={wz.mean():.3f}  (floor=0.0, ceiling={CEILING_HEIGHT:.2f}, "
                      f"camera={pos_cam_w[2]:.3f})")
            if plot is not None:
                plot.update(pts_raw, pts_filtered, tracks, i)
    finally:
        if plot is not None:
            plot.close()
        env.close()


# =============================================================================
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", choices=["ros2", "isaac"], default="ros2")

    # shared pipeline params (same names/defaults as depth_obstacle_estimator.py's usage in eval_crazieflie1.py)
    p.add_argument("--max_range", type=float, default=5.0)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--voxel_size", type=float, default=0.05)
    p.add_argument("--outlier_radius", type=float, default=0.15)
    p.add_argument("--outlier_min_neighbors", type=int, default=4)
    p.add_argument("--eps", type=float, default=0.12)
    p.add_argument("--min_samples", type=int, default=4)
    p.add_argument("--xy_only", action="store_true",
                    help="cluster/track in (x,y) only, dropping z right before DBSCAN. "
                         "Real corridor obstacles are vertical columns spanning the full "
                         "flight envelope height; 3D clustering can chop one physical column "
                         "into several tracks stacked at different heights (points >eps apart "
                         "in z alone). This merges those height-slices back into one track. "
                         "Point-cloud display (both --plot_mode 3d and 2d) is unaffected -- "
                         "only clustering/tracking is flattened, not what you see plotted.")
    p.add_argument("--rate", type=float, default=10.0, help="replay/print rate in Hz")
    p.add_argument("--radius_limit", type=float, default=None,
                    help="spherical crop radius in metres from the camera origin, applied "
                         "to the raw backprojected cloud before filtering (in addition to "
                         "--max_range, which only bounds per-ray depth, not true 3D distance). "
                         "Also sets the plot axes when --viz is used. Default: no extra crop.")
    p.add_argument("--viz", action="store_true",
                    help="show a live-updating scatter plot of raw/filtered points and tracks")
    p.add_argument("--plot_mode", choices=["3d", "2d"], default="3d",
                    help="'3d': single rotatable 3D scatter. '2d': single flat panel (see "
                         "--plot_axes for which pair). Only used with --viz.")
    p.add_argument("--plot_axes", choices=["topdown", "frontal"], default="topdown",
                    help="Only used with --viz --plot_mode 2d. 'topdown' (default, recommended): "
                         "bird's-eye, right vs forward-depth -- camera sits at the near edge, "
                         "geometrically honest since a camera can't see behind itself. "
                         "'frontal': front-on, right vs up, depth dropped -- camera sits in the "
                         "middle with points on every side, which visually implies 360 deg "
                         "coverage the sensor doesn't actually have. Use only if you specifically "
                         "need lateral/vertical spread irrespective of distance.")

    # ros2 source (subscribes to an already-running publisher instead)
    p.add_argument("--ros2_depth_topic", type=str,
                    default="/camera/camera/aligned_depth_to_color/image_raw")
    p.add_argument("--ros2_camera_info_topic", type=str,
                    default="/camera/camera/aligned_depth_to_color/camera_info")

    # isaac source (boots the real Crazyflie env)
    p.add_argument("--n_frames", type=int, default=600, help="episode length, control steps")
    p.add_argument("--isaac_device", type=str, default="cuda:0")
    p.add_argument("--isaac_dt", type=float, default=0.005, help="sim physics dt (s)")
    p.add_argument("--isaac_hover_xy", type=float, nargs=2, default=[0.0, 0.0], metavar=("X", "Y"),
                    help="fixed (x, y) hover setpoint, world frame -- drone holds this position "
                         "for the whole run rather than flying anywhere. Default (0,0) is the "
                         "spawn point, facing the cylinder corridor along +x.")
    p.add_argument("--isaac_altitude", type=float, default=0.5, help="fixed hover altitude (world z) in metres")
    p.add_argument("--isaac_dynamic_obstacles", action="store_true", default=False,
                    help="move all cylinders sinusoidally while hovering, instead of a static corridor "
                         "-- exercises the static/dynamic classifier without the drone itself moving")
    p.add_argument("--isaac_surface_margin", type=float, default=0.1,
                    help="metres of world-z clearance cropped off both the floor (z=0) and the "
                         "real ceiling (CEILING_HEIGHT, imported from crazyflie_env_cfg.py -- NOT "
                         "a hardcoded guess) before clustering, so real floor/ceiling patches "
                         "aren't DBSCAN'd into fake 'obstacle' tracks alongside the real cylinders.")

    args = p.parse_args()
    tracker = ObstacleTracker()

    if args.source == "ros2":
        run_ros2(args, tracker)
    elif args.source == "isaac":
        run_isaac(args, tracker)


if __name__ == "__main__":
    main()