from __future__ import annotations
import argparse
import importlib
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import depth_obstacle_estimator as detect_mod  # noqa: E402
from depth_obstacle_estimator import (  # noqa: E402
    detect_obstacles_umap, camera_world_pose, quat_apply,
    DEPTH_FX, DEPTH_FY, DEPTH_CX, DEPTH_CY,
)
detect_mod.DEBUG_DETECT = False
UMAP_MAX_RANGE = 5.0
UMAP_BIN_SIZE = 100
UMAP_T_POI = 500.0
UMAP_T_THO = 1800.0
UMAP_MIN_PIXEL_COUNT = 20  # raw pixel-count floor for a U-map cell to count as a real
                          # surface (was UMAP_BIN_THRESH=150 on a per-frame-normalized
                          # 0-255 scale -- see depth_obstacle_estimator._umap_contours()'s
                          # min_pixel_count docstring; this default is unvalidated
                          # against real depth data, retune as needed)
DEPTH_OBSTACLE_RADIUS = 0.3
MAX_DEPTH_OBSTACLES = 10

EDGE_CROP_FRAC = 0.12  # fraction of frame width blanked out on each left/right edge before
                       # detection. Stereo depth cameras (e.g. RealSense) are known-noisier
                       # near the frame's outer columns -- reduced left/right-imager overlap
                       # there -- visibly confirmed as speckled edge noise in a real depth
                       # frame screenshot (2026-09-06). Blanking to 0 makes _umap_contours()
                       # treat that margin as out-of-frame (it already drops <=0/non-finite
                       # pixels), same as it does for genuinely out-of-range background.
                       # Default here first; port to eval_crazieflie1pos.py/eval_crazieflieros2.py
                       # once validated against real data.


def _crop_depth_edges(depth, frac=EDGE_CROP_FRAC):
    """Blank the outermost `frac` of columns on each side of a (H,W) depth frame (metres),
    in place on a copy. See EDGE_CROP_FRAC above for why."""
    if frac <= 0:
        return depth
    margin = int(depth.shape[1] * frac)
    if margin <= 0:
        return depth
    depth = depth.copy()
    depth[:, :margin] = 0.0
    depth[:, -margin:] = 0.0
    return depth


def detect_depth_obstacles(depth_frame, pos_body_w, quat_body_w, fx, fy, cx, cy):
    """Identical logic to eval_crazieflie1pos.py/eval_crazieflieros2.py's same-named
    function. Returns [(x, y, radius), ...] world-frame."""
    depth_frame = _crop_depth_edges(depth_frame)
    pos_cam_w, quat_cam_w = camera_world_pose(pos_body_w, quat_body_w)
    detections = detect_obstacles_umap(
        depth_frame, fx, fy, cx, cy,
        max_range_m=UMAP_MAX_RANGE, bin_size=UMAP_BIN_SIZE,
        t_poi=UMAP_T_POI, t_tho=UMAP_T_THO, min_pixel_count=UMAP_MIN_PIXEL_COUNT,
        max_obstacles=MAX_DEPTH_OBSTACLES,
    )
    points = []
    for pos_cam, half_w, _half_h in detections:
        world_xyz = pos_cam_w + quat_apply(quat_cam_w, pos_cam)
        # half_h is the obstacle's VERTICAL extent, not horizontal -- see
        # eval_crazieflie1pos.py's identical fix / [[umap_obstacle_detector_bugs]].
        radius = max(DEPTH_OBSTACLE_RADIUS, half_w)
        points.append((float(world_xyz[0]), float(world_xyz[1]), float(radius)))
    return points


def closest_ground_truth(x, y, cylinders):
    """Nearest (cx, cy) in `cylinders` to (x, y), plus the distance -- so the printout
    reads directly as an accuracy number instead of raw coordinates to cross-reference
    by hand."""
    if not cylinders:
        return None, None
    dists = [float(np.hypot(x - cx, y - cy)) for cx, cy in cylinders]
    i = int(np.argmin(dists))
    return cylinders[i], dists[i]


def print_detections(frame_idx, points, cylinders=None):
    print(f"\n--- frame {frame_idx}  {len(points)} detection(s) ---")
    for (x, y, r) in points:
        if cylinders:
            gt, dist = closest_ground_truth(x, y, cylinders)
            print(f"  ({x:+.3f}, {y:+.3f}) r={r:.2f}  nearest truth={gt}  err={dist:.3f}m")
        else:
            print(f"  ({x:+.3f}, {y:+.3f}) r={r:.2f}")


# =============================================================================
# Source: ROS2 subscriber (subscribes to an already-running depth publisher,
# e.g. the realsense2_camera node -- doesn't open the camera itself, so it can
# run alongside rviz2 or on a different machine than the camera)
# =============================================================================

def run_ros2(args):

    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image
    from geometry_msgs.msg import PoseStamped
    from rclpy.qos import qos_profile_sensor_data

    _IDENTITY_QUAT = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # (w,x,y,z)

    class DepthUmapNode(Node):
        """Pose-subscribed (unlike the old camera-relative-only version): transforms each
        detection to world frame the same way eval_crazieflieros2.py's
        detect_depth_obstacles() does, via camera_world_pose()+quat_apply(). Prints '(no
        pose yet)' and skips the world transform for any frame that arrives before the
        first pose message."""

        def __init__(self):
            super().__init__("depth_umap_test")
            self.frame_idx = 0
            self.fx, self.fy, self.cx, self.cy = DEPTH_FX, DEPTH_FY, DEPTH_CX, DEPTH_CY
            self.pos_body_w = None
            self.quat_body_w = _IDENTITY_QUAT.copy()
            self.create_subscription(Image, args.ros2_depth_topic, self._depth_cb, qos_profile_sensor_data)
            self.create_subscription(PoseStamped, args.ros2_pose_topic, self._pose_cb, qos_profile_sensor_data)
            self.get_logger().info(f"Subscribed: depth={args.ros2_depth_topic}"f"pose={args.ros2_pose_topic}  camera_info=<hardcoded>")

        def _pose_cb(self, msg: "PoseStamped"):
            p, o = msg.pose.position, msg.pose.orientation
            self.pos_body_w = np.array([p.x, p.y, p.z], dtype=np.float32)
            self.quat_body_w = np.array([o.w, o.x, o.y, o.z], dtype=np.float32)

        def _depth_cb(self, msg: "Image"):
            depth = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
            depth = depth.astype(np.float32) * 0.001  # RealSense: raw mm -> metres
            depth = _crop_depth_edges(depth)

            detections = detect_obstacles_umap(
                depth, self.fx, self.fy, self.cx, self.cy,
                max_range_m=UMAP_MAX_RANGE, bin_size=UMAP_BIN_SIZE,
                t_poi=UMAP_T_POI, t_tho=UMAP_T_THO, min_pixel_count=UMAP_MIN_PIXEL_COUNT,
                max_obstacles=MAX_DEPTH_OBSTACLES,
            )
            if self.pos_body_w is not None:
                pos_cam_w, quat_cam_w = camera_world_pose(self.pos_body_w, self.quat_body_w)
            print(f"\n--- frame {self.frame_idx}  {len(detections)} detection(s) ---")
            for pos_cam, half_w, _half_h in detections:
                radius = max(DEPTH_OBSTACLE_RADIUS, half_w)
                if self.pos_body_w is not None:
                    world_xyz = pos_cam_w + quat_apply(quat_cam_w, pos_cam)
                    print(f"  world=({world_xyz[0]:+.3f},{world_xyz[1]:+.3f},{world_xyz[2]:+.3f})m  "
                          f"cam=({pos_cam[0]:+.3f},{pos_cam[1]:+.3f},{pos_cam[2]:+.3f})m  r={radius:.2f}")
                else:
                    print(f"  (no pose yet) cam=({pos_cam[0]:+.3f},{pos_cam[1]:+.3f},{pos_cam[2]:+.3f})m  r={radius:.2f}")
            self.frame_idx += 1

    rclpy.init()
    node = DepthUmapNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


# =============================================================================
# Source: Isaac Sim (the drone's own FPV depth camera, live inside a running
# IsaacLab episode, held at a fixed hover setpoint -- self-contained smoke test,
# no checkpoint required. Ground truth comes from env.get_cylinder_positions().)
# =============================================================================

def run_isaac(args):
    os.environ["CRAZYFLIE_ENV_HEADLESS"] = "1"
    cfg = importlib.import_module("config.avoiding-crazyflie")
    cfg.USE_DEPTH = True  # must be set before crazyflie_env_cfg is first imported below

    from isaac.scripts.crazyflie_env import Crazyflie, CrazyflieEnvCfg

    env_cfg = CrazyflieEnvCfg(
        num_envs=1, device=args.isaac_device, dt=args.isaac_dt,
        dynamic_obstacles=args.isaac_dynamic_obstacles,
    )
    env = Crazyflie(env_cfg)
    env.reset()

    # fixed hover setpoint, held for the whole run -- not flown anywhere.
    hover_xyz = np.array([args.isaac_hover_xy[0], args.isaac_hover_xy[1],
                           args.isaac_altitude], dtype=np.float32)
    print(f"[INFO] Hovering at ({hover_xyz[0]}, {hover_xyz[1]}, {hover_xyz[2]})")
    env.step(hover_xyz)
    K = env.cam.data.intrinsic_matrices[0].detach().cpu().numpy()
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    print(f"[INFO] intrinsics (from IsaacSim camera): fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}")

    try:
        for i in range(args.n_frames):
            env.step(hover_xyz)
            depth = env.get_depth()
            depth_2d = depth[..., 0] if depth.ndim == 3 else depth
            root = env.robot.data.root_state_w[0].detach().cpu().numpy()
            pos_body_w, quat_body_w = root[0:3], root[3:7]

            points = detect_depth_obstacles(depth_2d, pos_body_w, quat_body_w, fx, fy, cx, cy)
            print_detections(i, points, env.get_cylinder_positions())
    finally:
        env.close()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", choices=["ros2", "isaac"], default="ros2")

    p.add_argument("--ros2_depth_topic", type=str, default="/camera/camera/depth/image_rect_raw")
    p.add_argument("--ros2_pose_topic", type=str, default="/mavros/local_position/pose",)

    # isaac source (boots the real Crazyflie env)
    p.add_argument("--n_frames", type=int, default=300, help="episode length, control steps")
    p.add_argument("--isaac_device", type=str, default="cuda:0")
    p.add_argument("--isaac_dt", type=float, default=0.005, help="sim physics dt (s)")
    p.add_argument("--isaac_hover_xy", type=float, nargs=2, default=[0.5, 0.0], metavar=("X", "Y"),
                    help="fixed (x, y) hover setpoint, world frame -- drone holds this position "
                         "for the whole run rather than flying anywhere. Default (0,0) is the "
                         "spawn point, facing the cylinder corridor along +x.")
    p.add_argument("--isaac_altitude", type=float, default=0.5, help="fixed hover altitude (world z) in metres")
    p.add_argument("--isaac_dynamic_obstacles", action="store_true", default=False,
                    help="move all cylinders sinusoidally while hovering, instead of a static corridor")

    args = p.parse_args()

    if args.source == "ros2":
        run_ros2(args)
    elif args.source == "isaac":
        run_isaac(args)


if __name__ == "__main__":
    main()
