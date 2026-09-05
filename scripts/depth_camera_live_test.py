"""
Isaac sim (boots a hovering drone, ground-truth comparison against config CYLINDERS):
python scripts/depth_camera_live_test.py --source isaac --n_frames 300 --isaac_hover_xy 0 0 --isaac_altitude 0.5

Real hardware (subscribes to an already-running depth publisher):
python scripts/depth_camera_live_test.py --source ros2 --ros2_depth_topic /camera/camera/depth/image_raw --ros2_hardcoded_intrinsics
"""
from __future__ import annotations
import argparse
import importlib
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from depth_obstacle_estimator import (  # noqa: E402
    detect_obstacles_umap, camera_world_pose, quat_apply,
    DEPTH_FX, DEPTH_FY, DEPTH_CX, DEPTH_CY,
)

# Same defaults as eval_crazieflie1pos.py / eval_crazieflieros2.py -- this tests the
# exact configuration actually deployed, not a fresh set of knobs.
UMAP_MAX_RANGE = 3.0
UMAP_BIN_SIZE = 200
UMAP_T_POI = 500.0
UMAP_T_THO = 1800.0
UMAP_BIN_THRESH = 150
DEPTH_OBSTACLE_RADIUS = 0.3
MAX_DEPTH_OBSTACLES = 5


def detect_depth_obstacles(depth_frame, pos_body_w, quat_body_w, fx, fy, cx, cy):
    """Identical logic to eval_crazieflie1pos.py/eval_crazieflieros2.py's same-named
    function. Returns [(x, y, radius), ...] world-frame."""
    pos_cam_w, quat_cam_w = camera_world_pose(pos_body_w, quat_body_w)
    detections = detect_obstacles_umap(
        depth_frame, fx, fy, cx, cy,
        max_range_m=UMAP_MAX_RANGE, bin_size=UMAP_BIN_SIZE,
        t_poi=UMAP_T_POI, t_tho=UMAP_T_THO, bin_thresh=UMAP_BIN_THRESH,
        max_obstacles=MAX_DEPTH_OBSTACLES,
    )
    points = []
    for pos_cam, half_w, half_h in detections:
        world_xyz = pos_cam_w + quat_apply(quat_cam_w, pos_cam)
        radius = max(DEPTH_OBSTACLE_RADIUS, half_w, half_h)
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
    from sensor_msgs.msg import Image, CameraInfo

    class DepthUmapNode(Node):
        """No real pose subscription here (unlike eval_crazieflieros2.py), so detections
        are reported camera-relative rather than world-frame -- see module docstring."""

        def __init__(self):
            super().__init__("depth_umap_test")
            self.frame_idx = 0
            if args.ros2_hardcoded_intrinsics:
                self.fx, self.fy, self.cx, self.cy = DEPTH_FX, DEPTH_FY, DEPTH_CX, DEPTH_CY
                camera_info_desc = "<hardcoded>"
            else:
                self.fx = self.fy = self.cx = self.cy = None
                self.create_subscription(CameraInfo, args.ros2_camera_info_topic, self._info_cb, 10)
                camera_info_desc = args.ros2_camera_info_topic
            self.create_subscription(Image, args.ros2_depth_topic, self._depth_cb, 10)
            self.get_logger().info(f"Subscribed: depth={args.ros2_depth_topic}  camera_info={camera_info_desc}")

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

            detections = detect_obstacles_umap(
                depth, self.fx, self.fy, self.cx, self.cy,
                max_range_m=UMAP_MAX_RANGE, bin_size=UMAP_BIN_SIZE,
                t_poi=UMAP_T_POI, t_tho=UMAP_T_THO, bin_thresh=UMAP_BIN_THRESH,
                max_obstacles=MAX_DEPTH_OBSTACLES,
            )
            print(f"\n--- frame {self.frame_idx}  {len(detections)} detection(s) (camera-relative) ---")
            for pos_cam, half_w, half_h in detections:
                radius = max(DEPTH_OBSTACLE_RADIUS, half_w, half_h)
                print(f"  cam=({pos_cam[0]:+.3f},{pos_cam[1]:+.3f},{pos_cam[2]:+.3f})m  r={radius:.2f}")
            self.frame_idx += 1

    rclpy.init()
    node = DepthUmapNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
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

    # ros2 source (subscribes to an already-running publisher instead)
    p.add_argument("--ros2_depth_topic", type=str,
                    default="/camera/camera/aligned_depth_to_color/image_raw")
    p.add_argument("--ros2_camera_info_topic", type=str,
                    default="/camera/camera/aligned_depth_to_color/camera_info")
    p.add_argument("--ros2_hardcoded_intrinsics", action="store_true",
                    help="skip subscribing to --ros2_camera_info_topic and use "
                         "depth_obstacle_estimator.DEPTH_FX/FY/CX/CY directly instead -- for "
                         "testing against the raw (non-aligned) depth topic, e.g. "
                         "/camera/camera/depth/image_raw, without needing its camera_info stream.")

    # isaac source (boots the real Crazyflie env)
    p.add_argument("--n_frames", type=int, default=300, help="episode length, control steps")
    p.add_argument("--isaac_device", type=str, default="cuda:0")
    p.add_argument("--isaac_dt", type=float, default=0.005, help="sim physics dt (s)")
    p.add_argument("--isaac_hover_xy", type=float, nargs=2, default=[0.0, 0.0], metavar=("X", "Y"),
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
