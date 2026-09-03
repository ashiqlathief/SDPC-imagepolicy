import importlib
from collections import deque
from types import SimpleNamespace
import time
import numpy as np
np.set_printoptions(precision=3, suppress=True)
import torch
import diffuser.utils as utils
from diffuser.sampling.projection import Projector
from diffuser.sampling.policies import temporal_consistency_distances
from depth_obstacle_estimator import (
    camera_intrinsics, camera_world_pose, quat_apply, detect_obstacles_umap,
    FPV_WIDTH, FPV_HEIGHT, FPV_FOCAL_LENGTH, FPV_HORIZONTAL_APERTURE,
)
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import PoseStamped, TwistStamped
from sensor_msgs.msg import Image, CameraInfo
cfg = importlib.import_module("config.avoiding-crazyflie")
_IDENTITY_QUAT = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

DEPTH_NEAR = cfg.DEPTH_NEAR
DEPTH_FAR = cfg.DEPTH_FAR
KEEPOUT_ZONES = getattr(cfg, 'KEEPOUT_ZONES', [])   # (x, y, radius) world-frame, planner-only  see config/avoiding-crazyflie.py
FLIGHT_Z_MIN = 0.0   # floor
FLIGHT_Z_MAX = 2.0   # ceiling — real corridor's wall/ceiling height, not the sim's
_Z_HALFSPACES = [
    ([0.0, 0.0,  1.0], FLIGHT_Z_MAX),   # z <= FLIGHT_Z_MAX : drone cannot fly above wall/ceiling
    ([0.0, 0.0, -1.0], FLIGHT_Z_MIN),   # z >= FLIGHT_Z_MIN : drone cannot go underground
]

VARIANT_CFG = {
    "sdpc-r": dict(num_candidates=1, selection="first", use_projection=True),
    "sdpc-c": dict(num_candidates=2, selection="minimum_projection_cost", use_projection=True),
    "sdpc-t": dict(num_candidates=2, selection="temporal_consistency", use_projection=True),
    "diffuser": dict(num_candidates=1, selection="first", use_projection=False),
}

def preprocess_rgb_stack(rgb_hist):
    """
    rgb_hist: list/deque of To frames, each (H,W,3) uint8
    returns torch tensor (1, To, 3, H, W) in [0,1]
    """
    arr = np.stack(rgb_hist, axis=0)  # (To,H,W,3)
    arr = arr.astype(np.float32) / 255.0
    arr = np.transpose(arr, (0, 3, 1, 2))  # (To,3,H,W)
    ten = torch.from_numpy(arr).unsqueeze(0)  # (1,To,3,H,W)
    return ten

def preprocess_obs_stack(obs_hist, use_depth):
    """
    obs_hist: list/deque of To frames, each from Ros2HardwareRunner._grab_frame().
    Returns torch tensor (1, To, 3, H, W) if use_depth=False,
                          (1, To, 4, H, W) if use_depth=True (4th chan = normalized depth),
    matching what CrazyflieImageDataset.__getitem__ builds for "obs_rgb" during training
    (same DEPTH_NEAR/DEPTH_FAR clip-and-minmax normalization).
    """
    if not use_depth:
        return preprocess_rgb_stack(obs_hist)

    rgb_hist = [frame[0] for frame in obs_hist]
    depth_hist = [frame[1] for frame in obs_hist]

    rgb_ten = preprocess_rgb_stack(rgb_hist)  # (1,To,3,H,W)

    depth = np.stack(depth_hist, axis=0)  # (To,H,W,1)
    depth = np.squeeze(depth, axis=-1)  # (To,H,W)
    non_finite = ~np.isfinite(depth)
    if non_finite.any():
        depth = depth.copy()
        depth[non_finite] = DEPTH_FAR
    depth = np.clip(depth, DEPTH_NEAR, DEPTH_FAR)
    depth = (depth - DEPTH_NEAR) / (DEPTH_FAR - DEPTH_NEAR)  # -> [0,1]
    depth_ten = torch.from_numpy(depth[None, :, None, :, :].astype(np.float32))  # (1,To,1,H,W)

    return torch.cat([rgb_ten, depth_ten], dim=2)  # (1,To,4,H,W)

def sample_action(diffusion, cond, horizon, action_dim):
    with torch.no_grad():
        x, _ = diffusion.conditional_sample(cond, horizon=horizon, projector=None)  # (1,H,D)
    return x[0, 0, :action_dim].detach().cpu().numpy()

def sample_action_horizon(diffusion, cond, horizon, action_dim, projector=None, num_candidates=1):
    obs_rgb = cond["obs_rgb"]  # (1,To,3or4,H,W)
    cond_k = {"obs_rgb": obs_rgb.repeat(num_candidates, 1, 1, 1, 1)}
    if "pose_now" in cond:
        cond_k["pose_now"] = cond["pose_now"].repeat(num_candidates, 1)
        cond_k["pose_target"] = cond["pose_target"].repeat(num_candidates, 1)
    with torch.no_grad():
        x, _ = diffusion.conditional_sample(cond_k, horizon=horizon, projector=projector)  # (K,H,D)
    return x[:, :, :action_dim].detach().cpu().numpy()  # (K, H, action_dim)

def detect_depth_obstacles(depth_frame, pos_body_w, quat_body_w, depth_fx, depth_fy, depth_cx, depth_cy,
                            umap_max_range, umap_bin_size, umap_t_poi, umap_t_tho, umap_bin_thresh,
                            depth_obstacle_radius, max_depth_obstacles):
    pos_cam_w, quat_cam_w = camera_world_pose(pos_body_w, quat_body_w)

    detections = detect_obstacles_umap(
        depth_frame, depth_fx, depth_fy, depth_cx, depth_cy,
        max_range_m=umap_max_range, bin_size=umap_bin_size,
        t_poi=umap_t_poi, t_tho=umap_t_tho, bin_thresh=umap_bin_thresh,
        max_obstacles=max_depth_obstacles,
    )
    points = []
    for pos_cam, half_w, half_h in detections:
        world_xyz = pos_cam_w + quat_apply(quat_cam_w, pos_cam)
        radius = max(depth_obstacle_radius, half_w, half_h)
        points.append((float(world_xyz[0]), float(world_xyz[1]), float(radius)))
    return points

def build_projector(horizon_H, device, static_points, drone_radius=0.0,
                     pos0=None, action_normalizer=None,
                     proj_tighten=0.15, proj_dt=0.1,
                     x_bounds=(-5.5, 4.5), y_bounds=(-1.95, 1.95),
                     keepout_zones=None):
    lb = np.array([x_bounds[0], y_bounds[0], FLIGHT_Z_MIN], dtype=np.float32)
    ub = np.array([x_bounds[1], y_bounds[1], FLIGHT_Z_MAX], dtype=np.float32)
    constraint_list = [("lb", lb), ("ub", ub)]

    for (x, y, r) in static_points:
        radius = r + drone_radius + proj_tighten
        constraint_list.append(("sphere_outside", [0, 1], [float(x), float(y)], float(radius)))
    for (x, y, zone_radius) in (keepout_zones or []):
        radius = float(zone_radius) + drone_radius + proj_tighten
        constraint_list.append(("sphere_outside", [0, 1], [float(x), float(y)], radius))
    for normal, rhs in _Z_HALFSPACES:
        constraint_list.append(("ineq", (np.array(normal, dtype=np.float32), float(rhs))))

    projector = Projector(
        horizon=horizon_H + 1, transition_dim=3, action_dim=0, goal_dim=0,
        constraint_list=constraint_list, normalizer=None, gradient=False,
        gradient_weights=[1, 0.5, 2], dt=proj_dt, variant="states",
        skip_initial_state=True, diffusion_timestep_threshold=0.8,
        device=str(device), solver="scipy", parallelize=True, goal_pull_weight=0.0,
    )
    if pos0 is not None:
        projector.inloop_slsqp = True
        projector.action_normalizer = action_normalizer
        projector.pos0 = pos0
    return projector

def project_deltas_from_pos(projector, pos0, deltas_real, device):

    K, H, _ = deltas_real.shape
    pos_traj = np.zeros((K, H + 1, 3), dtype=np.float32)
    pos_traj[:, 0] = pos0.astype(np.float32)[None, :]
    pos_traj[:, 1:] = pos_traj[:, :1] + np.cumsum(deltas_real, axis=1)

    state_t = torch.tensor(pos_traj, dtype=torch.float32, device=device)
    state_proj_t, proj_costs = projector.project(state_t)  # (K,H+1,3), (K,)
    pos_proj = state_proj_t.detach().cpu().numpy()
    proj_deltas = (pos_proj[:, 1:] - pos_proj[:, :-1]).astype(np.float32)
    return proj_deltas, proj_costs.astype(np.float32)

def choose_candidate(a_horizon_real, proj_costs, prev_actions_real, strategy):
    K = a_horizon_real.shape[0]
    if K == 1 or strategy == "first":
        return 0
    if strategy == "minimum_projection_cost" and proj_costs is not None:
        return int(np.argmin(proj_costs))
    if strategy == "temporal_consistency" and prev_actions_real is not None:
        dists = temporal_consistency_distances(a_horizon_real, prev_actions_real[None, :, :])
        return int(np.argmin(dists))
    return 0

IMAGE_SPECS = {
    # topic (see COLOR_TOPIC/DEPTH_TOPIC below): sensor_msgs/msg/Image
    "color": dict(encoding="rgb8",  dtype=np.uint8,  channels=3),   # /camera/camera/color/image_raw
    "depth": dict(encoding="16UC1", dtype=np.uint16, channels=1),   # /camera/camera/depth/image_rect_raw (raw mm)
}

class Ros2HardwareRunner(Node):
    RUN_DIR = None  # <-- REQUIRED: set to your trained run's checkpoint dir before running.
    VARIANT = "diffuser"  # one of VARIANT_CFG's keys -- single projection variant to fly (no sweep on real hardware)
    POSE_TOPIC = "/mavros/local_position/pose"
    CMD_VEL_TOPIC = "/mpc/set_pose"
    COLOR_TOPIC = "camera/camera/color/image_raw"
    DEPTH_TOPIC = "/camera/camera/depth/image_raw"
    DEPTH_CAMERA_INFO_TOPIC = "/camera/camera/depth/camera_info"
    START_DELAY = 5.0  # seconds to wait for the first pose/camera message before giving up
    TARGET_X = 4.0
    TARGET_Y = 0.75

    DEPTH_OBSTACLE_RADIUS = 0.3       # floor for the detected obstacle's keep-out radius (m), on top of drone_radius
    MAX_DEPTH_OBSTACLES = 5
    UMAP_MAX_RANGE = 3.0              # depth range covered by the U-map histogram (m)
    UMAP_BIN_SIZE = 200               # number of depth bins across UMAP_MAX_RANGE
    UMAP_T_POI = 500.0                # point-of-interest threshold
    UMAP_T_THO = 1800.0               # U-map contour threshold
    UMAP_BIN_THRESH = 150
    PROJ_TIGHTEN = 0.15               # extra margin on top of the detected radius + drone_radius
    PROJ_DT = 0.1
    DEVICE = "cuda:0"
    DRONE_RADIUS = 0.1
    CONTROL_HZ = 30

    def __init__(self):
        super().__init__("diffusion_policy_hardware")

        if self.RUN_DIR is None:
            raise ValueError("Set Ros2HardwareRunner.RUN_DIR to your trained run's checkpoint dir before running.")
        if self.VARIANT not in VARIANT_CFG:
            raise ValueError(f"VARIANT={self.VARIANT!r} must be one of {list(VARIANT_CFG)}")

        # Kept as a SimpleNamespace (not scattered self.xxx reads) purely so
        # every method below that already reads self.args.xxx / args.xxx needed
        # no further changes when CLI args were replaced by class attributes.
        self.args = SimpleNamespace(
            run_dir=self.RUN_DIR, variant=self.VARIANT, pose_topic=self.POSE_TOPIC,
            cmd_vel_topic=self.CMD_VEL_TOPIC, color_topic=self.COLOR_TOPIC,
            depth_topic=self.DEPTH_TOPIC, depth_camera_info_topic=self.DEPTH_CAMERA_INFO_TOPIC,
            start_delay=self.START_DELAY, target_x=self.TARGET_X, target_y=self.TARGET_Y,
        )
        args = self.args
        run_dir = self.RUN_DIR
        device = torch.device(self.DEVICE)
        drone_radius = self.DRONE_RADIUS

        self.run_dir = run_dir
        self.drone_radius = drone_radius

        # ------------------ Load trained model (single run_dir, no sweep) ------------------
        print(f"\n[INFO] Loading run dir: {run_dir}")
        diff_exp = utils.load_diffusion(run_dir, epoch="best", device=str(device))
        self.dataset = diff_exp.dataset
        self.diffusion = diff_exp.diffusion.to(device)
        self.diffusion.eval()
        self.device = device

        self.use_depth = bool(getattr(self.diffusion.model, "use_depth", False))
        self.horizon = int(getattr(self.diffusion, "horizon", 16))
        self.action_dim = int(getattr(self.diffusion, "action_dim", 3))
        self.To = int(getattr(self.dataset, "n_obs_steps", 2))

        # ------------------ Pose conditioning (same as eval_crazieflie1pos.py) ------------------
        # No Isaac Sim env here, so there's no env.cfg.gate_x_max -- use TARGET_X
        # instead (defaults to the sim's gate_x_max=4.0, see crazyflie_envpos.py).
        self.use_pose_cond = bool(getattr(self.dataset, "use_pose_cond", False))
        self.pose_target_world = None
        if self.use_pose_cond:
            self.pose_target_world = np.array(
                [args.target_x, args.target_y, 1.0], dtype=np.float32
            )
            print(f"[INFO] Pose-conditioned model: fixed goal for this run = {np.round(self.pose_target_world, 3).tolist()}")

        self.vcfg = VARIANT_CFG[args.variant]
        self.depth_static_pts_latest = []  # latest detection, kept only for logging/inspection

        self.depth_fx, self.depth_fy, self.depth_cx, self.depth_cy = camera_intrinsics(
            FPV_WIDTH, FPV_HEIGHT, FPV_FOCAL_LENGTH, FPV_HORIZONTAL_APERTURE
        )
        self._depth_intrinsics_ready = False

        print(f"[INFO] Variant: {args.variant} (num_candidates={self.vcfg['num_candidates']}, "
              f"selection={self.vcfg['selection']}, use_projection={self.vcfg['use_projection']})")

        # ------------------ Subscriptions/publisher (self IS the node -- rclpy.init()
        # + node construction happen in main(), before this class exists) ------------
        self.cam_state = {"color": None, "depth": None}
        self.need_depth_frame = self.use_depth or self.vcfg["use_projection"]
        self.create_subscription(Image, args.color_topic, self._color_cb, qos_profile_sensor_data)
        if self.need_depth_frame:
            self.create_subscription(Image, args.depth_topic, self._depth_cb, qos_profile_sensor_data)
        if self.vcfg["use_projection"]:
            self.create_subscription(
                CameraInfo, args.depth_camera_info_topic, self._depth_info_cb, qos_profile_sensor_data
            )

        self.state = {"pos": None, "quat": _IDENTITY_QUAT.copy()}
        self.create_subscription(PoseStamped, args.pose_topic, self._pose_cb, qos_profile_sensor_data)
        self.cmd_pub = self.create_publisher(PoseStamped, args.cmd_vel_topic, 1)

        self._t0 = time.time()
        self._rgb_hist = None   # still None == warmup; set once by _enter_running()
        self._prev_actions_real = None
        self._step = 0
        self._cam_topics_desc = args.color_topic + (f" + {args.depth_topic}" if self.need_depth_frame else "")
        if self.vcfg["use_projection"]:
            self._cam_topics_desc += f" + {args.depth_camera_info_topic}"
        print(f"[INFO] Waiting up to {args.start_delay}s for pose ({args.pose_topic}) "
              f"and camera ({self._cam_topics_desc}) ...")
        self.timer = self.create_timer(1.0 / self.CONTROL_HZ, self._tick)

    # ------------------ ROS2 subscription callbacks ------------------
    def _center_crop(self, img):
        """Center-crop to (FPV_HEIGHT, FPV_WIDTH) -- no resize, so pixel scale
        stays 1:1 with the source instead of being stretched/squished like a
        plain cv2.resize would. Source must be >= 96x96 in both dims (true for
        any real color/depth topic here)."""
        h, w = img.shape[:2]
        if h < FPV_HEIGHT or w < FPV_WIDTH:
            raise ValueError(f"frame {w}x{h} smaller than crop target {FPV_WIDTH}x{FPV_HEIGHT}")
        y0 = (h - FPV_HEIGHT) // 2
        x0 = (w - FPV_WIDTH) // 2
        return img[y0:y0 + FPV_HEIGHT, x0:x0 + FPV_WIDTH]

    def _decode_image(self, msg, spec):
        """Shared decode for _color_cb/_depth_cb, driven by IMAGE_SPECS instead
        of an if/elif per possible encoding -- returns None (after logging) if
        this message doesn't match the one real encoding that topic ever sends."""
        if msg.encoding != spec["encoding"]:
            self.get_logger().warning(
                f"expected '{spec['encoding']}' encoding, got '{msg.encoding}', skipping frame",
                throttle_duration_sec=5.0
            )
            return None
        shape = (msg.height, msg.width, spec["channels"]) if spec["channels"] > 1 else (msg.height, msg.width)
        return np.frombuffer(msg.data, dtype=spec["dtype"]).reshape(shape)

    def _color_cb(self, msg):
        frame = self._decode_image(msg, IMAGE_SPECS["color"])
        if frame is not None:
            self.cam_state["color"] = self._center_crop(frame)

    def _depth_cb(self, msg):
        depth = self._decode_image(msg, IMAGE_SPECS["depth"])
        if depth is not None:
            depth = depth.astype(np.float32) * 0.001  # RealSense: raw mm -> metres
            self.cam_state["depth"] = depth

    def _depth_info_cb(self, msg):
        """Real depth-sensor intrinsics (fx/fy/cx/cy from K), NOT the simulated
        FPV camera's -- backprojection needs the actual calibration. Depth is
        no longer cropped (see _depth_cb), so cx/cy are used as-is, uncorrected."""
        self.depth_fx, self.depth_fy = float(msg.k[0]), float(msg.k[4])
        self.depth_cx = float(msg.k[2])
        self.depth_cy = float(msg.k[5])
        self._depth_intrinsics_ready = True

    def _pose_cb(self, msg):
        p = msg.pose.position
        o = msg.pose.orientation
        print('pos:', p)
        self.state["pos"] = np.array([p.x, p.y, p.z], dtype=np.float32)
        self.state["quat"] = np.array([o.w, o.x, o.y, o.z], dtype=np.float32)

    # ------------------ small helpers ------------------
    def _cam_ready(self):
        depth_ok = not self.need_depth_frame or self.cam_state["depth"] is not None
        intrinsics_ok = not self.vcfg["use_projection"] or self._depth_intrinsics_ready
        return self.cam_state["color"] is not None and depth_ok and intrinsics_ok

    def _grab_frame(self):
        if not self.use_depth:
            return self.cam_state["color"]
        return self.cam_state["color"], self.cam_state["depth"]

    def _publish_stop(self):
        """Hold the drone's last known position -- NOT a TwistStamped zero
        (cmd_pub is a PoseStamped publisher; publishing a TwistStamped on it
        raises a TypeError) and NOT an all-zero pose (that would command a
        flight to the origin instead of stopping in place)."""
        if not rclpy.ok():
            return
        pos = self.state.get("pos")
        if pos is None:
            return  # never got a pose fix -- nothing safe to hold to
        stop = PoseStamped()
        stop.header.stamp = self.get_clock().now().to_msg()
        stop.pose.position.x = float(pos[0])
        stop.pose.position.y = float(pos[1])
        stop.pose.position.z = float(pos[2])
        self.cmd_pub.publish(stop)

    # ------------------ timer-driven state machine (replaces run()) ------------------
    def _tick(self):
        """The one and only recurring callback -- serviced by main()'s single
        rclpy.spin(self), same executor that also services every subscription
        callback above. No manual spin_once() polling anywhere in this class.

        No explicit phase flag: _rgb_hist is None until _enter_running() sets
        it, so that alone distinguishes warmup from running. Nothing to check
        for "aborted" either -- _abort() cancels the timer, so this simply
        never gets called again after that."""
        if self._rgb_hist is None:
            self._warmup_tick()
        else:
            self._control_step()

    def _warmup_tick(self):
        args = self.args
        elapsed = time.time() - self._t0
        if self.state["pos"] is None:
            if elapsed > args.start_delay:
                self.get_logger().error(
                    f"No pose received on {args.pose_topic} within {args.start_delay}s -- "
                    "check mavros is running and the topic name."
                )
                self._abort()
            return
        if not self._cam_ready():
            if elapsed > args.start_delay:
                self.get_logger().error(
                    f"No camera frame received on {self._cam_topics_desc} within {args.start_delay}s -- "
                    "check the camera driver is running and the topic names."
                )
                self._abort()
            return
        self._enter_running()

    def _abort(self):
        self.timer.cancel()
        rclpy.shutdown()   # makes main()'s rclpy.spin(self) return

    def _enter_running(self):
        # init obs history (same seeding pattern as sim: repeat the first real frame To times)
        frame0 = self._grab_frame()
        self._rgb_hist = deque(maxlen=self.To)
        for _ in range(self.To):
            self._rgb_hist.append((frame0[0].copy(), frame0[1].copy()) if self.use_depth else frame0.copy())

        self._prev_actions_real = None
        self._step = 0
        print(f"[INFO] Starting control loop at {self.CONTROL_HZ} Hz Ctrl+C to stop.")

    def _control_step(self):
        vcfg = self.vcfg
        use_depth = self.use_depth
        device = self.device

        pos = self.state["pos"].copy()

        obs_rgb_t = preprocess_obs_stack(self._rgb_hist, use_depth).to(device)
        cond = {"obs_rgb": obs_rgb_t}

        if self.use_pose_cond:
            pose_now_norm = self.dataset.pose_normalizer.normalize(pos[:3].astype(np.float32))
            pose_target_norm = self.dataset.pose_normalizer.normalize(self.pose_target_world)
            cond["pose_now"] = torch.from_numpy(pose_now_norm).float().unsqueeze(0).to(device)
            cond["pose_target"] = torch.from_numpy(pose_target_norm).float().unsqueeze(0).to(device)

        if vcfg["use_projection"]:
            static_pts = detect_depth_obstacles(
                self.cam_state["depth"], pos, self.state["quat"],
                self.depth_fx, self.depth_fy, self.depth_cx, self.depth_cy,
                self.UMAP_MAX_RANGE, self.UMAP_BIN_SIZE, self.UMAP_T_POI, self.UMAP_T_THO,
                self.UMAP_BIN_THRESH, self.DEPTH_OBSTACLE_RADIUS, self.MAX_DEPTH_OBSTACLES,
            )
            self.depth_static_pts_latest = static_pts
            print(f"[OBSTACLES] static_pts={[tuple(round(v, 3) for v in p) for p in static_pts]}")
            projector = build_projector(
                self.horizon, device, static_pts, self.drone_radius,
                pos0=pos[:3], action_normalizer=self.dataset.action_normalizer,
                proj_tighten=self.PROJ_TIGHTEN, proj_dt=self.PROJ_DT,
                keepout_zones=KEEPOUT_ZONES,
            )

            _infer_start = time.time()
            a_horizon_norm = sample_action_horizon(self.diffusion, cond, self.horizon, self.action_dim,
                                                    projector=projector, num_candidates=vcfg["num_candidates"])
            inference_time = time.time() - _infer_start

            a_horizon_real = self.dataset.action_normalizer.unnormalize(a_horizon_norm)  # (K,H,D)
            a_horizon_real[:, :, :3], proj_costs = project_deltas_from_pos(
                projector, pos[:3], a_horizon_real[:, :, :3], device
            )
            choice = choose_candidate(a_horizon_real, proj_costs, self._prev_actions_real, vcfg["selection"])
            a0_real = a_horizon_real[choice, 0]
            self._prev_actions_real = a_horizon_real[choice]
        else:
            _infer_start = time.time()
            a0_norm = sample_action(self.diffusion, cond, self.horizon, self.action_dim)
            inference_time = time.time() - _infer_start
            a0_real = self.dataset.action_normalizer.unnormalize(a0_norm)

        print(f"[INFERENCE] step={self._step} time={inference_time:.1f}s")

        pos_cmd = pos.copy()
        inc_action = a0_real[:self.action_dim]
        inc_action = np.clip(inc_action, [-0.5, -0.5, -0.1], [0.5, 0.5, 0.1])
        pos_cmd[:self.action_dim] = pos + inc_action
        pos_cmd[2] = np.clip(pos_cmd[2], FLIGHT_Z_MIN, FLIGHT_Z_MAX)

        cmd_pos = PoseStamped()
        cmd_pos.header.stamp = self.get_clock().now().to_msg()
        cmd_pos.pose.position.x = float(pos_cmd[0])
        cmd_pos.pose.position.y = float(pos_cmd[1])
        cmd_pos.pose.position.z = float(pos_cmd[2])

        print(f"[step {self._step}] pos={pos} a0_real={a0_real} pos_cmd={pos_cmd}")

        self.cmd_pub.publish(cmd_pos)

        self._rgb_hist.append(self._grab_frame())
        self._step += 1

    def finalize(self):
        """Called once from main()'s finally block (Ctrl+C, timeout abort, or
        clean exit) -- publishes a zero-velocity stop. Replaces run()'s
        try/finally, since there's no single blocking call left to wrap now
        that control happens in _tick()."""
        if hasattr(self, "timer"):
            self.timer.cancel()
        self._publish_stop()
        self.get_logger().info("Hardware run stopped, zero velocity published.")

def main():
    rclpy.init()
    node = Ros2HardwareRunner()
    rclpy.spin(node)
    node.finalize()
    node.destroy_node()
    rclpy.try_shutdown()

if __name__ == "__main__":
    main()
