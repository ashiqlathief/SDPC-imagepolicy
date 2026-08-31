import argparse
import importlib
import os
import re
from collections import deque
from pathlib import Path
import time
import numpy as np
import torch

import diffuser.utils as utils
from diffuser.sampling.projection import Projector
from diffuser.sampling.policies import temporal_consistency_distances
from depth_obstacle_estimator import (
    camera_intrinsics, camera_world_pose, backproject_depth_to_world, quat_apply,
    filter_points, cluster_points, ObstacleTracker, tracks_to_constraints,
    keep_nearest_along_z,
    FPV_WIDTH, FPV_HEIGHT, FPV_FOCAL_LENGTH, FPV_HORIZONTAL_APERTURE,
)
import rclpy
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import PoseStamped, TwistStamped
from sensor_msgs.msg import Image
cfg = importlib.import_module("config.avoiding-crazyflie")
CYLINDERS = cfg.CYLINDERS
KEEPOUT_ZONES = getattr(cfg, 'KEEPOUT_ZONES', [])   # (x, y, radius) world-frame, planner-only  see config/avoiding-crazyflie.py
_IDENTITY_POS = np.zeros(3, dtype=np.float32)
_IDENTITY_QUAT = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

corridor_halfspaces = cfg.CORRIDOR_HALFSPACES
DEPTH_NEAR = cfg.DEPTH_NEAR
DEPTH_FAR = cfg.DEPTH_FAR

CYL_RADIUS = 0.20
CYL_RADIUS_PROJECTOR_PAD = 0.01

FLIGHT_Z_MIN = 0.0   # floor
FLIGHT_Z_MAX = 2.0   # ceiling — real corridor's wall/ceiling height, not the sim's

z_halfspaces = [
    ([0.0, 0.0,  1.0], FLIGHT_Z_MAX),   # z <= FLIGHT_Z_MAX : drone cannot fly above wall/ceiling
    ([0.0, 0.0, -1.0], FLIGHT_Z_MIN),   # z >= FLIGHT_Z_MIN : drone cannot go underground
]

projection_variants = [
  'sdpc-r',
#   'sdpc-r-tightened',
  'sdpc-c',
#   'sdpc-c-tightened',
  'sdpc-t',
#   'sdpc-t-tightened',
  'diffuser',
  'gradient',
  'gradient-tightened',
  'post_processing',
  'post_processing-tightened',
]

def variant_cfg(name: str):
    cfg = dict(
        num_candidates=1,
        selection="first",
        use_projection=False,
        projection_mode="none",    # "none" | "post" | "gradient"
        tighten=0.0,
    )

    if name == "diffuser":
        cfg.update(num_candidates=1, selection="first", use_projection=False)

    elif name == "post_processing":
        cfg.update(num_candidates=1, selection="first", use_projection=True, projection_mode="post", tighten=0.0)

    elif name == "post_processing-tightened":
        cfg.update(num_candidates=1, selection="first", use_projection=True, projection_mode="post", tighten=0.05)

    elif name == "gradient":
        cfg.update(num_candidates=1, selection="first", use_projection=True, projection_mode="gradient", tighten=0.0)

    elif name == "gradient-tightened":
        cfg.update(num_candidates=1, selection="first", use_projection=True, projection_mode="gradient", tighten=0.05)

    elif name == "sdpc-c":
        cfg.update(num_candidates=2, selection="minimum_projection_cost", use_projection=True, projection_mode="sdpc", tighten=0.0)

    elif name == "sdpc-c-tightened":
        cfg.update(num_candidates=2, selection="minimum_projection_cost", use_projection=True, projection_mode="sdpc", tighten=0.05)

    elif name == "sdpc-t":
        cfg.update(num_candidates=2, selection="temporal_consistency", use_projection=True, projection_mode="sdpc", tighten=0.0)

    elif name == "sdpc-t-tightened":
        cfg.update(num_candidates=2, selection="temporal_consistency", use_projection=True, projection_mode="sdpc", tighten=0.05)

    elif name == "sdpc-r":
        # sdpc-r often means single sample with projection (repair)
        cfg.update(num_candidates=1, selection="first", use_projection=True, projection_mode="sdpc", tighten=0.0)

    elif name == "sdpc-r-tightened":
        cfg.update(num_candidates=1, selection="first", use_projection=True, projection_mode="sdpc", tighten=0.05)

    return cfg

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

def sample_action_candidates(diffusion, cond, horizon, action_dim, num_candidates, projector = None):
    """
    Returns:
      a_candidates_norm: (K, H, action_dim) numpy
      infos: dict
    """
    # Repeat condition across batch to get K samples
    obs_rgb = cond["obs_rgb"]  # (1,To,3,H,W)
    cond_k = {"obs_rgb": obs_rgb.repeat(num_candidates, 1, 1, 1, 1)}  # (K,To,3,H,W)
    if "pose_now" in cond:
        cond_k["pose_now"] = cond["pose_now"].repeat(num_candidates, 1)        # (K,3)
        cond_k["pose_target"] = cond["pose_target"].repeat(num_candidates, 1)  # (K,3)

    with torch.no_grad():
        x, infos = diffusion.conditional_sample(cond_k, horizon=horizon,projector=projector)  # x: (K,H,D)

    x = x[:, :, :action_dim]   # (K,H,action_dim)
    return x.detach().cpu().numpy(), infos

def choose_trajectory(actions_real, strategy="first", prev_actions_real=None):
    """
    actions_real: (K, H, D)
    prev_actions_real: (1, H, D) or None
    """
    K = actions_real.shape[0]

    if K == 1 or strategy == "first":
        return 0

    if strategy == "temporal_consistency" and prev_actions_real is not None:
        # compare shifted action chunks
        dists = temporal_consistency_distances(actions_real, prev_actions_real)
        return int(np.argmin(dists))

    # "minimum_projection_cost" is handled by the caller directly (it already has
    # the post-hoc proj_costs in hand before calling this function), so it never
    # reaches here.
    return 0

def build_obstacle_constraint_list(cylinders, x_bounds=None,
                                    y_bounds=None, z_bounds=None,
                                    corridor_halfspaces=None, z_halfspaces=None, tighten=0.0,
                                    cyl_extra_radius=0.0, drone_radius=0.0,
                                    dynamic_cylinder_predictions=None, keepout_zones=None):
    """
    dynamic_cylinder_predictions: optional {cylinder_index: [(x,y), ...]}: one
    predicted (x,y) per horizon step (in order), from env.predict_cylinder_positions().
    Cylinders with an entry here get a per-timestep obstacle constraint (the
    projector avoids where the obstacle WILL be at each planned step); all others
    keep the usual single current-position constraint applied across the horizon.

    keepout_zones: optional [(x, y, radius), ...] world-frame virtual no-fly zones
    (see KEEPOUT_ZONES in config/avoiding-crazyflie.py)  planner-only, no physical
    object, no hard collision-fail; just another circular exclusion like cylinders.
    """
    constraint_list = []
    state_dim = 3

    # ---------------- bounds ----------------
    lb = np.full(state_dim, -np.inf, dtype=np.float32)
    ub = np.full(state_dim,  np.inf, dtype=np.float32)

    if x_bounds is not None:
        lb[0], ub[0] = float(x_bounds[0]), float(x_bounds[1])
    if y_bounds is not None:
        lb[1], ub[1] = float(y_bounds[0]), float(y_bounds[1])
    if z_bounds is not None:
        lb[2], ub[2] = float(z_bounds[0]), float(z_bounds[1])

    constraint_list.append(("lb", lb))
    constraint_list.append(("ub", ub))

    # ---------------- obstacles ----------------
    # drone_radius is the Minkowski expansion: treat the drone as a point but
    # expand every obstacle by the drone's bounding circle radius so the
    # constraint ||p_drone - p_obs|| >= r_obs + drone_radius is exact.
    _dr = float(drone_radius)
    for i, (x, y) in enumerate(cylinders):
        radius = CYL_RADIUS + CYL_RADIUS_PROJECTOR_PAD + _dr + float(tighten) + float(cyl_extra_radius)
        if dynamic_cylinder_predictions is not None and i in dynamic_cylinder_predictions:
            centers_per_t = [[float(cx), float(cy)] for cx, cy in dynamic_cylinder_predictions[i]]
            constraint_list.append(("sphere_outside_dynamic", [0, 1], centers_per_t, radius))
        else:
            center = [float(x), float(y)]
            constraint_list.append(("sphere_outside", [0, 1], center, radius))

    for (x, y, zone_radius) in (keepout_zones or []):
        radius = float(zone_radius) + _dr + float(tighten)
        constraint_list.append(("sphere_outside", [0, 1], [float(x), float(y)], radius))

    # ---------------- halfspaces in XY ----------------
    if corridor_halfspaces:
        _trajectory_dim    = state_dim
        _act_obs_indices   = {"x": 0, "y": 1, "z": 2}
        for hs in corridor_halfspaces:
            C_row, d = utils.formulate_halfspace_constraints(
                hs,
                enlarge_constraints=0.025 + float(tighten),
                trajectory_dim=_trajectory_dim,
                act_obs_indices=_act_obs_indices,
            )
            constraint_list.append(("ineq", (C_row.astype(np.float32), float(d))))

    # ---------------- z halfspace envelope ----------------
    # Each entry: (normal_3d, rhs)  →  normal · [x, y, z] <= rhs - tighten
    # Ceiling: [0,0,1]·p <= WALL_HEIGHT - tighten  (shrinks ceiling down)
    # Floor:   [0,0,-1]·p <= 0.0 - tighten        (shrinks floor up: use tighten=0 here)
    if z_halfspaces is not None:
        for (normal, rhs) in z_halfspaces:
            C_row = np.zeros(state_dim, dtype=np.float32)
            C_row[:3] = normal
            constraint_list.append(("ineq", (C_row, float(rhs) - float(tighten))))

    return constraint_list

def build_position_projector(horizon_H, gradient, device, cylinders,
                             normalizer=None, tighten=0.0, dt=0.1, use_dynamics=True,
                             obs_amplitude=0.0, drone_radius=0.0,
                             active_halfspaces=None, dynamic_cylinder_predictions=None,
                             keepout_zones=None, goal_pull_weight=0.0):
    # We project POSITIONS, so we need horizon = H+1
    Hp1 = horizon_H + 1
    x_bounds = (-0.5, 4.5)
    y_bounds = (-0.95, 0.95)
    z_bounds = (0.0, 1.0)
    transition_dim = 3

    constraint_list = build_obstacle_constraint_list(
        cylinders=cylinders,
        x_bounds=x_bounds,
        y_bounds=y_bounds,
        z_bounds=z_bounds,
        corridor_halfspaces=active_halfspaces or [],
        z_halfspaces=z_halfspaces,
        tighten=tighten,
        cyl_extra_radius=obs_amplitude,
        drone_radius=drone_radius,
        dynamic_cylinder_predictions=dynamic_cylinder_predictions,
        keepout_zones=KEEPOUT_ZONES if keepout_zones is None else keepout_zones,
    )

    projector = Projector(
        horizon=Hp1,
        transition_dim=transition_dim,   # x,y,z positions
        action_dim=0,
        goal_dim=0,
        constraint_list=constraint_list,
        normalizer=normalizer,          # start without normalizer for debugging
        gradient=gradient,
        gradient_weights=[1, 0.5, 2],
        dt=dt if use_dynamics else 0.0,
        variant="states",
        skip_initial_state=True,
        diffusion_timestep_threshold=0.8,
        device=str(device),
        solver="scipy",           # your file uses scipy SLSQP path
        parallelize=True,         # candidates solve independently: run them concurrently
        goal_pull_weight=goal_pull_weight,
    )

    return projector

def build_point_obstacle_constraints(static_points, dynamic_predictions, radius):
    """depth_obstacles counterpart to the cylinder loop inside
    build_obstacle_constraint_list(): same ("sphere_outside", ...) /
    ("sphere_outside_dynamic", ...) tuple shapes, one entry per detected
    surface point instead of one per fitted circle (see
    depth_obstacle_estimator.py's module docstring "option b")."""
    constraints = []
    for (x, y) in static_points:
        constraints.append(("sphere_outside", [0, 1], [float(x), float(y)], float(radius)))
    for centers_per_t in dynamic_predictions.values():
        constraints.append(("sphere_outside_dynamic", [0, 1],
                             [[float(cx), float(cy)] for cx, cy in centers_per_t],
                             float(radius)))
    return constraints

def build_position_projector_from_points(horizon_H, gradient, device,
                                          static_points, dynamic_predictions, point_radius,
                                          normalizer=None, tighten=0.0, dt=0.1,
                                          use_dynamics=True, drone_radius=0.0,
                                          active_halfspaces=None, keepout_zones=None,
                                          goal_pull_weight=0.0):
    """depth_obstacles counterpart to build_position_projector(): identical
    bounds/halfspace setup (reuses build_obstacle_constraint_list unchanged,
    with cylinders=[] so no ground-truth cylinder constraints get added), but
    cylinder obstacles come from a detected point cloud (see
    depth_obstacle_estimator.py) instead of CYLINDERS.
    """
    Hp1 = horizon_H + 1
    x_bounds = (-0.5, 4.5)
    y_bounds = (-0.95, 0.95)
    z_bounds = (0.0, 1.0)
    transition_dim = 3

    constraint_list = build_obstacle_constraint_list(
        cylinders=[],
        x_bounds=x_bounds,
        y_bounds=y_bounds,
        z_bounds=z_bounds,
        corridor_halfspaces=active_halfspaces or [],
        z_halfspaces=z_halfspaces,
        tighten=tighten,
        drone_radius=drone_radius,
        keepout_zones=KEEPOUT_ZONES if keepout_zones is None else keepout_zones,
    )
    constraint_list += build_point_obstacle_constraints(
        static_points, dynamic_predictions,
        radius=point_radius + drone_radius + tighten,
    )

    projector = Projector(
        horizon=Hp1,
        transition_dim=transition_dim,
        action_dim=0,
        goal_dim=0,
        constraint_list=constraint_list,
        normalizer=normalizer,
        gradient=gradient,
        gradient_weights=[1, 0.5, 2],
        dt=dt if use_dynamics else 0.0,
        variant="states",
        skip_initial_state=True,
        diffusion_timestep_threshold=0.8,
        device=str(device),
        solver="scipy",
        parallelize=True,
        goal_pull_weight=goal_pull_weight,
    )
    return projector

def project_action_candidates_with_positions(projector, pos0, a_candidates_real, device):
    """
    pos0: (3,) current position
    a_candidates_real: (K,H,3) delta-pos
    Returns:
      a_proj_real: (K,H,3)
      proj_costs: (K,)
    """
    K, H, _ = a_candidates_real.shape

    # integrate deltas -> positions (K,H+1,3), all K candidates share pos0
    pos_traj = np.zeros((K, H + 1, 3), dtype=np.float32)
    pos_traj[:, 0] = pos0.astype(np.float32)
    pos_traj[:, 1:] = pos_traj[:, :1] + np.cumsum(a_candidates_real, axis=1)

    state_t = torch.tensor(pos_traj, dtype=torch.float32, device=device)  # (K,H+1,transition_dim)

    # one batched call instead of K single-candidate calls — projector.project()
    # already accepts a batched input, and building it once avoids rebuilding the
    # (identical) constraint set K times.
    state_proj_t, proj_costs = projector.project(state_t)  # (K,H+1,transition_dim), cost shape (K,)
    state_proj = state_proj_t.detach().cpu().numpy()
    pos_proj = state_proj[..., :3]  # (K,H+1,3)

    # convert back to deltas (K,H,3)
    a_proj_real = (pos_proj[:, 1:] - pos_proj[:, :-1]).astype(np.float32)
    proj_costs = proj_costs.astype(np.float32)

    return a_proj_real, proj_costs

class Ros2HardwareRunner:

    def __init__(self, args, run_dir, device, drone_radius, active_halfspaces, goal_pull_weight=0.0):

        # stash the rclpy/message modules so the callbacks and run() can use them
        self.rclpy = rclpy
        self.PoseStamped = PoseStamped
        self.TwistStamped = TwistStamped

        self.args = args
        self.run_dir = run_dir
        self.drone_radius = drone_radius
        self.active_halfspaces = active_halfspaces
        self.goal_pull_weight = goal_pull_weight

        # ------------------ Load trained model (single run_dir, no sweep) ------------------
        print(f"\n[INFO] Loading run dir: {run_dir}")
        diff_exp = utils.load_diffusion(run_dir, epoch="best", device=str(device))
        self.dataset = diff_exp.dataset
        self.diffusion = diff_exp.diffusion.to(device)
        self.diffusion.eval()
        self.device = device

        # encoder_type/latent_dim: same folder-naming convention as eval_crazieflie1.py
        # (e.g. ".../H16_K20_ENCvit_LAT256/9"), just for labeling the saved .npz.
        run_name = Path(run_dir).parent.name
        enc_match = re.search(r'E([^_]+)', run_name)
        lat_match = re.search(r'L(\d+)', run_name)
        self.encoder_type = enc_match.group(1) if enc_match else "unknown"
        self.latent_dim = int(lat_match.group(1)) if lat_match else 256

        self.use_depth = bool(getattr(self.diffusion.model, "use_depth", False))
        self.horizon = int(getattr(self.diffusion, "horizon", 16))
        self.action_dim = int(getattr(self.diffusion, "action_dim", 3))
        self.To = int(getattr(self.dataset, "n_obs_steps", 2))

        # ------------------ Pose conditioning (same as eval_crazieflie1.py) ------------------
        # No Isaac Sim env here, so there's no env.cfg.gate_x_max -- use --target_x
        # instead (defaults to the sim's gate_x_max=4.0, see crazyflie_env.py).
        self.use_pose_cond = bool(getattr(self.dataset, "use_pose_cond", False))
        self.pose_target_world = None
        if self.use_pose_cond:
            self.pose_target_world = np.array(
                [args.target_x, args.target_y, 1.0], dtype=np.float32
            )
            print(f"[INFO] Pose-conditioned model: fixed goal for this run = {self.pose_target_world.tolist()}")

        # ------------------ Optional static obstacle projector (ground-truth
        # CYLINDERS/KEEPOUT_ZONES from config.avoiding-crazyflie.py -- assumes
        # the real corridor matches that layout) or --depth_obstacles (built
        # from the live depth stream instead, see _rebuild_depth_projector) ---
        self.vcfg = variant_cfg(args.variant)
        if args.num_candidates is not None and self.vcfg["selection"] != "first":
            self.vcfg["num_candidates"] = args.num_candidates
        self.gradient = (self.vcfg["projection_mode"] == "gradient")
        self.pos_projector = None
        self.proj_dt = 0.1
        self.depth_tracker = ObstacleTracker() if args.depth_obstacles else None
        self._last_depth_rebuild_time = None  # set on first _rebuild_depth_projector() call
        # latest detection, kept only so it's available if you want to log/inspect it
        self.depth_static_pts_latest, self.depth_dyn_preds_latest = [], {}
        self.depth_fx, self.depth_fy, self.depth_cx, self.depth_cy = camera_intrinsics(
            FPV_WIDTH, FPV_HEIGHT, FPV_FOCAL_LENGTH, FPV_HORIZONTAL_APERTURE
        )
        if self.vcfg["use_projection"] and not args.depth_obstacles:
            self.pos_projector = build_position_projector(
                horizon_H=self.horizon, gradient=self.gradient, device=device,
                cylinders=CYLINDERS,
                normalizer=None, tighten=self.vcfg["tighten"], dt=self.proj_dt,
                drone_radius=drone_radius,
                active_halfspaces=active_halfspaces,
                keepout_zones=KEEPOUT_ZONES,
                goal_pull_weight=goal_pull_weight,
            )
            if self.vcfg["projection_mode"] == "sdpc":
                self.pos_projector.inloop_slsqp = True
                self.pos_projector.action_normalizer = self.dataset.action_normalizer
                self.pos_projector.pos0 = None
        # --depth_obstacles: pos_projector is left None here -- run() builds it
        # from the first depth frame once the camera is ready, then rebuilds it
        # every step (see _rebuild_depth_projector).
        print(f"[INFO] Variant: {args.variant} (projection={self.vcfg['use_projection']}, "
              f"mode={self.vcfg['projection_mode']}, depth_obstacles={args.depth_obstacles})")

        if not args.live:
            print("[WARN] --live not set: DRY RUN. Velocity commands will be computed "
                  "and printed but NOT published. Pass --live to actually fly.")

        # ------------------ ROS2 node (must exist before any Subscription/Publisher) ------
        rclpy.init()
        self.node = rclpy.create_node("diffusion_policy_hardware")
        self.cam_state = {"color": None, "depth": None}
        # need a depth frame if the model itself consumes it as an input channel
        # OR --depth_obstacles wants it for obstacle detection (independent of
        # what the model needs -- an RGB-only model can still fly with
        # depth_obstacles on, same as eval_crazieflie1.py's cfg.USE_DEPTH = use_depth or args.depth_obstacles)
        self.need_depth_frame = self.use_depth or args.depth_obstacles
        self.node.create_subscription(Image, args.color_topic, self._color_cb, qos_profile_sensor_data)
        if self.need_depth_frame:
            self.node.create_subscription(Image, args.depth_topic, self._depth_cb, qos_profile_sensor_data)

        # ------------------ pose in, velocity setpoint out (mirrors flight_test.py's
        # /mavros/local_position/pose -> /mavros/setpoint_velocity/cmd_vel) ------------
        # "pos" starts as None (not a placeholder zero) so run()'s "wait for first
        # pose" loop below actually waits instead of falling through instantly --
        # a fixed bug: this used to be immediately overwritten with zeros right
        # after being set to None, which made that wait a no-op.
        self.state = {"pos": None, "quat": _IDENTITY_QUAT.copy()}
        self.node.create_subscription(PoseStamped, args.pose_topic, self._pose_cb, qos_profile_sensor_data)
        self.cmd_pub = self.node.create_publisher(PoseStamped, args.cmd_vel_topic, 1)

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

    def _color_cb(self, msg):
        if msg.encoding not in ("rgb8", "bgr8"):
            self.node.get_logger().warning(
                f"unsupported color encoding '{msg.encoding}', skipping frame",
                throttle_duration_sec=5.0
            )
            return
        frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        if msg.encoding == "bgr8":
            frame = frame[:, :, ::-1]
        self.cam_state["color"] = self._center_crop(frame)

    def _depth_cb(self, msg):
        if msg.encoding == "16UC1":
            depth = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
            depth = depth.astype(np.float32) * 0.001  # RealSense: raw mm -> metres
        elif msg.encoding == "32FC1":
            depth = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
        else:
            self.node.get_logger().warning(
                f"unsupported depth encoding '{msg.encoding}', skipping frame",
                throttle_duration_sec=5.0
            )
            return
        self.cam_state["depth"] = self._center_crop(depth)

    def _pose_cb(self, msg):
        p = msg.pose.position
        o = msg.pose.orientation
        print('pos:', p)
        self.state["pos"] = np.array([p.x, p.y, p.z], dtype=np.float32)
        # ROS quaternion fields are (x,y,z,w); depth_obstacle_estimator.py's
        # quat_apply/camera_world_pose expect (w,x,y,z) (Isaac's convention).
        self.state["quat"] = np.array([o.w, o.x, o.y, o.z], dtype=np.float32)

    # ------------------ small helpers ------------------
    def _cam_ready(self):
        return self.cam_state["color"] is not None and (not self.need_depth_frame or self.cam_state["depth"] is not None)

    def _grab_frame(self):
        if not self.use_depth:
            return self.cam_state["color"]
        return self.cam_state["color"], self.cam_state["depth"]

    def _publish_stop(self):
        if self.rclpy.ok():
            stop = self.TwistStamped()
            stop.header.stamp = self.node.get_clock().now().to_msg()
            self.cmd_pub.publish(stop)

    def _rebuild_depth_projector(self):
        """--depth_obstacles: capture the current depth frame, back-project ->
        filter -> cluster -> track (see depth_obstacle_estimator.py), and build
        the point-based projector from that instead of ground-truth cylinder
        geometry. Called once after camera warm-up and then every control step.
        Mirrors eval_crazieflie1.py's module-level _rebuild_depth_projector,
        adapted to read pose/depth from ROS2 state instead of an Isaac Sim env.
        """
        args = self.args
        depth = self.cam_state["depth"]
        pos_body_w = self.state["pos"]
        quat_body_w = self.state["quat"]
        pos_cam_w, quat_cam_w = camera_world_pose(pos_body_w, quat_body_w)

        # Camera-frame first (identity pose), NOT world frame directly -- same
        # reasoning as eval_crazieflie1.py: keep_nearest_along_z needs real
        # camera-frame lateral position to correctly drop flying-pixel/multipath
        # echo points behind the true front surface.
        pts_cam = backproject_depth_to_world(
            depth, self.depth_fx, self.depth_fy, self.depth_cx, self.depth_cy,
            _IDENTITY_POS, _IDENTITY_QUAT,
            max_range=args.depth_obstacle_max_range, stride=args.depth_obstacle_stride,
        )
        if len(pts_cam):
            pts_cam = keep_nearest_along_z(pts_cam, xy_bin_size=0.05)
        pts = pos_cam_w[None, :] + quat_apply(quat_cam_w, pts_cam)  # -> world frame

        # z-crop centered on the drone's CURRENT altitude, clamped to the real
        # flight envelope (see --depth_obstacle_z_band help).
        _z_band = args.depth_obstacle_z_band
        _z_lo = max(FLIGHT_Z_MIN, pos_body_w[2] - _z_band)
        _z_hi = min(FLIGHT_Z_MAX, pos_body_w[2] + _z_band)
        pts = filter_points(
            pts, x_bounds=(-0.5, 4.5), y_bounds=(-0.95, 0.95), z_bounds=(_z_lo, _z_hi),
            voxel_size=args.depth_obstacle_voxel, output_2d=True,
        )
        clusters = cluster_points(pts)

        # Real elapsed wall-clock time since the last call -- this runs once per
        # control-loop iteration on hardware (no fixed substep count like sim's
        # env.count), so measure it directly rather than assume a rate.
        now = time.time()
        rebuild_dt = now - self._last_depth_rebuild_time if self._last_depth_rebuild_time is not None else 1.0 / 30
        self._last_depth_rebuild_time = now
        active_tracks = self.depth_tracker.update(clusters, dt=rebuild_dt)

        static_pts, dyn_preds = tracks_to_constraints(
            active_tracks, horizon=self.horizon, proj_dt=self.proj_dt,
            max_points_per_obstacle=args.depth_obstacle_max_points,
        )
        self.depth_static_pts_latest, self.depth_dyn_preds_latest = static_pts, dyn_preds

        projector = build_position_projector_from_points(
            horizon_H=self.horizon, gradient=self.gradient, device=self.device,
            static_points=static_pts, dynamic_predictions=dyn_preds,
            point_radius=args.depth_obstacle_radius,
            normalizer=None, tighten=self.vcfg["tighten"], dt=self.proj_dt,
            use_dynamics=self.vcfg.get("use_dynamics", True),
            drone_radius=self.drone_radius,
            active_halfspaces=self.active_halfspaces,
            goal_pull_weight=self.goal_pull_weight,
        )
        if self.vcfg["projection_mode"] == "sdpc":
            projector.inloop_slsqp = True
            projector.action_normalizer = self.dataset.action_normalizer
            projector.pos0 = pos_body_w
        self.pos_projector = projector

    def run(self):
        args = self.args
        rclpy = self.rclpy
        node = self.node
        vcfg = self.vcfg
        use_depth = self.use_depth
        device = self.device
        run_start_time = time.time()

        print(f"[INFO] Waiting up to {args.start_delay}s for first pose on {args.pose_topic} ...")
        t0 = time.time()
        # In ROS2, we must spin to receive messages
        while self.state["pos"] is None and time.time() - t0 < args.start_delay and rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)

        if self.state["pos"] is None:
            raise RuntimeError(f"No pose received on {args.pose_topic} within {args.start_delay}s -- "
                                "check mavros is running and the topic name.")

        cam_topics = args.color_topic + (f" + {args.depth_topic}" if use_depth else "")
        print(f"[INFO] Waiting up to {args.start_delay}s for first camera frame on {cam_topics} ...")
        t0 = time.time()
        while not self._cam_ready() and time.time() - t0 < args.start_delay and rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)

        if not self._cam_ready():
            raise RuntimeError(f"No camera frame received on {cam_topics} within {args.start_delay}s -- "
                                "check the camera driver is running and the topic names.")

        if args.depth_obstacles and vcfg["use_projection"]:
            self._rebuild_depth_projector()
            print(f"[INFO] depth_obstacles: initial projector built from "
                  f"{len(self.depth_static_pts_latest)} detected static keep-out points.")

        # init obs history (same seeding pattern as sim: repeat the first real frame To times)
        frame0 = self._grab_frame()
        rgb_hist = deque(maxlen=self.To)
        for _ in range(self.To):
            rgb_hist.append((frame0[0].copy(), frame0[1].copy()) if use_depth else frame0.copy())

        prev_actions_real = None
        control_hz = 30
        step = 0
        print(f"[INFO] Starting control loop at {control_hz} Hz Ctrl+C to stop.")

        # trajectory log for the .npz save below (same idea as eval_crazieflie1.py's
        # traj_xyz/actions_taken, but logging the pose each action was computed from --
        # hardware has no synchronous "post-step position", pose arrives async via _pose_cb)
        pos_init = self.state["pos"].copy()
        traj_xyz = [pos_init.copy()]
        actions_taken = []

        try:
            while rclpy.ok():
                # Spin once to process latest callbacks before acting
                rclpy.spin_once(node, timeout_sec=0)

                pos = self.state["pos"].copy()

                obs_rgb_t = preprocess_obs_stack(rgb_hist, use_depth).to(device)
                cond = {"obs_rgb": obs_rgb_t}

                if self.use_pose_cond:
                    pose_now_norm = self.dataset.pose_normalizer.normalize(pos[:3].astype(np.float32))
                    pose_target_norm = self.dataset.pose_normalizer.normalize(self.pose_target_world)
                    cond["pose_now"] = torch.from_numpy(pose_now_norm).float().unsqueeze(0).to(device)
                    cond["pose_target"] = torch.from_numpy(pose_target_norm).float().unsqueeze(0).to(device)

                if args.depth_obstacles and vcfg["use_projection"]:
                    # Always rebuild: unlike ground-truth cylinders, the detected
                    # point cloud changes every step regardless of whether any
                    # obstacle is actually moving.
                    self._rebuild_depth_projector()

                if vcfg["projection_mode"] == "sdpc" and self.pos_projector is not None:
                    self.pos_projector.pos0 = pos[:3]
                in_loop_projector = self.pos_projector if vcfg["projection_mode"] == "sdpc" else None

                a_candidates_norm, _ = sample_action_candidates(
                    diffusion=self.diffusion, cond=cond, horizon=self.horizon, action_dim=self.action_dim,
                    num_candidates=vcfg["num_candidates"], projector=in_loop_projector,
                )
                a_candidates_real = self.dataset.action_normalizer.unnormalize(a_candidates_norm) * float(args.action_scale)

                if vcfg["use_projection"] and vcfg["projection_mode"] in ("post", "sdpc"):
                    a_candidates_proj_real, proj_costs = project_action_candidates_with_positions(
                        projector=self.pos_projector, pos0=pos[:3], a_candidates_real=a_candidates_real, device=device,
                    )
                else:
                    a_candidates_proj_real, proj_costs = a_candidates_real, None

                if vcfg["selection"] == "minimum_projection_cost" and proj_costs is not None:
                    which = int(np.argmin(proj_costs))
                else:
                    which = choose_trajectory(a_candidates_proj_real, strategy=vcfg["selection"],
                                                prev_actions_real=prev_actions_real)

                a0_real = a_candidates_proj_real[which, 0]
                prev_actions_real = a_candidates_proj_real[which:which + 1]

                # delta-position -> velocity setpoint
                pos_cmd = np.zeros(3, dtype=np.float32)

                inc_action = a0_real[:self.action_dim] * args.action_scale
                inc_action = np.clip(inc_action, [-0.5, -0.5, -0.1], [0.5, 0.5, 0.1])

                pos_cmd[:self.action_dim] = pos + inc_action
                # pos_cmd = np.clip(pos_cmd, -args.max_speed, args.max_speed)

                cmd_pos = self.PoseStamped()
                cmd_pos.header.stamp = node.get_clock().now().to_msg()
                cmd_pos.pose.position.x = float(pos_cmd[0])
                cmd_pos.pose.position.y = float(pos_cmd[1])
                cmd_pos.pose.position.z = float(pos_cmd[2])

                print(f"[step {step}] pos={pos} a0_real={a0_real} pos_cmd={pos_cmd}"
                        f"{'  (DRY RUN)' if not args.live else ''}")

                if args.live:
                    self.cmd_pub.publish(cmd_pos)

                traj_xyz.append(pos.copy())
                actions_taken.append(a0_real.copy())

                rgb_hist.append(self._grab_frame())
                step += 1
                time.sleep(1.0 / control_hz)
        finally:
            self._save_trajectory(traj_xyz, actions_taken, time.time() - run_start_time)

        self._publish_stop()
        print("[INFO] Hardware run stopped, zero velocity published.")
        node.destroy_node()
        rclpy.try_shutdown()

    def _save_trajectory(self, traj_xyz, actions_taken, wall_time_sec):
        """Same .npz schema as eval_crazieflie1.py's per-episode save (xyz, actions,
        variant/encoder/latent_dim, cylinders, halfspaces, projection settings), minus
        the fields that only make sense in sim (success/fell come from env.step(), no
        equivalent on hardware; no candidate-rollout snapshots -- see run() docstring)."""
        args = self.args
        out_dir = Path(self.run_dir) / "hardware_trajectories"
        out_dir.mkdir(parents=True, exist_ok=True)
        traj_path = out_dir / f"traj_{self.encoder_type}_L{self.latent_dim}_{args.variant}_{time.strftime('%Y%m%d_%H%M%S')}.npz"

        np.savez(
            traj_path,
            xyz            = np.array(traj_xyz),        # (T, 3)
            actions        = np.array(actions_taken),    # (T, 3)
            variant        = args.variant,
            encoder        = self.encoder_type,
            latent_dim     = self.latent_dim,
            wall_time_sec  = float(wall_time_sec),
            wall_time_min  = float(wall_time_sec) / 60.0,
            cylinders      = np.array(CYLINDERS),         # (N, 2) assumed ground-truth layout
            halfspaces     = np.array([[hs[0], hs[1]] for hs in self.active_halfspaces], dtype=object),
            hs_sides       = np.array([hs[2] for hs in self.active_halfspaces]),
            tighten        = self.vcfg.get("tighten", 0.0),
            use_projection = self.vcfg["use_projection"],
            projection_mode = self.vcfg["projection_mode"],
            num_candidates = self.vcfg["num_candidates"] if self.vcfg["num_candidates"] > 0 else 1,
            selection      = self.vcfg["selection"],
            action_scale   = float(args.action_scale),
            live           = bool(args.live),
            use_pose_cond  = bool(self.use_pose_cond),
            pose_target    = self.pose_target_world if self.pose_target_world is not None else np.zeros(3, dtype=np.float32),
        )
        print(f"[TRAJ] saved: {traj_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--action_scale", type=float, default=1.0)
    parser.add_argument("--num_candidates", type=int, default=2,)
    parser.add_argument("--use_halfspaces", action="store_true", default=False,)
    parser.add_argument("--variant", type=str, choices=projection_variants, default="diffuser",
                        help="Single projection variant to fly (no sweep on real hardware).")
    parser.add_argument("--pose_topic", type=str, default="/mavros/local_position/pose",)
    parser.add_argument("--cmd_vel_topic", type=str, default="/mpc/set_pose",)
    parser.add_argument("--max_speed", type=float, default=0.5,
                        help="Per-axis velocity clamp in m/s, applied to every published "
                             "command independent of --action_scale.")
    parser.add_argument("--color_topic", type=str, default="camera/camera/color/image_raw",)
    parser.add_argument("--depth_topic", type=str, default="/camera/aligned_depth_to_color/image_raw",)
    parser.add_argument("--start_delay", type=float, default=5.0,
                        help="Seconds to wait for the first pose/camera message before giving up.")
    parser.add_argument("--live", action="store_true", default=False,)
    parser.add_argument("--target_x", type=float, default=4.0,)
    parser.add_argument("--target_y", type=float, default=0.75,)
    parser.add_argument("--depth_obstacles", action="store_true", default=False,
                        help="Detect obstacles from the live depth stream instead of the "
                             "assumed ground-truth CYLINDERS/KEEPOUT_ZONES layout in config.")
    parser.add_argument("--depth_obstacle_radius", type=float, default=0.2,
                        help="Keep-out radius (m) around each detected surface point, on top "
                             "of drone_radius (depth_obstacles only).")
    parser.add_argument("--depth_obstacle_max_range", type=float, default=1.0,
                        help="Drop depth points farther than this from the camera (m).")
    parser.add_argument("--depth_obstacle_stride", type=int, default=2,
                        help="Pixel stride when back-projecting the depth image (speed/density trade-off).")
    parser.add_argument("--depth_obstacle_voxel", type=float, default=0.05,
                        help="Voxel size (m) for downsampling the back-projected point cloud.")
    parser.add_argument("--depth_obstacle_max_points", type=int, default=12,
                        help="Cap on keep-out points passed to the projector per tracked obstacle "
                             "(bounds the SLSQP solve's constraint count).")
    parser.add_argument("--depth_obstacle_z_band", type=float, default=0.1,
                        help="Half-height (m) of the z-band around the drone's current altitude "
                             "that depth points are kept from.")

    args, _unknown = parser.parse_known_args()

    device         = torch.device("cuda:0")
    drone_radius   = 0.1
    goal_pull_weight = 0.05

    # ── active halfspaces: from config only when --use_halfspaces is set ─────────
    active_halfspaces = corridor_halfspaces if args.use_halfspaces else []
    if args.use_halfspaces:
        print(f"[INFO] Halfspace constraints enabled ({len(active_halfspaces)} halfspaces)")

    Ros2HardwareRunner(args, args.run_dir, device, drone_radius, active_halfspaces, goal_pull_weight).run()



if __name__ == "__main__":
    main()