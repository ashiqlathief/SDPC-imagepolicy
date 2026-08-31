import argparse
import importlib
import os
import sys
from collections import deque
from pathlib import Path
import time
import numpy as np
np.set_printoptions(precision=3, suppress=True)
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import re 
from matplotlib.patches import Rectangle, Circle, Polygon

import diffuser.utils as utils
from diffuser.sampling.projection import Projector
from diffuser.sampling.policies import temporal_consistency_distances
from metrics_logger import MetricsLogger
from depth_obstacle_estimator import (
    camera_intrinsics, camera_world_pose, backproject_depth_to_world, quat_apply,
    filter_points, cluster_points, ObstacleTracker, tracks_to_constraints,
    keep_nearest_along_z,
    FPV_WIDTH, FPV_HEIGHT, FPV_FOCAL_LENGTH, FPV_HORIZONTAL_APERTURE,
)
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
FLIGHT_Z_MAX = 2.5   # ceiling

z_halfspaces = [
    ([0.0, 0.0,  1.0], FLIGHT_Z_MAX),   # z <= FLIGHT_Z_MAX :drone cannot fly above wall/ceiling
    ([0.0, 0.0, -1.0], FLIGHT_Z_MIN),   # z >= FLIGHT_Z_MIN :drone cannot go underground
]

projection_variants = [
  'sdpc-r',
#   'sdpc-r-tightened',
  'sdpc-c',
# #   'sdpc-c-tightened',
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

def plot_constraint_overlay(ax, cylinders, tighten=0.03, drone_radius=0.0,
                            x_bounds=(-0.5, 4.5), y_bounds=(-1.0, 1.0)):
    """
    Overlays safety margins as blue-shaded regions on an existing XY axes.
    Shows the actual exclusion zone the projector enforces.
    """
    cyl_r = CYL_RADIUS + drone_radius + tighten

    # - obstacle margins (blue circles) -
    for (x, y) in cylinders:
        ax.add_patch(plt.Circle((x, y), cyl_r,
                                color="royalblue", alpha=0.15, zorder=2))
        ax.add_patch(plt.Circle((x, y), cyl_r,
                                fill=False, edgecolor="royalblue",
                                linewidth=1.2, linestyle="--", zorder=3))

    # - corridor bounds (blue shaded strips outside the allowed region) -
    if x_bounds is not None:
        xmin, xmax = x_bounds
        ax.axvspan(ax.get_xlim()[0], xmin, color="royalblue", alpha=0.08, zorder=1)
        ax.axvspan(xmax, ax.get_xlim()[1], color="royalblue", alpha=0.08, zorder=1)
    if y_bounds is not None:
        ymin, ymax = y_bounds
        ax.axhspan(ax.get_ylim()[0], ymin, color="royalblue", alpha=0.08, zorder=1)
        ax.axhspan(ymax, ax.get_ylim()[1], color="royalblue", alpha=0.08, zorder=1)

    # legend proxy
    margin_patch = mpatches.Patch(color="royalblue", alpha=0.4, label=f"constraint margin (tighten={tighten})")
    bound_line   = mpatches.Patch(color="royalblue", alpha=0.15, label="out-of-bounds zone")
    return [margin_patch, bound_line]

def plot_halfspace_constraints_xy(ax, halfspaces, xlim, ylim, alpha=0.10):
    """
    halfspaces: list of [[x1,y1],[x2,y2], side] where side in {'above','below'}
    Draws the boundary line and shades the feasible half-plane.
    We interpret:
   'below' : y <= m x + b
   'above' : y >= m x + b
    """
    xmin, xmax = xlim
    ymin, ymax = ylim

    for hs in halfspaces:
        p1, p2, side = hs
        x1, y1 = p1
        x2, y2 = p2

        # Handle vertical line separately
        if abs(x2 - x1) < 1e-8:
            x = x1
            ax.plot([x, x], [ymin, ymax], linewidth=2.0, alpha=0.7)

            # Shade feasible side: left/right isn't defined by above/below.
            # So for vertical walls, you should encode with a different convention.
            # We'll skip shading for vertical.
            continue

        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1

        xs = np.array([xmin, xmax], dtype=np.float32)
        ys = m * xs + b

        # boundary line
        ax.plot(xs, ys, linewidth=2.0, alpha=0.7)

        # shade feasible side
        if side == "above":
            # fill from line down to ymin
            ax.fill_between(xs, ys, ymin, alpha=alpha)
        else:
            # 'above': fill from line up to ymax
            ax.fill_between(xs, ys, ymax, alpha=alpha)

def add_obstacles_xy(ax, cylinders, cyl_radius=CYL_RADIUS):
    # Cylinders as circles in XY
    for x, y in cylinders:
        ax.add_patch(
            Circle(
                (x, y),
                cyl_radius,
                linewidth=1.0,
                edgecolor="black",
                facecolor="tab:orange",
                alpha=0.30,
            )
        )

def add_dynamic_cylinders_xy(ax, cyl_rest, cyl_axes, cand_snapshots, obs_amplitude,
                              cyl_radius=CYL_RADIUS, drone_radius=0.0,
                              x_clamp=(0.3, 4.7), y_clamp=(-0.85, 0.85)):
    """
    Draw dynamic cylinder visualisation on an XY axes:
   orange band  = oscillation sweep of the exclusion zone, shaped per cylinder's
                        motion axis ("y": vertical band, "x": horizontal band,
                        "xy": square bounding box:diagonal motion is a 1D line through
                        it, so the square is a conservative over-approximation, not exact)
   dashed circle = exclusion zone at rest position
   faded dots    = physical cylinder position sampled every few steps
    `cyl_axes` must be the same length/order as `cyl_rest` (one axis per cylinder).
    """
    excl_r = cyl_radius + drone_radius   # total exclusion radius

    for (x0, y0), axis in zip(cyl_rest, cyl_axes):
        if axis == "y":
            ylo = max(y_clamp[0], y0 - obs_amplitude)
            yhi = min(y_clamp[1], y0 + obs_amplitude)
            band = Rectangle((x0 - excl_r, ylo), 2 * excl_r, yhi - ylo,
                              facecolor="tab:orange", alpha=0.15, zorder=1)
        elif axis == "x":
            xlo = max(x_clamp[0], x0 - obs_amplitude)
            xhi = min(x_clamp[1], x0 + obs_amplitude)
            band = Rectangle((xlo, y0 - excl_r), xhi - xlo, 2 * excl_r,
                              facecolor="tab:orange", alpha=0.15, zorder=1)
        else:  # "xy":same delta applied to both axes → 45° diagonal line sweep
            x_lo = max(x_clamp[0], x0 - obs_amplitude)
            x_hi = min(x_clamp[1], x0 + obs_amplitude)
            y_lo = max(y_clamp[0], y0 - obs_amplitude)
            y_hi = min(y_clamp[1], y0 + obs_amplitude)
            p1 = np.array([x_lo, y_lo])
            p2 = np.array([x_hi, y_hi])
            seg = p2 - p1
            d_hat = seg / (np.linalg.norm(seg) + 1e-8)
            n_hat = np.array([-d_hat[1], d_hat[0]])   # perpendicular
            corners = np.array([p1 - excl_r*n_hat, p1 + excl_r*n_hat,
                                 p2 + excl_r*n_hat, p2 - excl_r*n_hat])
            band = Polygon(corners, facecolor="tab:orange", alpha=0.15, zorder=1)
        ax.add_patch(band)
        # exclusion zone at rest (dashed ring)
        ax.add_patch(Circle(
            (x0, y0), excl_r,
            linewidth=1.2, linestyle="--",
            edgecolor="tab:orange", facecolor="none", alpha=0.5, zorder=2,
        ))

    # per-step physical cylinder positions (sub-sampled to avoid clutter)
    stride = max(1, len(cand_snapshots) // 30)
    for snap in cand_snapshots[::stride]:
        if snap.get("cyl_xy") is None:
            continue
        for (cx, cy) in snap["cyl_xy"]:
            ax.add_patch(Circle(
                (cx, cy), cyl_radius,   # physical size only, not exclusion zone
                facecolor="tab:orange", alpha=0.10, zorder=2,
            ))

def get_rgb_from_env(env):
    """
    Fetch RGB frame only from env.get_rgb().
    Returns uint8 image in (H, W, 3).
    """
    if not hasattr(env, "get_rgb"):
        raise RuntimeError("Env does not have get_rgb() method.")

    frame = env.get_rgb()
    # print(frame.shape, frame.dtype, frame.min(), frame.max())
    return frame

def get_obs_frame_from_env(env, use_depth):
    """
    Fetch one observation frame for the policy's history buffer.
    Returns either:
   rgb: (H,W,3) uint8                          if use_depth=False
   (rgb, depth): (H,W,3) uint8, (H,W) float32  if use_depth=True
    Depth comes from env.get_depth(), which already clamps non-finite
    (inf/nan) pixels to DEPTH_FAR (see crazyflie_env.py), matching what
    quadcopter.py does at collection time.
    """
    rgb = get_rgb_from_env(env)
    if not use_depth:
        return rgb
    if not hasattr(env, "get_depth"):
        raise RuntimeError("use_depth=True but env does not have get_depth() method.")
    depth = env.get_depth()
    return rgb, depth

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
    obs_hist: list/deque of To frames from get_obs_frame_from_env().
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

def integrate_candidates_xyz(pos0_xyz, a_candidates_real):
    """
    Integrates delta-position candidates into absolute (x,y,z) positions.
    Kept as xyz (not xy-only) so candidate/planned horizons carry altitude
    too, not just the top-down (x,y) path.
    pos0_xyz: (3,)
    a_candidates_real: (K,H,3)
    Returns: traj_xyz (K, H+1, 3)
    """
    K, H, _ = a_candidates_real.shape
    traj_xyz = np.zeros((K, H + 1, 3), dtype=np.float32)
    traj_xyz[:, 0, :] = pos0_xyz[:3][None, :]
    traj_xyz[:, 1:, :] = pos0_xyz[:3][None, None, :] + np.cumsum(a_candidates_real[:, :, :3], axis=1)
    return traj_xyz

def build_obstacle_constraint_list(cylinders, x_bounds=None,
                                    y_bounds=None, z_bounds=None,
                                    corridor_halfspaces=None, z_halfspaces=None, tighten=0.0,
                                    cyl_extra_radius=0.0, drone_radius=0.0,
                                    dynamic_cylinder_predictions=None, keepout_zones=None):
    """
    dynamic_cylinder_predictions: optional {cylinder_index: [(x,y), ...]}:one
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
    # Floor:   [0,0,-1]·p <= 0.0 - tighten        (shrinks floor up:use tighten=0 here)
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
    x_bounds = (-6.5, 4.5)
    y_bounds = (-1.95, 1.95)
    z_bounds = (0.0, 2.0)
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
        parallelize=True,         # candidates solve independently:run them concurrently
        goal_pull_weight=goal_pull_weight,
    )

    return projector

def build_point_obstacle_constraints(static_points, dynamic_predictions, radius):
    """depth_obstacles counterpart to the cylinder loop inside
    build_obstacle_constraint_list(): same ("sphere_outside", ...) /
    ("sphere_outside_dynamic", ...) tuple shapes, one entry per detected
    surface point instead of one per fitted circle (see
    depth_obstacle_estimator.py's module docstring  "option b")."""
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
    bounds/halfspace setup (reuses build_obstacle_constraint_list
    unchanged, with cylinders=[] so no ground-truth cylinder constraints get
    added), but cylinder obstacles come from a detected point cloud (see
    depth_obstacle_estimator.py) instead of CYLINDERS / env.get_cylinder_positions().
    """
    Hp1 = horizon_H + 1
    x_bounds = (-6.5, 4.5)
    y_bounds = (-1.95, 1.95)
    z_bounds = (0.0, 2.0)
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

    # one batched call instead of K single-candidate calls:projector.project()
    # already accepts a batched input, and building it once avoids rebuilding the
    # (identical) constraint set K times.
    state_proj_t, proj_costs = projector.project(state_t)  # (K,H+1,transition_dim), cost shape (K,)
    state_proj = state_proj_t.detach().cpu().numpy()
    pos_proj = state_proj[..., :3]  # (K,H+1,3)

    # convert back to deltas (K,H,3)
    a_proj_real = (pos_proj[:, 1:] - pos_proj[:, :-1]).astype(np.float32)
    proj_costs = proj_costs.astype(np.float32)

    return a_proj_real, proj_costs

def _rebuild_depth_projector(current_pos, env, args, eval_dt,
                              depth_fx, depth_fy, depth_cx, depth_cy,
                              depth_tracker, depth_static_accum, depth_dynamic_accum,
                              horizon, proj_dt, gradient, device, vcfg,
                              drone_radius, active_halfspaces, dataset,
                              goal_pull_weight=0.0):
    """depth_obstacles: capture a depth frame, back-project -> filter ->
    cluster -> track (see depth_obstacle_estimator.py), and build the
    point-based projector from that instead of ground-truth cylinder
    geometry. Called once after camera warm-up and then every step.

    Pulled out of main() to module level  everything it used to close over
    (env, args, the per-episode depth_tracker/accumulators, etc.) is now an
    explicit parameter instead."""
    depth = env.get_depth()
    root = env.robot.data.root_state_w[0].detach().cpu().numpy()
    pos_body_w, quat_body_w = root[0:3], root[3:7]
    pos_cam_w, quat_cam_w = camera_world_pose(pos_body_w, quat_body_w)

    # Camera-frame first (identity pose), NOT world frame directly 
    # keep_nearest_along_z needs real camera-frame lateral position to
    # correctly drop flying-pixel/multipath echo points (same viewing
    # ray, spurious extra depth) behind the true front surface; doing
    # this after the world transform would bucket by world (x,y)
    # instead, which is a different and incorrect operation (see
    # _IDENTITY_POS/_IDENTITY_QUAT comment above).
    pts_cam = backproject_depth_to_world(
        depth, depth_fx, depth_fy, depth_cx, depth_cy, _IDENTITY_POS, _IDENTITY_QUAT,
        max_range=args.depth_obstacle_max_range, stride=args.depth_obstacle_stride,
    )
    # Always applied (not a CLI toggle)  drops flying-pixel/
    # multipath echo points behind the true front surface, which
    # would otherwise become spurious extra keep-out constraints.
    # Bin width fixed at 5cm rather than exposed as a knob.
    if len(pts_cam):
        pts_cam = keep_nearest_along_z(pts_cam, xy_bin_size=0.05)
    pts = pos_cam_w[None, :] + quat_apply(quat_cam_w, pts_cam)  # -> world frame

    # z-crop is centered on the drone's CURRENT altitude (not a fixed
    # floor-to-ceiling band) -- narrows how many candidate points reach
    # filter_points/cluster_points/tracking each step, since a real depth
    # sensor's filtering cost scales with point count and this matters on
    # hardware. Trade-off: only obstacle segments within +/-z_band of the
    # drone's current height are seen at all (see --depth_obstacle_z_band
    # help). Clamped to the real flight envelope so the band never extends
    # below the floor or above the ceiling.
    _z_band = args.depth_obstacle_z_band
    _z_lo = max(FLIGHT_Z_MIN, pos_body_w[2] - _z_band)
    _z_hi = min(FLIGHT_Z_MAX, pos_body_w[2] + _z_band)
    pts = filter_points(
        pts, x_bounds=(-6.5, 4.5), y_bounds=(-2.0, 2.0), z_bounds=(_z_lo, _z_hi),
        voxel_size=args.depth_obstacle_voxel, output_2d=True,
    )
    clusters = cluster_points(pts)
    # One outer eval `step()` call  and therefore one call to this
    # function  advances the sim by env.count physics substeps at
    # eval_dt each (see Crazyflie.step() in crazyflie_env.py), not by
    # eval_dt alone. Using eval_dt alone here would inflate every
    # velocity estimate by ~env.count x, since vel = disp / dt.
    _rebuild_dt = eval_dt * getattr(env, "count", 100)
    active_tracks = depth_tracker.update(clusters, dt=_rebuild_dt)

    # Accumulate every freshly-matched (missed==0) point this episode 
    # separate from tracks_to_constraints' capped output below, which is
    # what the projector actually uses. Skips missed!=0 tracks so a stale
    # track's frozen points (unchanged since its last real match) aren't
    # re-added every single step. Deduped by voxel cell (so re-detecting
    # the same surface repeatedly doesn't grow the set unbounded), but the
    # REAL continuous position is stored, not the snapped cell coordinate
    #  keeps a curved surface looking curved in the plot, not gridded.
    _voxel = args.depth_obstacle_voxel
    for _tid, _tr in active_tracks:
        if _tr["missed"] != 0:
            continue
        _accum = depth_dynamic_accum if _tr["is_dynamic"] else depth_static_accum
        for _p in _tr["points"]:
            _x, _y = float(_p[0]), float(_p[1])
            _key = (round(_x / _voxel), round(_y / _voxel))
            _accum.setdefault(_key, (_x, _y))

    static_pts, dyn_preds = tracks_to_constraints(
        active_tracks, horizon=horizon, proj_dt=proj_dt,
        max_points_per_obstacle=args.depth_obstacle_max_points,
    )
    projector = build_position_projector_from_points(
        horizon_H=horizon, gradient=gradient, device=device,
        static_points=static_pts, dynamic_predictions=dyn_preds,
        point_radius=args.depth_obstacle_radius,
        normalizer=None, tighten=vcfg["tighten"], dt=proj_dt,
        use_dynamics=vcfg.get("use_dynamics", True),
        drone_radius=drone_radius,
        active_halfspaces=active_halfspaces,
        goal_pull_weight=goal_pull_weight,
    )
    if vcfg["projection_mode"] == "sdpc":
        projector.inloop_slsqp = True
        projector.action_normalizer = dataset.action_normalizer
        projector.pos0 = current_pos
    return projector, static_pts, dyn_preds

def main():
    run_start_time = time.time()   # whole-run clock  printed at the end, see env.close() below
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7])
    parser.add_argument("--max_steps", type=int, default=700)
    parser.add_argument("--action_scale", type=float, default=1.0)
    parser.add_argument("--dynamic_obstacles", type=str, nargs="*", default=None, metavar="IDX:AXIS",
                        help="Enable sinusoidal cylinder movement. Omit this flag entirely to disable. "
                             "Pass with no values to move ALL cylinders laterally (axis 'y'). "
                             "Or give 'idx:axis' tokens (axis is 'x', 'y', or 'xy'; ':axis' optional, "
                             "defaults to 'y'), e.g. dynamic_obstacles 0:y 2:x 4:xy")
    parser.add_argument("--num_candidates", type=int, default=2,
                        help="Override K (number of sampled candidates) for variants that select among multiple candidates")
    parser.add_argument("--use_halfspaces", action="store_true", default=False,
                        help="Enforce corridor halfspace constraints (from CORRIDOR_HALFSPACES "
                             "in config) in the projector and show them in XY plots.")
    parser.add_argument("--depth_obstacles", action="store_true", default=False)
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
    parser.add_argument("--depth_obstacle_z_band", type=float, default=0.1)
    parser.add_argument("--save_frames", action="store_true", default=False)
    parser.add_argument("--target_y", type=float, default=0.75,
                         help="y-coordinate of the fixed goal fed to pose-conditioned models "
                              "as pose_target=(env.cfg.gate_x_max, target_y, 1.0). Ignored for "
                              "image-only models (dataset.use_pose_cond=False).")
    args, _unknown = parser.parse_known_args()

    device         = torch.device("cuda:0")
    drone_radius   = 0.1
    obs_amplitude  = 0.35
    obs_frequency  = 0.25
    # Applied to every SLSQP-projected variant uniformly (see Projector.
    # goal_pull_weight) -- 0.05 is a first guess, not tuned. Set to 0.0 to
    # reproduce the exact pre-goal-pull nearest-point-only projection.
    goal_pull_weight = 0.05

    depth_fx, depth_fy, depth_cx, depth_cy = camera_intrinsics(
        FPV_WIDTH, FPV_HEIGHT, FPV_FOCAL_LENGTH, FPV_HORIZONTAL_APERTURE
    )

    # ── active halfspaces: from config only when --use_halfspaces is set ─────────
    active_halfspaces = corridor_halfspaces if args.use_halfspaces else []
    if args.use_halfspaces:
        print(f"[INFO] Halfspace constraints enabled ({len(active_halfspaces)} halfspaces)")

    # ── parse --dynamic_obstacles tokens into (enabled, indices, axes) ──────────
    #   --dynamic_obstacles          → all cylinders, y-axis
    #   --dynamic_obstacles xy       → all cylinders, xy diagonal
    #   --dynamic_obstacles x        → all cylinders, x-axis
    #   --dynamic_obstacles 0:xy 3:y → per-cylinder idx:axis specification
    if args.dynamic_obstacles is None:
        args.dynamic_obstacles_enabled = False
        args.dynamic_cyl_indices = None
        args.obs_axes = None
    else:
        args.dynamic_obstacles_enabled = True
        if len(args.dynamic_obstacles) == 0:
            args.dynamic_cyl_indices = None   # all cylinders
            args.obs_axes = None              # defaults to "y" in env
        elif len(args.dynamic_obstacles) == 1 and args.dynamic_obstacles[0] in ("x", "y", "xy"):
            # single bare axis → apply to all cylinders
            args.dynamic_cyl_indices = None
            args.obs_axes = [args.dynamic_obstacles[0]]  # broadcast in env
        else:
            indices, axes = [], []
            for tok in args.dynamic_obstacles:
                idx_str, _, axis = tok.partition(":")
                axis = axis or "y"
                if axis not in ("x", "y", "xy"):
                    parser.error(f"dynamic_obstacles: invalid axis '{axis}' in '{tok}' (use x, y, or xy)")
                indices.append(int(idx_str))
                axes.append(axis)
            args.dynamic_cyl_indices = indices
            args.obs_axes = axes

    if args.seeds:
        run_dirs = [os.path.join(args.run_dir, str(s)) for s in args.seeds]
    else:
        run_dirs = [args.run_dir]

    env = None
    shared_use_depth = None

    for run_dir in run_dirs:
        # ------------------ Load trained experiment ------------------
        print(f"\n[INFO] Loading run dir: {run_dir}")

        seedmodel = int(Path(run_dir).name)   # gives 9
        diff_exp = utils.load_diffusion(run_dir,epoch="best",device=str(device),)
        dataset = diff_exp.dataset
        diffusion = diff_exp.diffusion.to(device)
        diffusion.eval()

        use_depth = bool(getattr(diffusion.model, "use_depth", False))
        assert bool(getattr(dataset, "use_depth", False)) == use_depth, (
            f"Loaded checkpoint's model.use_depth={use_depth} but dataset.use_depth="
            f"{getattr(dataset, 'use_depth', False)} -- mismatched run dir?"
        )
        print(f"[INFO] Running evaluation in {'RGB-D' if use_depth else 'RGB'} mode "
              f"(use_depth={use_depth}, in_chans={getattr(diffusion.model, 'in_chans', 3)})")

        cfg.USE_DEPTH = use_depth or args.depth_obstacles

        from isaac.scripts.crazyflie_env import Crazyflie, CrazyflieEnvCfg

        eval_dt = 1/30
        if env is None:
            env_cfg = CrazyflieEnvCfg(
                num_envs=1,
                device=str(device),
                dynamic_obstacles=args.dynamic_obstacles_enabled,
                obs_amplitude=obs_amplitude,
                obs_frequency=obs_frequency,
                dynamic_cyl_indices=args.dynamic_cyl_indices,
                obs_axes=args.obs_axes,
                drone_radius=drone_radius,
            )
            env = Crazyflie(env_cfg)
            shared_use_depth = use_depth
        elif use_depth != shared_use_depth:
            raise RuntimeError(
                f"run_dir {run_dir}'s checkpoint has use_depth={use_depth}, but that doesn't match "
                f"the already-running sim (use_depth={shared_use_depth}, set from an earlier seed "
                "in this seeds batch). Mixed RGB/RGB-D seeds can't share one sim instance  "
                "run this seed separately."
            )

        run_name = Path(run_dir).parent.name  # e.g. "H16_K20_ENCvit_LAT256"
        enc_match = re.search(r'E([^_]+)', run_name)
        lat_match = re.search(r'L(\d+)', run_name)
        encoder_type = enc_match.group(1) if enc_match else "unknown"
        latent_dim   = int(lat_match.group(1)) if lat_match else 256
        print(f"[INFO] Inferred encoder type: {encoder_type}, latent dim: {latent_dim}")
        # Infer dims
        horizon = int(getattr(diffusion, "horizon", 16))
        action_dim = int(getattr(diffusion, "action_dim", 3))
        To = int(getattr(dataset, "n_obs_steps", 2)) if hasattr(dataset, "n_obs_steps") else 2
        print(f"[INFO] Online eval started. To={To}, H={horizon}, action_dim={action_dim}")

        use_pose_cond = bool(getattr(dataset, "use_pose_cond", False))
        pose_target_world = None
        if use_pose_cond:
            pose_target_world = np.array(
                [env.cfg.gate_x_max, args.target_y, 1.0], dtype=np.float32
            )
            print(f"[INFO] Pose-conditioned model: fixed goal for this run = {pose_target_world.tolist()}")
        logger = MetricsLogger(
            save_dir=os.path.join(run_dir, "results"),                               
            encoder_type=encoder_type,
            latent_dim=latent_dim,
            horizon=horizon,
            n_diffusion_steps=20,
            corridor_halfspaces  = corridor_halfspaces,
            cylinders            = CYLINDERS,
        )

        # ── auto-increment output dirs so each run gets its own folder ────────────
        def _next_free(base, name):
            path = os.path.join(base, name)
            if not os.path.exists(path):
                return path
            i = 1
            while os.path.exists(os.path.join(base, f"{name}{i}")):
                i += 1
            return os.path.join(base, f"{name}{i}")

        traj_dir = _next_free(run_dir, "trajectories")
        plot_dir = _next_free(run_dir, "plots")
        os.makedirs(traj_dir, exist_ok=True)
        os.makedirs(plot_dir, exist_ok=True)
        print(f"[INFO] trajectories → {traj_dir}")
        print(f"[INFO] plots        → {plot_dir}")

        # ------------------ Episodes ------------------
        for ep, variant_name in enumerate(projection_variants):
            vcfg = variant_cfg(variant_name)
            if args.num_candidates is not None and vcfg["selection"] != "first":
                vcfg["num_candidates"] = args.num_candidates
            # ------------------ Optional projection ------------------
            projection_mode=vcfg["projection_mode"]
            gradient = (projection_mode == "gradient")
            pos_projector = None   # obstacle-aware SLSQP projector (post-hoc + sdpc in-loop)
            depth_tracker = ObstacleTracker() if args.depth_obstacles else None
            # latest --depth_obstacles detection, stashed per step into cand_snapshots
            # below so the final XY plot can show what the projector actually saw.
            depth_static_pts_latest, depth_dyn_preds_latest = [], {}
            # Full-episode accumulation of every point ever detected, independent of
            # the small per-step cap (--depth_obstacle_max_points) the projector's
            # constraints use -- for the "show/save everything" plot only, never fed
            # back into build_position_projector_from_points. Keyed by voxel-grid
            # cell (for dedup) but stores the real, continuous detected position as
            # the value -- so a curved surface still plots as a curve, not a grid of
            # voxel-snapped squares.
            depth_static_accum: dict[tuple[int, int], tuple[float, float]] = {}
            depth_dynamic_accum: dict[tuple[int, int], tuple[float, float]] = {}


            if vcfg["use_projection"] and not args.depth_obstacles:
                proj_dt = 0.1
                # All cylinders (static and dynamic) use the same base radius.
                # Dynamic ones get their actual positions via dynamic_obstacles (see below).
                _proj_cyls    = CYLINDERS
                pos_projector = build_position_projector(
                    horizon_H=horizon, gradient=gradient, device=device,
                    cylinders=_proj_cyls,
                    normalizer=None, tighten=vcfg["tighten"], dt=proj_dt,
                    use_dynamics=vcfg.get("use_dynamics", True),
                    drone_radius=drone_radius,
                    active_halfspaces=active_halfspaces,
                    goal_pull_weight=goal_pull_weight,
                )
                if vcfg["projection_mode"] == "sdpc":
                    # Enable proper in-loop SLSQP with obstacle constraints.
                    # pos0 is set per control step below (current drone position).
                    pos_projector.inloop_slsqp = True
                    pos_projector.action_normalizer = dataset.action_normalizer
                    pos_projector.pos0 = None  # filled in per step
            elif vcfg["use_projection"]:
                proj_dt = 0.1   # set here too  _rebuild_depth_projector below needs it

            print(f"\n[INFO] ===== Episode {ep+1}/{len(projection_variants)}_{variant_name} =====")
            episode_start_time = time.time()
            _ = env.reset(seed=ep)
            logger.begin_episode(variant_name, episode=ep, seed=ep)

            # Camera warm-up (important for Isaac/Replicator)
            for _ in range(3):
                pos = env._pos_world().detach().cpu().numpy()[0]
                cmd_xyz = pos.copy()   # hold position
                try:
                    env.step(cmd_xyz)
                except Exception:
                    pass

            if vcfg["use_projection"] and args.depth_obstacles:
                pos_projector, _init_static_pts, _init_dyn_preds = _rebuild_depth_projector(
                    pos[:3], env, args, eval_dt,
                    depth_fx, depth_fy, depth_cx, depth_cy,
                    depth_tracker, depth_static_accum, depth_dynamic_accum,
                    horizon, proj_dt, gradient, device, vcfg,
                    drone_radius, active_halfspaces, dataset,
                    goal_pull_weight=goal_pull_weight,
                )
                depth_static_pts_latest, depth_dyn_preds_latest = _init_static_pts, _init_dyn_preds
                print(f"[INFO] depth_obstacles: initial projector built from "
                      f"{len(_init_static_pts)} detected static keep-out points.")

            # init rgb(d) history
            rgb0 = get_obs_frame_from_env(env, use_depth)
            rgb_hist = deque(maxlen=To)
            for _ in range(To):
                rgb_hist.append((rgb0[0].copy(), rgb0[1].copy()) if use_depth else rgb0.copy())

            traj_xyz = []
            actions_taken = []
            frames_taken = []
            prev_actions_real = None
            cand_snapshots = []

            pos_init = env._pos_world().detach().cpu().numpy()[0]
            traj_xyz.append(pos_init.copy())
            if args.save_frames:
                frames_taken.append(rgb0[0].copy() if use_depth else rgb0.copy())

            for step in range(args.max_steps):
                # current position (for logging/command conversion)
                pos = env._pos_world().detach().cpu().numpy()[0]  # [x,y,z]
                _elapsed = time.strftime('%M:%S', time.gmtime(time.time() - episode_start_time))

                # ── diffusion path ────────────────────────────────────────────
                # build condition
                obs_rgb_t = preprocess_obs_stack(rgb_hist, use_depth).to(device)  # (1,To,3or4,H,W)
                cond = {"obs_rgb": obs_rgb_t}

                if use_pose_cond:
                    pose_now_norm = dataset.pose_normalizer.normalize(pos[:3].astype(np.float32))
                    pose_target_norm = dataset.pose_normalizer.normalize(pose_target_world)
                    cond["pose_now"] = torch.from_numpy(pose_now_norm).float().unsqueeze(0).to(device)
                    cond["pose_target"] = torch.from_numpy(pose_target_norm).float().unsqueeze(0).to(device)

                # -------------------------------------------------
                # Sample K candidate chunks (normalized action space)
                # -------------------------------------------------
                if vcfg["projection_mode"] == "sdpc" and pos_projector is not None:
                    pos_projector.pos0 = pos[:3]  # current drone position for in-loop coord conversion
                in_loop_projector = pos_projector if vcfg["projection_mode"] == "sdpc" else None

                # capture RNG state before sampling so the plain (unguided) resample
                # below (sdpc mode only) can reuse the exact same noise draw
                _rng_cpu = torch.get_rng_state()
                _rng_cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

                a_candidates_norm, _ = sample_action_candidates(diffusion=diffusion,cond=cond,horizon=horizon,action_dim=action_dim,
                                                                    num_candidates=vcfg["num_candidates"],projector=in_loop_projector)   # (K, H, D)

                # Unnormalize all candidates to real delta-pos, then apply action_scale
                # here  before projection/selection  so the projector validates
                # (and the selection scores) the trajectory that will actually be
                # executed, not an unscaled stand-in for it.
                a_candidates_real = dataset.action_normalizer.unnormalize(a_candidates_norm) * float(args.action_scale)   # (K, H, D)

                if vcfg["projection_mode"] == "sdpc":
                    torch.set_rng_state(_rng_cpu)
                    if _rng_cuda is not None:
                        torch.cuda.set_rng_state_all(_rng_cuda)
                    a_plain_norm, _ = sample_action_candidates(diffusion=diffusion, cond=cond, horizon=horizon, action_dim=action_dim,
                                                               num_candidates=vcfg["num_candidates"], projector=None)
                    a_plain_real = dataset.action_normalizer.unnormalize(a_plain_norm) * float(args.action_scale)
                else:
                    a_plain_real = a_candidates_real

                # ── per-step projector update ──────────
                if args.depth_obstacles and vcfg["use_projection"]:
                    # Always rebuild: unlike ground-truth cylinders, the detected
                    # point cloud changes every step regardless of whether any
                    # obstacle is actually moving (viewpoint, noise, newly-visible
                    # surfaces), so there's no "no-op, skip it" case here.
                    pos_projector, _static_pts_dbg, _dyn_preds_dbg = _rebuild_depth_projector(
                        pos[:3], env, args, eval_dt,
                        depth_fx, depth_fy, depth_cx, depth_cy,
                        depth_tracker, depth_static_accum, depth_dynamic_accum,
                        horizon, proj_dt, gradient, device, vcfg,
                        drone_radius, active_halfspaces, dataset,
                        goal_pull_weight=goal_pull_weight,
                    )
                    depth_static_pts_latest, depth_dyn_preds_latest = _static_pts_dbg, _dyn_preds_dbg

                _need_rebuild = args.dynamic_obstacles_enabled
                if _need_rebuild and vcfg["use_projection"] and not args.depth_obstacles:
                    # get_cylinder_positions() returns current positions for ALL cylinders
                    # (static ones at rest, dynamic ones at actual current position)
                    cyl_now = env.get_cylinder_positions()
                    # Predict each dynamic cylinder's exact future (x,y) at every planned
                    # horizon step (t = now + k*proj_dt), using the sim's own closed-form
                    # sinusoid:so the projector avoids where the obstacle WILL be, not
                    # just where it is right now. This is what makes `proj_dt` matter.
                    dyn_cyl_preds = (
                        env.predict_cylinder_positions([k * proj_dt for k in range(1, horizon + 1)])
                        if args.dynamic_obstacles_enabled else None
                    )
                    pos_projector = build_position_projector(
                        horizon_H=horizon, gradient=gradient, device=device,
                        cylinders=cyl_now,
                        normalizer=None, tighten=vcfg["tighten"], dt=proj_dt,
                        use_dynamics=vcfg.get("use_dynamics", True),
                        obs_amplitude=0.0,  # exact positions:no extra radius needed
                        drone_radius=drone_radius,
                        active_halfspaces=active_halfspaces,
                        dynamic_cylinder_predictions=dyn_cyl_preds,
                        goal_pull_weight=goal_pull_weight,
                    )
                    if vcfg["projection_mode"] == "sdpc":
                        pos_projector.inloop_slsqp = True
                        pos_projector.action_normalizer = dataset.action_normalizer
                        pos_projector.pos0 = pos[:3]

                # ── projection + selection  -
                if vcfg["use_projection"] and vcfg["projection_mode"] in ("post", "sdpc"):
                    a_candidates_proj_real, proj_costs = project_action_candidates_with_positions(
                        projector=pos_projector, pos0=pos[:3],
                        a_candidates_real=a_candidates_real, device=device,
                    )
                else:
                    a_candidates_proj_real = a_candidates_real
                    proj_costs = None

                if vcfg["selection"] == "minimum_projection_cost" and proj_costs is not None:
                    which = int(np.argmin(proj_costs))
                else:
                    which = choose_trajectory(
                        a_candidates_proj_real,
                        strategy=vcfg["selection"],
                        prev_actions_real=prev_actions_real,
                    )

                a0_real = a_candidates_proj_real[which, 0]  # action_scale already applied pre-projection
                prev_actions_real = a_candidates_proj_real[which:which+1]

                # Convert delta-pos -> absolute command (your env seems to accept xyz setpoints)
                cmd_xyz = pos.copy()
                cmd_xyz[:action_dim] = cmd_xyz[:action_dim] + a0_real[:action_dim]
                # cmd_xyz[2] = 0.3 # If you want fixed altitude, uncomment:
                print(f"[MODEL OUTPUT] which={which} a0_real={a0_real}" f" cmd_xyz={cmd_xyz[:3]}")
                obs_next, _rew, done_vec, info = env.step(cmd_xyz)

                # update rgb(d) history after step
                rgb = get_obs_frame_from_env(env, use_depth)
                rgb_hist.append(rgb)
                if args.save_frames:
                    frames_taken.append(rgb[0].copy() if use_depth else rgb.copy())

                # log
                pos2 = obs_next[0] 
                traj_xyz.append(pos2.copy())
                actions_taken.append(a0_real.copy())

                logger.step(pos=pos, action=a0_real)

                done = bool(done_vec[0]) if isinstance(done_vec, (list, tuple, np.ndarray, torch.Tensor)) else bool(done_vec)
                print(f"{_elapsed} step {step:04d} pos={pos2} done={done}")

                #"step": the step index (int) in this episode.
                #"pos": pos2.copy():the drone's actual (x,y,z) position after this step actually executed (obs_next[0], from
                #env.step(cmd_xyz)). Not a prediction:where the drone really ended up.
                #"traj_xy": cand_xyz, shape (K, H+1, 3):the raw model output, ALL K sampled candidates, each integrated forward
                #from pos2 (cumulative sum of that candidate's per-step deltas) to get its predicted H-step future trajectory.
                #Despite the name, it's full (x,y,z), not just xy. This is before any safety projection.
                #"traj_xy_proj": cand_xyz_proj, shape (H+1, 3):just for the chosen candidate (index which), its trajectory
                #after SLSQP safety projection:i.e. what the projector actually corrected it to for constraint satisfaction
                #(obstacles/bounds/keep-out zones/halfspaces). This is the "safe plan."
                #"traj_xy_plain": cand_xyz_plain, shape (H+1, 3):the same candidate slot and same noise draw, but from a
                #second, unguided sampling pass (a_plain_real) with no in-loop SLSQP guidance during denoising. Lets you visually
                #compare "what the model would have predicted without SDPC guidance" against the guided/projected version.
                #"chosen": int(which):which of the K candidates was actually selected and executed this step (per
                #vcfg["selection"]'s strategy:first, min-projection-cost, etc.).
                #"cyl_xy": only set if dynamic_obstacles is on:the cylinders' actual current (x,y) positions this step (they
                #move), for drawing the swept motion band later. None otherwise.
                #"depth_static_pts" / "depth_dyn_preds": only set if depth_obstacles is on:that step's snapshot of what the
                #depth camera currently detects as static keep-out points / tracked dynamic-obstacle predictions. None otherwise.

                cand_xyz = integrate_candidates_xyz(pos2, a_candidates_real)  # (K,H+1,3)
                cand_xyz_proj = integrate_candidates_xyz(
                    pos2, a_candidates_proj_real[which:which + 1]
                )[0]  # (H+1,3)
                cand_xyz_plain = integrate_candidates_xyz(
                    pos2, a_plain_real[which:which + 1]
                )[0]  # (H+1,3)
                cand_snapshots.append({
                    "step": step,
                    "pos": pos2.copy(),
                    "traj_xy": cand_xyz,
                    "traj_xy_proj": cand_xyz_proj,
                    "traj_xy_plain": cand_xyz_plain,
                    "chosen": int(which),
                    "cyl_xy":  env.get_cylinder_positions() if args.dynamic_obstacles_enabled else None,
                    "depth_static_pts": list(depth_static_pts_latest) if args.depth_obstacles else None,
                    "depth_dyn_preds":  dict(depth_dyn_preds_latest)  if args.depth_obstacles else None,
                })

                if done:
                    print("[INFO] Done=True. Breaking episode loop.")
                    break

            episode_wall_time_sec = time.time() - episode_start_time
            print(f"[INFO] Episode '{variant_name}' (seed {seedmodel}) took "
                  f"{episode_wall_time_sec / 60.0:.2f} minutes ({episode_wall_time_sec:.1f}s)")

            success = bool(info["success"][0])
            fell    = bool(info["fell"][0])
            logger.end_episode(success=success, fell=fell, wall_time_sec=episode_wall_time_sec)
            if (ep + 1) % 5 == 0:
                logger.print_live_summary()
        
            # save raw trajectory and metadata for this episode
            traj_path = os.path.join(
                traj_dir,
                f"traj_{encoder_type}_L{latent_dim}_{variant_name}_seed{seedmodel}.npz"
            )
            if args.save_frames:
                frames_path = os.path.join(
                    traj_dir,
                    f"frames_{encoder_type}_L{latent_dim}_{variant_name}_seed{seedmodel}.npy"
                )
                np.save(frames_path, np.array(frames_taken, dtype=np.uint8))  # (T,H,W,3)
                print(f"[INFO] Saved {len(frames_taken)} frames -> {frames_path}")
            np.savez(
                traj_path,
                xyz        = np.array(traj_xyz),        # (T, 3) full trajectory
                actions    = np.array(actions_taken),    # (T, 3) actions taken
                variant    = variant_name,
                encoder    = encoder_type,
                latent_dim = latent_dim,
                success    = success,
                fell       = fell,
                episode    = ep,
                episode_wall_time_sec = float(episode_wall_time_sec),
                episode_wall_time_min = float(episode_wall_time_sec) / 60.0,
                cylinders   = np.array(CYLINDERS),       # (N, 2) cylinder centers
                halfspaces  = np.array([[hs[0], hs[1]] for hs in active_halfspaces], dtype=object),
                hs_sides    = np.array([hs[2] for hs in active_halfspaces]),
                tighten     = vcfg.get("tighten", 0.0),
                use_projection = vcfg["use_projection"],
                projection_mode = vcfg["projection_mode"],
                num_candidates = vcfg["num_candidates"] if vcfg["num_candidates"] > 0 else 1,
                selection   = vcfg["selection"],
                depth_obstacles = bool(args.depth_obstacles),
                depth_obstacle_max_range = (float(args.depth_obstacle_max_range)if args.depth_obstacles else 0.0),
                depth_static_points = (np.array(list(depth_static_accum.values()), dtype=np.float32)
                                       if args.depth_obstacles else np.zeros((0, 2), dtype=np.float32)),
                depth_dynamic_points = (np.array(list(depth_dynamic_accum.values()), dtype=np.float32)
                                        if args.depth_obstacles else np.zeros((0, 2), dtype=np.float32)),
                depth_static_pts_traj = np.array(
                    [np.array(s.get("depth_static_pts") or [], dtype=np.float32) for s in cand_snapshots],
                    dtype=object,
                ) if args.depth_obstacles else np.array([], dtype=object),
                depth_dynamic_pts_traj = np.array(
                    [np.array([c[0] for c in (s.get("depth_dyn_preds") or {}).values() if c],
                              dtype=np.float32)
                     for s in cand_snapshots],
                    dtype=object,
                ) if args.depth_obstacles else np.array([], dtype=object),
                snap_pos     = np.array([s["pos"]     for s in cand_snapshots]),
                snap_chosen  = np.array([s["chosen"]  for s in cand_snapshots]),
                cand_traj_xy = np.array([s["traj_xy"] for s in cand_snapshots]),  # (N_snap, K, H+1, 3) all candidate rollouts, now with z
                cand_traj_xy_proj = np.array([s["traj_xy_proj"] for s in cand_snapshots]),  # (N_snap, H+1, 3) projected/safe horizon of the chosen candidate
                cand_traj_xy_plain = np.array([s["traj_xy_plain"] for s in cand_snapshots]),  # (N_snap, H+1, 3) plain/unguided model output, same noise draw as chosen candidate

                # ── dynamic cylinder positions per step (T, N_cyl, 2) ──
                cyl_xy_traj = np.array(
                    [s["cyl_xy"] for s in cand_snapshots if s.get("cyl_xy") is not None],
                    dtype=np.float32,
                ) if any(s.get("cyl_xy") is not None for s in cand_snapshots) else np.zeros((0, len(CYLINDERS), 2), dtype=np.float32),
                dynamic_cyl_indices = np.array(
                    args.dynamic_cyl_indices if args.dynamic_cyl_indices is not None else list(range(len(CYLINDERS))),
                    dtype=np.int32,
                ) if args.dynamic_obstacles_enabled else np.array([], dtype=np.int32),
                obs_amplitude  = float(obs_amplitude),
                obs_frequency  = float(obs_frequency),
                obs_axes = np.array(
                    args.obs_axes if args.obs_axes is not None
                    else (["y"] * len(CYLINDERS) if args.dynamic_obstacles_enabled else []),
                    dtype="U2",
                ),
                dynamic_obstacles = bool(args.dynamic_obstacles_enabled),
            )
            print(f"[TRAJ] saved: {traj_path}")

            # ------------------ Plot episode ------------------
            if len(traj_xyz) > 0:
                traj_xyz_np = np.stack(traj_xyz, axis=0).astype(np.float32)

                # Z vs t
                plt.figure(figsize=(7, 4))
                plt.plot(traj_xyz_np[:, 2])
                plt.ylim(-0.05, 2.15)   # fix to flight envelope [0, 1] with small padding
                plt.axhline(0.0, color="#888888", linewidth=0.8, linestyle=":")   # floor
                plt.axhline(2.0, color="#888888", linewidth=0.8, linestyle=":")   # ceiling
                plt.xlabel("timestep")
                plt.ylabel("z  (m)")
                plt.title(f"Z over time with {encoder_type}, {latent_dim} and {variant_name}")
                plt.tight_layout()

                z_path =os.path.join(plot_dir, f"z_{variant_name}.pdf")
                plt.savefig(z_path)
                plt.close()
                print(f"[PLOT] saved: {z_path}")

                # XY plot (one figure, one axes)
                xy_exec = traj_xyz_np[:, :2]
                fig, ax = plt.subplots(figsize=(8, 7))

                # Executed trajectory
                ax.plot(xy_exec[:, 0], xy_exec[:, 1], linewidth=2.5, label="executed")
                ax.scatter(pos_init[0], pos_init[1], marker="o", s=70, color="green", zorder=5, label="start")
                ax.scatter(xy_exec[-1, 0], xy_exec[-1, 1], marker="x", s=60, label="end")

                # Obstacles overlay
                depth_legend_handles = []
                if args.depth_obstacles:
                    # Ground truth shown faint/dashed for reference only  the
                    # projector never saw this, it is NOT what was enforced.
                    for (gx, gy) in CYLINDERS:
                        ax.add_patch(Circle((gx, gy), CYL_RADIUS, linewidth=1.0, linestyle="--",
                                             edgecolor="gray", facecolor="none", alpha=0.4, zorder=2))
                    # Every point detected this episode (voxel-deduped accumulator,
                    # see depth_static_accum/depth_dynamic_accum above)  NOT just
                    # what any single step's capped projector constraints used.
                    if depth_static_accum:
                        _sa = np.array(list(depth_static_accum.values()))
                        ax.scatter(_sa[:, 0], _sa[:, 1], s=6, color="tab:red", alpha=0.35, zorder=3)
                    if depth_dynamic_accum:
                        _da = np.array(list(depth_dynamic_accum.values()))
                        ax.scatter(_da[:, 0], _da[:, 1], s=6, color="tab:purple", alpha=0.35, zorder=3)
                    depth_legend_handles = [
                        mpatches.Patch(color="tab:red", alpha=0.5, label="depth-detected static point"),
                        mpatches.Patch(color="tab:purple", alpha=0.5, label="depth-detected dynamic point"),
                        mpatches.Patch(facecolor="none", edgecolor="gray", label="ground truth (reference only)"),
                    ]
                elif args.dynamic_obstacles_enabled:
                    dyn_idx = args.dynamic_cyl_indices if args.dynamic_cyl_indices is not None \
                              else list(range(len(CYLINDERS)))
                    _raw_axes = args.obs_axes if args.obs_axes is not None else ["y"]
                    dyn_axes_resolved = (_raw_axes * len(dyn_idx))[:len(dyn_idx)] if len(_raw_axes) == 1 else _raw_axes
                    dyn_set = set(dyn_idx)
                    static_cyls = [CYLINDERS[i] for i in range(len(CYLINDERS)) if i not in dyn_set]
                    dyn_cyls    = [CYLINDERS[i] for i in dyn_idx]
                    add_obstacles_xy(ax, static_cyls, cyl_radius=CYL_RADIUS)
                    add_dynamic_cylinders_xy(
                        ax, dyn_cyls, dyn_axes_resolved, cand_snapshots,
                        obs_amplitude=obs_amplitude, cyl_radius=CYL_RADIUS,
                        drone_radius=drone_radius,
                    )
                else:
                    add_obstacles_xy(ax, CYLINDERS, cyl_radius=CYL_RADIUS)

                # Keep-out zones overlay (virtual, planner-only  see KEEPOUT_ZONES
                # in config/avoiding-crazyflie.py)
                for (kx, ky, kr) in KEEPOUT_ZONES:
                    ax.add_patch(Circle(
                        (kx, ky), kr,
                        linewidth=1.2, edgecolor="crimson", facecolor="crimson",
                        alpha=0.15, linestyle="--", zorder=2,
                    ))

                ax.set_xlim(-6.5, 4.5)
                ax.set_ylim(-2.0, 2.0)
                if active_halfspaces:
                    plot_halfspace_constraints_xy(ax, active_halfspaces, (-0.5, 4.5), (-1.0, 1.0))

                #  blue constraint margin overlay 
                # Skipped for depth_obstacles: it draws the ground-truth CYLINDERS'
                # margin, which is exactly what this run did NOT enforce  the red/
                # purple detected-point scatter above is the honest equivalent.
                tighten_val = vcfg.get("tighten", 0.0)
                constraint_handles = list(depth_legend_handles)
                if vcfg["use_projection"] and not args.depth_obstacles:
                    if args.dynamic_obstacles_enabled:
                        _dyn_set_plot = set(args.dynamic_cyl_indices) if args.dynamic_cyl_indices is not None \
                                        else set(range(len(CYLINDERS)))
                        _static_c = [CYLINDERS[i] for i in range(len(CYLINDERS)) if i not in _dyn_set_plot]
                        # dynamic cylinders: no blue circle:orange swept band already shows range
                        constraint_handles = plot_constraint_overlay(
                            ax, _static_c,
                            tighten=tighten_val, drone_radius=drone_radius,
                            x_bounds=(-6.5, 4.5), y_bounds=(-2.0, 2.0),
                        )
                    else:
                        constraint_handles = plot_constraint_overlay(
                            ax, CYLINDERS,
                            tighten=tighten_val,
                            drone_radius=drone_radius,
                            x_bounds=(-6.5, 4.5),
                            y_bounds=(-2.0, 2.0),
                        )
                # --------------------------------------------

                # Overlay candidate rollout snapshot start positions
                for snap in cand_snapshots:
                    ax.scatter(snap["pos"][0], snap["pos"][1], s=12, alpha=0.4)

                print("[DEBUG] cand_snapshots count:", len(cand_snapshots))
                if len(cand_snapshots) > 0:
                    print("[DEBUG] first snapshot traj_xy shape:", cand_snapshots[0]["traj_xy"].shape)

                ax.set_xlabel("x")
                ax.set_ylabel("y")
                ax.set_title(f"XY trajectory with{encoder_type}, {latent_dim} and {variant_name}")  #and {variant_name}
                ax.set_aspect("equal", adjustable="box")  # equal scale without expanding y beyond corridor
                ax.set_xlim(-6.5, 4.5)
                ax.set_ylim(-2.25, 2.25)
                ax.grid(True, alpha=0.3)
                ax.legend()

                # merge legend handles
                handles, _ = ax.get_legend_handles_labels()
                hs_legend = [mpatches.Patch(facecolor="royalblue", alpha=0.3,
                                            label="halfspace constraint")] if active_halfspaces else []
                ax.legend(handles=handles + constraint_handles + hs_legend,
                          loc="upper left", fontsize=8)
            
                out_path = os.path.join(plot_dir, f"xy_{variant_name}.pdf")
                fig.tight_layout()
                fig.savefig(out_path)
                # plt.show()
                plt.close(fig)
                print("[PLOT] saved:", out_path)

            env.reset()
        logger.save()

    total_wall_time_sec = time.time() - run_start_time
    print(f"[INFO] Total eval run time: {total_wall_time_sec / 60.0:.2f} minutes "
          f"({total_wall_time_sec:.1f}s)")

    env.close()
    os._exit(0)


if __name__ == "__main__":
    main()