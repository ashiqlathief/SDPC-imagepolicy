import argparse
import importlib
import cv2
from collections import deque
import time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle, Circle, Polygon

import diffuser.utils as utils
from diffuser.sampling.projection import Projector
from diffuser.sampling.policies import temporal_consistency_distances
from depth_obstacle_estimator import FPV_WIDTH, FPV_HEIGHT
cfg = importlib.import_module("config.avoiding-crazyflie")
BOXES = cfg.BOXES
CYLINDERS = cfg.CYLINDERS

corridor_halfspaces = cfg.CORRIDOR_HALFSPACES
DEPTH_NEAR = cfg.DEPTH_NEAR
DEPTH_FAR = cfg.DEPTH_FAR

# Z flight-envelope halfspaces.
# Format: (normal_3d, rhs)  →  normal · [x, y, z] <= rhs
# tighten shrinks the envelope from both ends (ceiling down, floor up).
# WALL_HEIGHT = 1.0   # metres — obstacle / wall tops in the corridor
z_halfspaces = [
    ([0.0, 0.0,  1.0], 1.0),   # z <=  1.0 m  — drone cannot fly above wall/ceiling
    ([0.0, 0.0, -1.0], 0.0),           # z >= 0.0 m   — drone cannot go underground
]

projection_variants = [
#   'sdpc-r',
#   'sdpc-r-tightened',
#   'sdpc-c',
#   'sdpc-c-tightened',
#   'sdpc-t',
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
        cfg.update(num_candidates=8, selection="minimum_projection_cost", use_projection=True, projection_mode="sdpc", tighten=0.0)

    elif name == "sdpc-c-tightened":
        cfg.update(num_candidates=8, selection="minimum_projection_cost", use_projection=True, projection_mode="sdpc", tighten=0.05)

    elif name == "sdpc-t":
        cfg.update(num_candidates=8, selection="temporal_consistency", use_projection=True, projection_mode="sdpc", tighten=0.0)

    elif name == "sdpc-t-tightened":
        cfg.update(num_candidates=8, selection="temporal_consistency", use_projection=True, projection_mode="sdpc", tighten=0.05)

    elif name == "sdpc-r":
        # sdpc-r often means single sample with projection (repair)
        cfg.update(num_candidates=1, selection="first", use_projection=True, projection_mode="sdpc", tighten=0.0)

    elif name == "sdpc-r-tightened":
        cfg.update(num_candidates=1, selection="first", use_projection=True, projection_mode="sdpc", tighten=0.05)

    return cfg

def plot_constraint_overlay(ax, boxes, cylinders, tighten=0.03, drone_radius=0.0,
                            x_bounds=(-0.5, 4.5), y_bounds=(-1.0, 1.0)):
    """
    Overlays safety margins as blue-shaded regions on an existing XY axes.
    Shows the actual exclusion zone the projector enforces.
    """
    box_r = 0.15 + drone_radius + tighten   # must match build_obstacle_constraint_list
    cyl_r = 0.06 + drone_radius + tighten

    # --- obstacle margins (blue circles) ---
    for (x, y) in boxes:
        # raw obstacle (already drawn in red by add_obstacles_xy)
        # safety margin ring
        ax.add_patch(plt.Circle((x, y), box_r,
                                color="royalblue", alpha=0.15, zorder=2))
        ax.add_patch(plt.Circle((x, y), box_r,
                                fill=False, edgecolor="royalblue",
                                linewidth=1.2, linestyle="--", zorder=3))

    for (x, y) in cylinders:
        ax.add_patch(plt.Circle((x, y), cyl_r,
                                color="royalblue", alpha=0.15, zorder=2))
        ax.add_patch(plt.Circle((x, y), cyl_r,
                                fill=False, edgecolor="royalblue",
                                linewidth=1.2, linestyle="--", zorder=3))

    # --- corridor bounds (blue shaded strips outside the allowed region) ---
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
      - 'below' : y <= m x + b
      - 'above' : y >= m x + b
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

def add_obstacles_xy(ax, boxes, cylinders, box_size_xy=0.20, cyl_radius=0.06):
    # Boxes as squares in XY
    for x, y in boxes:
        ax.add_patch(
            Rectangle(
                (x - box_size_xy/2, y - box_size_xy/2),
                box_size_xy, box_size_xy,
                linewidth=1.0,
                edgecolor="black",
                facecolor="tab:red",
                alpha=0.25,
            )
        )
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
                              cyl_radius=0.06, drone_radius=0.0,
                              x_clamp=(0.3, 4.7), y_clamp=(-0.85, 0.85)):
    """
    Draw dynamic cylinder visualisation on an XY axes:
      - orange band  = oscillation sweep of the exclusion zone, shaped per cylinder's
                        motion axis ("y": vertical band, "x": horizontal band,
                        "xy": square bounding box — diagonal motion is a 1D line through
                        it, so the square is a conservative over-approximation, not exact)
      - dashed circle = exclusion zone at rest position
      - faded dots    = physical cylinder position sampled every few steps
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
        else:  # "xy" — same delta applied to both axes → 45° diagonal line sweep
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
      - rgb: (H,W,3) uint8                          if use_depth=False
      - (rgb, depth): (H,W,3) uint8, (H,W) float32  if use_depth=True
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

def build_obstacle_constraint_list(boxes, cylinders, spheres=None, x_bounds=None,
                                    y_bounds=None, z_bounds=None,
                                    corridor_halfspaces=None, z_halfspaces=None, tighten=0.0,
                                    cyl_extra_radius=0.0, drone_radius=0.0,
                                    sphere_radius=0.0, dynamic_cylinder_predictions=None):
    """
    dynamic_cylinder_predictions: optional {cylinder_index: [(x,y), ...]} — one
    predicted (x,y) per horizon step (in order), from env.predict_cylinder_positions().
    Cylinders with an entry here get a per-timestep obstacle constraint (the
    projector avoids where the obstacle WILL be at each planned step); all others
    keep the usual single current-position constraint applied across the horizon.
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
    for (x, y) in boxes:
        center = [float(x), float(y)]
        constraint_list.append(("sphere_outside", [0, 1], center, 0.15 + _dr + float(tighten)))

    for i, (x, y) in enumerate(cylinders):
        radius = 0.07 + _dr + float(tighten) + float(cyl_extra_radius)
        if dynamic_cylinder_predictions is not None and i in dynamic_cylinder_predictions:
            centers_per_t = [[float(cx), float(cy)] for cx, cy in dynamic_cylinder_predictions[i]]
            constraint_list.append(("sphere_outside_dynamic", [0, 1], centers_per_t, radius))
        else:
            center = [float(x), float(y)]
            constraint_list.append(("sphere_outside", [0, 1], center, radius))

    # 3D sphere constraints (floating obstacles)
    if spheres:
        _sr = float(sphere_radius)
        for (x, y, z) in spheres:
            constraint_list.append(("sphere_outside", [0, 1, 2],
                                     [float(x), float(y), float(z)],
                                     _sr + _dr + float(tighten)))

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
    # Floor:   [0,0,-1]·p <= 0.0 - tighten        (shrinks floor up — use tighten=0 here)
    if z_halfspaces is not None:
        for (normal, rhs) in z_halfspaces:
            C_row = np.zeros(state_dim, dtype=np.float32)
            C_row[:3] = normal
            constraint_list.append(("ineq", (C_row, float(rhs) - float(tighten))))

    return constraint_list

def verify_projection(pos0, a_candidates_proj_real, which,
                      x_bounds=(-0.5, 4.5), y_bounds=(-0.95, 0.95), z_bounds=(0.0, 1.0),
                      boxes=None, cylinders=None, tighten=0.0, step=None):
    """
    Check whether the chosen projected trajectory actually satisfies all bounds.
    Prints a warning line for every violation found.

    Call this right after project_action_candidates_with_positions() to confirm
    the projector is working — if no warnings appear the bounds are being enforced.
    """
    _, H, _ = a_candidates_proj_real.shape
    traj = np.zeros((H + 1, 3), dtype=np.float32)
    traj[0]  = pos0.astype(np.float32)
    traj[1:] = traj[0] + np.cumsum(a_candidates_proj_real[which], axis=0)

    prefix = f"[VERIFY step={step}]" if step is not None else "[VERIFY]"
    violations = []

    for t in range(1, H + 1):   # skip t=0 (skip_initial_state)
        x, y, z = traj[t]

        if x < x_bounds[0] - 1e-4:
            violations.append(f"  t={t}: x={x:.4f} < x_min={x_bounds[0]}")
        if x > x_bounds[1] + 1e-4:
            violations.append(f"  t={t}: x={x:.4f} > x_max={x_bounds[1]}")
        if y < y_bounds[0] - 1e-4:
            violations.append(f"  t={t}: y={y:.4f} < y_min={y_bounds[0]}")
        if y > y_bounds[1] + 1e-4:
            violations.append(f"  t={t}: y={y:.4f} > y_max={y_bounds[1]}")
        if z < z_bounds[0] - 1e-4:
            violations.append(f"  t={t}: z={z:.4f} < z_min={z_bounds[0]}")
        if z > z_bounds[1] + 1e-4:
            violations.append(f"  t={t}: z={z:.4f} > z_max={z_bounds[1]}")

        if boxes is not None:
            for (bx, by) in boxes:
                dist = np.hypot(x - bx, y - by)
                if dist < 0.15 + tighten - 1e-4:
                    violations.append(f"  t={t}: inside box ({bx},{by}), dist={dist:.4f}")
        if cylinders is not None:
            for (cx, cy) in cylinders:
                dist = np.hypot(x - cx, y - cy)
                if dist < 0.06 + tighten - 1e-4:
                    violations.append(f"  t={t}: inside cyl ({cx},{cy}), dist={dist:.4f}")

    if violations:
        print(f"{prefix} BOUND VIOLATIONS in projected traj (candidate {which}):")
        for v in violations:
            print(v)
    else:
        print(f"{prefix} OK — all bounds satisfied in projected traj")


def build_position_projector(horizon_H, gradient, device, boxes, cylinders, spheres=None,
                             normalizer=None, tighten=0.0, dt=0.1, use_dynamics=True,
                             obs_amplitude=0.0, drone_radius=0.0, sphere_radius=0.0,
                             active_halfspaces=None, dynamic_cylinder_predictions=None):
    # We project POSITIONS, so we need horizon = H+1
    Hp1 = horizon_H + 1
    x_bounds = (-0.5, 4.5)
    y_bounds = (-0.95, 0.95)
    z_bounds = (0.0, 1.0)
    transition_dim = 3

    constraint_list = build_obstacle_constraint_list(
        boxes=boxes,
        cylinders=cylinders,
        spheres=spheres,
        x_bounds=x_bounds,
        y_bounds=y_bounds,
        z_bounds=z_bounds,
        corridor_halfspaces=active_halfspaces or [],
        z_halfspaces=z_halfspaces,
        tighten=tighten,
        cyl_extra_radius=obs_amplitude,
        drone_radius=drone_radius,
        sphere_radius=sphere_radius,
        dynamic_cylinder_predictions=dynamic_cylinder_predictions,
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
        parallelize=True,         # candidates solve independently — run them concurrently
    )

    return projector

def build_point_obstacle_constraints(static_points, dynamic_predictions, radius):
    """--depth_obstacles counterpart to the cylinder loop inside
    build_obstacle_constraint_list(): same ("sphere_outside", ...) /
    ("sphere_outside_dynamic", ...) tuple shapes, one entry per detected
    surface point instead of one per fitted circle (see
    depth_obstacle_estimator.py's module docstring -- "option b")."""
    constraints = []
    for (x, y) in static_points:
        constraints.append(("sphere_outside", [0, 1], [float(x), float(y)], float(radius)))
    for centers_per_t in dynamic_predictions.values():
        constraints.append(("sphere_outside_dynamic", [0, 1],
                             [[float(cx), float(cy)] for cx, cy in centers_per_t],
                             float(radius)))
    return constraints

def build_position_projector_from_points(horizon_H, gradient, device, boxes,
                                          static_points, dynamic_predictions, point_radius,
                                          spheres=None, normalizer=None, tighten=0.0, dt=0.1,
                                          use_dynamics=True, drone_radius=0.0, sphere_radius=0.0,
                                          active_halfspaces=None):
    """--depth_obstacles counterpart to build_position_projector(): identical
    bounds/box/sphere/halfspace setup (reuses build_obstacle_constraint_list
    unchanged, with cylinders=[] so no ground-truth cylinder constraints get
    added), but cylinder obstacles come from a detected point cloud (see
    depth_obstacle_estimator.py) instead of CYLINDERS / env.get_cylinder_positions().
    """
    Hp1 = horizon_H + 1
    x_bounds = (-0.5, 4.5)
    y_bounds = (-0.95, 0.95)
    z_bounds = (0.0, 1.0)
    transition_dim = 3

    constraint_list = build_obstacle_constraint_list(
        boxes=boxes,
        cylinders=[],
        spheres=spheres,
        x_bounds=x_bounds,
        y_bounds=y_bounds,
        z_bounds=z_bounds,
        corridor_halfspaces=active_halfspaces or [],
        z_halfspaces=z_halfspaces,
        tighten=tighten,
        drone_radius=drone_radius,
        sphere_radius=sphere_radius,
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


# def run_hardware_ros1(args, run_dir, device, drone_radius, active_halfspaces):
#     """
#     --mode ros2: live hardware deployment. Named "ros2" on the CLI for
#     consistency with how the lab talks about this, but the actual transport
#     mirrors flight_test.py exactly: ROS1 (rospy) + mavros for pose, and (as of
#     this version) color + aligned depth also come in over ROS1 sensor_msgs/Image
#     topics (e.g. published by realsense-ros's ROS1 driver) instead of grabbing
#     the RealSense SDK directly -- same one-ROS-version approach as pose, so
#     this camera can live on a different node/machine than this process.

#     Runs ONE model on ONE variant, no seed sweep, no Isaac Sim, no
#     ground-truth obstacle logging -- those only make sense in --mode sim.
#     """
#     try:
#         import rospy
#         from geometry_msgs.msg import PoseStamped, TwistStamped
#         from sensor_msgs.msg import Image
#     except ImportError as e:
#         raise RuntimeError(
#             "--mode ros2 requires rospy, geometry_msgs, and sensor_msgs -- "
#             "source the lab's ROS1 workspace first (same deps as flight_test.py)."
#         ) from e

#     # ------------------ Load trained model (single run_dir, no sweep) ------------------
#     print(f"\n[INFO] Loading run dir: {run_dir}")
#     diff_exp = utils.load_diffusion(run_dir, epoch="best", device=str(device))
#     dataset = diff_exp.dataset
#     diffusion = diff_exp.diffusion.to(device)
#     diffusion.eval()

#     use_depth = bool(getattr(diffusion.model, "use_depth", False))
#     horizon = int(getattr(diffusion, "horizon", 16))
#     action_dim = int(getattr(diffusion, "action_dim", 3))
#     To = int(getattr(dataset, "n_obs_steps", 2))
#     control_dt = float(getattr(dataset, "control_dt", getattr(dataset, "dt", 0.1)))
#     print(f"[INFO] Hardware eval: To={To} H={horizon} action_dim={action_dim} "
#           f"use_depth={use_depth} control_dt={control_dt}s")

#     # ------------------ Optional static obstacle projector (ground-truth BOXES/
#     # CYLINDERS from config.avoiding-crazyflie.py -- assumes the real corridor
#     # matches that layout; no dynamic-obstacle or depth-obstacle support here) ---
#     vcfg = variant_cfg(args.variant)
#     if args.num_candidates is not None and vcfg["selection"] != "first":
#         vcfg["num_candidates"] = args.num_candidates
#     gradient = (vcfg["projection_mode"] == "gradient")
#     pos_projector = None
#     if vcfg["use_projection"]:
#         proj_dt = 0.1
#         pos_projector = build_position_projector(
#             horizon_H=horizon, gradient=gradient, device=device,
#             boxes=BOXES, cylinders=CYLINDERS, spheres=None,
#             normalizer=None, tighten=vcfg["tighten"], dt=proj_dt,
#             drone_radius=drone_radius, sphere_radius=0.0,
#             active_halfspaces=active_halfspaces,
#         )
#         if vcfg["projection_mode"] == "sdpc":
#             pos_projector.inloop_slsqp = True
#             pos_projector.action_normalizer = dataset.action_normalizer
#             pos_projector.pos0 = None
#     print(f"[INFO] Variant: {args.variant} (projection={vcfg['use_projection']}, "
#           f"mode={vcfg['projection_mode']})")

#     if not args.live:
#         print("[WARN] --live not set: DRY RUN. Velocity commands will be computed "
#               "and printed but NOT published. Pass --live to actually fly.")

#     # ------------------ ROS1 node (must exist before any Subscriber/Publisher) ------
#     rospy.init_node("diffusion_policy_hardware", anonymous=True)

#     # ------------------ ROS1 camera topics (color + aligned depth) ------------------
#     # Same decode conventions as depth_camera_live_test.py's ROS2 _depth_cb (16UC1
#     # raw mm -> metres, 32FC1 already metres), just over rospy instead of rclpy so
#     # it shares one ROS version with the pose subscriber below. Frames are cached
#     # by the subscriber callbacks and grab_frame() just reads the latest one --
#     # same pattern as how `state["pos"]` is read from pose_cb, not fetched inline.
#     cam_state = {"color": None, "depth": None}

#     def color_cb(msg):
#         if msg.encoding not in ("rgb8", "bgr8"):
#             rospy.logwarn_throttle(5.0, f"unsupported color encoding '{msg.encoding}', skipping frame")
#             return
#         frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
#         if msg.encoding == "bgr8":
#             frame = frame[:, :, ::-1]
#         cam_state["color"] = cv2.resize(frame, (FPV_WIDTH, FPV_HEIGHT), interpolation=cv2.INTER_AREA)

#     def depth_cb(msg):
#         if msg.encoding == "16UC1":
#             depth = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
#             depth = depth.astype(np.float32) * 0.001  # RealSense: raw mm -> metres
#         elif msg.encoding == "32FC1":
#             depth = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
#         else:
#             rospy.logwarn_throttle(5.0, f"unsupported depth encoding '{msg.encoding}', skipping frame")
#             return
#         cam_state["depth"] = cv2.resize(depth, (FPV_WIDTH, FPV_HEIGHT), interpolation=cv2.INTER_NEAREST)

#     rospy.Subscriber(args.color_topic, Image, color_cb, queue_size=1)
#     if use_depth:
#         rospy.Subscriber(args.depth_topic, Image, depth_cb, queue_size=1)

#     def cam_ready():
#         return cam_state["color"] is not None and (not use_depth or cam_state["depth"] is not None)

#     def grab_frame():
#         if not use_depth:
#             return cam_state["color"]
#         return cam_state["color"], cam_state["depth"]

#     # ------------------ pose in, velocity setpoint out (mirrors flight_test.py's
#     # /mavros/local_position/pose -> /mavros/setpoint_velocity/cmd_vel) ------------
#     state = {"pos": None}

#     def pose_cb(msg):
#         p = msg.pose.position
#         state["pos"] = np.array([p.x, p.y, p.z], dtype=np.float32)

#     rospy.Subscriber(args.pose_topic, PoseStamped, pose_cb)
#     cmd_pub = rospy.Publisher(args.cmd_vel_topic, TwistStamped, queue_size=1)

#     def publish_stop():
#         stop = TwistStamped()
#         stop.header.stamp = rospy.Time.now()
#         cmd_pub.publish(stop)

#     rospy.on_shutdown(publish_stop)  # always zero the drone on shutdown/Ctrl+C

#     print(f"[INFO] Waiting up to {args.start_delay}s for first pose on {args.pose_topic} ...")
#     t0 = time.time()
#     while state["pos"] is None and time.time() - t0 < args.start_delay and not rospy.is_shutdown():
#         rospy.sleep(0.05)
#     if state["pos"] is None:
#         raise RuntimeError(f"No pose received on {args.pose_topic} within {args.start_delay}s -- "
#                             "check mavros is running and the topic name.")

#     cam_topics = args.color_topic + (f" + {args.depth_topic}" if use_depth else "")
#     print(f"[INFO] Waiting up to {args.start_delay}s for first camera frame on {cam_topics} ...")
#     t0 = time.time()
#     while not cam_ready() and time.time() - t0 < args.start_delay and not rospy.is_shutdown():
#         rospy.sleep(0.05)
#     if not cam_ready():
#         raise RuntimeError(f"No camera frame received on {cam_topics} within {args.start_delay}s -- "
#                             "check the camera driver is running and the topic names.")

#     # init obs history (same seeding pattern as sim: repeat the first real frame To times)
#     frame0 = grab_frame()
#     rgb_hist = deque(maxlen=To)
#     for _ in range(To):
#         rgb_hist.append((frame0[0].copy(), frame0[1].copy()) if use_depth else frame0.copy())

#     prev_actions_real = None
#     rate = rospy.Rate(1.0 / control_dt)
#     step = 0
#     print(f"[INFO] Starting control loop at {1.0 / control_dt:.1f} Hz "
#           f"(control_dt={control_dt}s, matches training). Ctrl+C to stop.")

#     try:
#         while not rospy.is_shutdown() and step < args.max_steps:
#             pos = state["pos"].copy()

#             obs_rgb_t = preprocess_obs_stack(rgb_hist, use_depth).to(device)
#             cond = {"obs_rgb": obs_rgb_t}

#             if vcfg["projection_mode"] == "sdpc" and pos_projector is not None:
#                 pos_projector.pos0 = pos[:3]
#             in_loop_projector = pos_projector if vcfg["projection_mode"] == "sdpc" else None

#             a_candidates_norm, _ = sample_action_candidates(
#                 diffusion=diffusion, cond=cond, horizon=horizon, action_dim=action_dim,
#                 num_candidates=vcfg["num_candidates"], projector=in_loop_projector,
#             )
#             a_candidates_real = dataset.action_normalizer.unnormalize(a_candidates_norm) * float(args.action_scale)

#             if vcfg["use_projection"] and vcfg["projection_mode"] in ("post", "sdpc"):
#                 a_candidates_proj_real, proj_costs = project_action_candidates_with_positions(
#                     projector=pos_projector, pos0=pos[:3], a_candidates_real=a_candidates_real, device=device,
#                 )
#             else:
#                 a_candidates_proj_real, proj_costs = a_candidates_real, None

#             if vcfg["selection"] == "minimum_projection_cost" and proj_costs is not None:
#                 which = int(np.argmin(proj_costs))
#             else:
#                 which = choose_trajectory(a_candidates_proj_real, strategy=vcfg["selection"],
#                                            prev_actions_real=prev_actions_real)

#             a0_real = a_candidates_proj_real[which, 0]  # delta-position over control_dt seconds
#             prev_actions_real = a_candidates_proj_real[which:which + 1]

#             # delta-position -> velocity setpoint (flight_test.py's mavros interface is
#             # velocity, not position, so convert here rather than assuming a position API)
#             vel = np.zeros(3, dtype=np.float32)
#             vel[:action_dim] = a0_real[:action_dim] / control_dt
#             vel = np.clip(vel, -args.max_speed, args.max_speed)

#             cmd = TwistStamped()
#             cmd.header.stamp = rospy.Time.now()
#             cmd.twist.linear.x = float(vel[0])
#             cmd.twist.linear.y = float(vel[1])
#             cmd.twist.linear.z = float(vel[2])
#             print(f"[step {step}] pos={pos} a0_real={a0_real} vel_cmd={vel}"
#                   f"{'  (DRY RUN)' if not args.live else ''}")
#             if args.live:
#                 cmd_pub.publish(cmd)

#             rgb_hist.append(grab_frame())
#             step += 1
#             rate.sleep()
#     finally:
#         publish_stop()
#         print("[INFO] Hardware run stopped, zero velocity published.")

def run_hardware_ros2(args, run_dir, device, drone_radius, active_halfspaces):
    """
    --mode ros2: live hardware deployment. Named "ros2" on the CLI for
    consistency with how the lab talks about this, but the actual transport
    mirrors flight_test.py exactly: ROS2 (rclpy) + mavros for pose, and (as of
    this version) color + aligned depth also come in over ROS2 sensor_msgs/Image
    topics (e.g. published by realsense-ros's ROS2 driver) instead of grabbing
    the RealSense SDK directly -- same one-ROS-version approach as pose, so
    this camera can live on a different node/machine than this process.

    Runs ONE model on ONE variant, no seed sweep, no Isaac Sim, no
    ground-truth obstacle logging -- those only make sense in --mode sim.
    """
    try:
        import rclpy
        from rclpy.qos import qos_profile_sensor_data
        from geometry_msgs.msg import PoseStamped, TwistStamped
        from sensor_msgs.msg import Image
    except ImportError as e:
        raise RuntimeError(
            "--mode ros2 requires rclpy, geometry_msgs, and sensor_msgs -- "
            "source the lab's ROS2 workspace first (same deps as flight_test.py)."
        ) from e

    # ------------------ Load trained model (single run_dir, no sweep) ------------------
    print(f"\n[INFO] Loading run dir: {run_dir}")
    diff_exp = utils.load_diffusion(run_dir, epoch="best", device=str(device))
    dataset = diff_exp.dataset
    diffusion = diff_exp.diffusion.to(device)
    diffusion.eval()

    use_depth = bool(getattr(diffusion.model, "use_depth", False))
    horizon = int(getattr(diffusion, "horizon", 16))
    action_dim = int(getattr(diffusion, "action_dim", 3))
    To = int(getattr(dataset, "n_obs_steps", 2))
    control_dt = float(getattr(dataset, "control_dt", getattr(dataset, "dt", 0.1)))
    print(f"[INFO] Hardware eval: To={To} H={horizon} action_dim={action_dim} "
          f"use_depth={use_depth} control_dt={control_dt}s")

    # ------------------ Optional static obstacle projector (ground-truth BOXES/
    # CYLINDERS from config.avoiding-crazyflie.py -- assumes the real corridor
    # matches that layout; no dynamic-obstacle or depth-obstacle support here) ---
    vcfg = variant_cfg(args.variant)
    if args.num_candidates is not None and vcfg["selection"] != "first":
        vcfg["num_candidates"] = args.num_candidates
    gradient = (vcfg["projection_mode"] == "gradient")
    pos_projector = None
    if vcfg["use_projection"]:
        proj_dt = 0.1
        pos_projector = build_position_projector(
            horizon_H=horizon, gradient=gradient, device=device,
            boxes=BOXES, cylinders=CYLINDERS, spheres=None,
            normalizer=None, tighten=vcfg["tighten"], dt=proj_dt,
            drone_radius=drone_radius, sphere_radius=0.0,
            active_halfspaces=active_halfspaces,
        )
        if vcfg["projection_mode"] == "sdpc":
            pos_projector.inloop_slsqp = True
            pos_projector.action_normalizer = dataset.action_normalizer
            pos_projector.pos0 = None
    print(f"[INFO] Variant: {args.variant} (projection={vcfg['use_projection']}, "
          f"mode={vcfg['projection_mode']})")

    if not args.live:
        print("[WARN] --live not set: DRY RUN. Velocity commands will be computed "
              "and printed but NOT published. Pass --live to actually fly.")

    # ------------------ ROS2 node (must exist before any Subscription/Publisher) ------
    rclpy.init()
    node = rclpy.create_node("diffusion_policy_hardware")

    # ------------------ ROS2 camera topics (color + aligned depth) ------------------
    # Frames are cached by the subscriber callbacks and grab_frame() just reads 
    # the latest one -- same pattern as how `state["pos"]` is read from pose_cb.
    cam_state = {"color": None, "depth": None}

    def color_cb(msg):
        if msg.encoding not in ("rgb8", "bgr8"):
            node.get_logger().warning(
                f"unsupported color encoding '{msg.encoding}', skipping frame", 
                throttle_duration_sec=5.0
            )
            return
        frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        if msg.encoding == "bgr8":
            frame = frame[:, :, ::-1]
        cam_state["color"] = cv2.resize(frame, (FPV_WIDTH, FPV_HEIGHT), interpolation=cv2.INTER_AREA)

    def depth_cb(msg):
        if msg.encoding == "16UC1":
            depth = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
            depth = depth.astype(np.float32) * 0.001  # RealSense: raw mm -> metres
        elif msg.encoding == "32FC1":
            depth = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
        else:
            node.get_logger().warning(
                f"unsupported depth encoding '{msg.encoding}', skipping frame", 
                throttle_duration_sec=5.0
            )
            return
        cam_state["depth"] = cv2.resize(depth, (FPV_WIDTH, FPV_HEIGHT), interpolation=cv2.INTER_NEAREST)

    node.create_subscription(Image, args.color_topic, color_cb, 1)
    if use_depth:
        node.create_subscription(Image, args.depth_topic, depth_cb, 1)

    def cam_ready():
        return cam_state["color"] is not None and (not use_depth or cam_state["depth"] is not None)

    def grab_frame():
        if not use_depth:
            return cam_state["color"]
        return cam_state["color"], cam_state["depth"]

    # ------------------ pose in, velocity setpoint out (mirrors flight_test.py's
    # /mavros/local_position/pose -> /mavros/setpoint_velocity/cmd_vel) ------------
    state = {"pos": None}
    state["pos"] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    def pose_cb(msg):
        p = msg.pose.position
        print('pos:',p)
        state["pos"] = np.array([p.x, p.y, p.z], dtype=np.float32)

    node.create_subscription(PoseStamped, args.pose_topic, pose_cb, qos_profile_sensor_data)
    cmd_pub = node.create_publisher(PoseStamped, args.cmd_vel_topic, 1)

    def publish_stop():
        if rclpy.ok():
            stop = TwistStamped()
            stop.header.stamp = node.get_clock().now().to_msg()
            cmd_pub.publish(stop)

    print(f"[INFO] Waiting up to {args.start_delay}s for first pose on {args.pose_topic} ...")
    t0 = time.time()
    # In ROS2, we must spin to receive messages
    while state["pos"] is None and time.time() - t0 < args.start_delay and rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.05)
        
    if state["pos"] is None:
        raise RuntimeError(f"No pose received on {args.pose_topic} within {args.start_delay}s -- "
                            "check mavros is running and the topic name.")

    cam_topics = args.color_topic + (f" + {args.depth_topic}" if use_depth else "")
    print(f"[INFO] Waiting up to {args.start_delay}s for first camera frame on {cam_topics} ...")
    t0 = time.time()
    while not cam_ready() and time.time() - t0 < args.start_delay and rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.05)
        
    if not cam_ready():
        raise RuntimeError(f"No camera frame received on {cam_topics} within {args.start_delay}s -- "
                            "check the camera driver is running and the topic names.")

    # init obs history (same seeding pattern as sim: repeat the first real frame To times)
    frame0 = grab_frame()
    rgb_hist = deque(maxlen=To)
    for _ in range(To):
        rgb_hist.append((frame0[0].copy(), frame0[1].copy()) if use_depth else frame0.copy())

    prev_actions_real = None
    rate = node.create_rate(30)
    step = 0
    print(f"[INFO] Starting control loop at {1.0 / control_dt:.1f} Hz "
          f"(control_dt={control_dt}s, matches training). Ctrl+C to stop.")


    while rclpy.ok() and step < args.max_steps:
        # Spin once to process latest callbacks before acting
        rclpy.spin_once(node, timeout_sec=0)
        
        pos = state["pos"].copy()

        obs_rgb_t = preprocess_obs_stack(rgb_hist, use_depth).to(device)
        cond = {"obs_rgb": obs_rgb_t}

        if vcfg["projection_mode"] == "sdpc" and pos_projector is not None:
            pos_projector.pos0 = pos[:3]
        in_loop_projector = pos_projector if vcfg["projection_mode"] == "sdpc" else None

        a_candidates_norm, _ = sample_action_candidates(
            diffusion=diffusion, cond=cond, horizon=horizon, action_dim=action_dim,
            num_candidates=vcfg["num_candidates"], projector=in_loop_projector,
        )
        a_candidates_real = dataset.action_normalizer.unnormalize(a_candidates_norm) * float(args.action_scale)

        if vcfg["use_projection"] and vcfg["projection_mode"] in ("post", "sdpc"):
            a_candidates_proj_real, proj_costs = project_action_candidates_with_positions(
                projector=pos_projector, pos0=pos[:3], a_candidates_real=a_candidates_real, device=device,
            )
        else:
            a_candidates_proj_real, proj_costs = a_candidates_real, None

        if vcfg["selection"] == "minimum_projection_cost" and proj_costs is not None:
            which = int(np.argmin(proj_costs))
        else:
            which = choose_trajectory(a_candidates_proj_real, strategy=vcfg["selection"],
                                        prev_actions_real=prev_actions_real)

        a0_real = a_candidates_proj_real[which, 0]  # delta-position over control_dt seconds
        prev_actions_real = a_candidates_proj_real[which:which + 1]

        # delta-position -> velocity setpoint
        vel = np.zeros(3, dtype=np.float32)

        inc_action = a0_real[:action_dim] * args.action_scale
        inc_action = np.clip(inc_action, [-0.5,-0.5,-0.1], [0.5,0.5,0.1])

        vel[:action_dim] = pos  +  inc_action
        # vel = np.clip(vel, -args.max_speed, args.max_speed)

        cmd_pos = PoseStamped()
        cmd_pos.header.stamp = node.get_clock().now().to_msg()
        cmd_pos.pose.position.x = float(vel[0])
        cmd_pos.pose.position.y = float(vel[1])
        cmd_pos.pose.position.z = float(vel[2])
        
        print(f"[step {step}] pos={pos} a0_real={a0_real} vel_cmd={vel}"
                f"{'  (DRY RUN)' if not args.live else ''}")
                
        if args.live:
            cmd_pub.publish(cmd_pos)

        rgb_hist.append(grab_frame())
        step += 1
    try:
        rate.sleep()    
    except KeyboardInterrupt:
        pass
    finally:
        publish_stop()
        print("[INFO] Hardware run stopped, zero velocity published.")
        node.destroy_node()
        rclpy.try_shutdown()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--max_steps", type=int, default=70000)
    parser.add_argument("--action_scale", type=float, default=5.0)
    parser.add_argument("--num_candidates", type=int, default=4,
                        help="Override K (number of sampled candidates) for variants that select among multiple candidates")
    parser.add_argument("--use_halfspaces", action="store_true", default=False,
                        help="Enforce corridor halfspace constraints (from CORRIDOR_HALFSPACES "
                             "in config) in the projector.")
    parser.add_argument("--variant", type=str, choices=projection_variants, default="diffuser",
                        help="Single projection variant to fly (no sweep on real hardware).")
    parser.add_argument("--pose_topic", type=str, default="/mavros/local_position/pose",
                        help="PoseStamped topic for drone world position.")
    parser.add_argument("--cmd_vel_topic", type=str, default="/mpc/set_pose",
                        help="TwistStamped topic the velocity setpoint is published to.")
    parser.add_argument("--max_speed", type=float, default=0.5,
                        help="Per-axis velocity clamp in m/s, applied to every published "
                             "command independent of --action_scale.")
    parser.add_argument("--color_topic", type=str, default="camera/camera/color/image_raw",
                        help="sensor_msgs/Image color topic (ROS1), e.g. published by "
                             "realsense-ros's ROS1 driver. Replaces the old direct "
                             "pyrealsense2 SDK grab so the camera can run as its own node.")
    parser.add_argument("--depth_topic", type=str, default="/camera/aligned_depth_to_color/image_raw",
                        help="sensor_msgs/Image depth topic (ROS1, aligned to color). Only "
                             "subscribed when the loaded model has use_depth=True.")
    parser.add_argument("--start_delay", type=float, default=5.0,
                        help="Seconds to wait for the first pose/camera message before giving up.")
    parser.add_argument("--live", action="store_true", default=False,
                        help="Actually publish velocity commands. Without this flag, runs a "
                             "dry run that computes and prints commands but never publishes "
                             "them -- use this first on real hardware.")
    args, _unknown = parser.parse_known_args()

    device         = torch.device("cuda:0")
    drone_radius   = 0.08

    # ── active halfspaces: from config only when --use_halfspaces is set ─────────
    active_halfspaces = corridor_halfspaces if args.use_halfspaces else []
    if args.use_halfspaces:
        print(f"[INFO] Halfspace constraints enabled ({len(active_halfspaces)} halfspaces)")

    # run_hardware_ros1(args, args.run_dir, device, drone_radius, active_halfspaces)
    run_hardware_ros2(args, args.run_dir, device, drone_radius, active_halfspaces)



if __name__ == "__main__":
    main()