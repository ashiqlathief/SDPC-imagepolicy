import argparse
import importlib
import os
import sys
import cv2
from collections import deque
from pathlib import Path
import time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import re 
from matplotlib.patches import Rectangle, Circle, Polygon

import diffuser.utils as utils
from diffuser.sampling.projection import Projector
from diffuser.sampling.policies import temporal_consistency_distances, sum_projection_costs
from metrics_logger import MetricsLogger   
cfg = importlib.import_module("config.avoiding-crazyflie")
BOXES = cfg.BOXES
CYLINDERS = cfg.CYLINDERS
SPHERES = getattr(cfg, 'SPHERES', [])
SPHERE_RADIUS = getattr(cfg, 'SPHERE_RADIUS', 0.10)
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
#   'diffuser',
#   'gradient',
#   'gradient-tightened',
#   'post_processing',
#   'post_processing-tightened',
  'sdpc-c-tightened-dt0p25',
  'sdpc-c-tightened-dt0p5',
  'sdpc-c-tightened-dt2p0',
  'sdpc-c-tightened-dt4p0',
]

# Per-axis speed bound used by the dt-sweep variants' dynamics constraint —
# taken from the sim's real position->velocity PID outer loop, which clamps
# commanded velocity to this exact range regardless of what the projector allows
# (see isaac/scripts/crazyflie_env.py: `vel_des = torch.clamp(vel_des, -0.5, 0.5)`).
RATE_LIMIT_V_MAX = 0.5

def variant_cfg(name: str):
    cfg = dict(
        num_candidates=1,
        selection="first",
        use_projection=False,
        projection_mode="none",    # "none" | "post" | "gradient"
        tighten=0.0,
        dt_ratio=None,   # t_hat_s / t_s (see dt-sweep branch below) — None = use default proj_dt
        rate_limit=False,
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

    # dt sweeps (Table 2 in Römer et al., "Diffusion Predictive Control with
    # Constraints" — model-mismatch ablation between the sampling time t_hat_s
    # assumed by the projection's Euler dynamics and the true system t_s).
    elif name.startswith("sdpc-c-tightened-dt"):
        # parse ratio from string, e.g. dt0p25 -> 0.25 == t_hat_s / t_s (NOT an
        # absolute dt in seconds — the real proj_dt = ratio * eval_dt, computed
        # in main() where eval_dt is known).
        dt_str = name.split("dt")[-1].replace("p", ".")
        # rate_limit=True is what makes dt actually matter here: it adds a
        # per-axis velocity dim + 'deriv' dynamics constraint (x[t+1]=x[t]+dt*v[t])
        # bounded by RATE_LIMIT_V_MAX, so dt directly scales the max per-step
        # displacement the projector allows. Without it, dt only affected the
        # (unused-here) dynamic-obstacle lookahead and was otherwise dead.
        cfg.update(num_candidates=8, selection="minimum_projection_cost", use_projection=True, projection_mode="sdpc", tighten=0.05, dt_ratio=float(dt_str), rate_limit=True)

    return cfg

def plot_constraint_overlay(ax, boxes, cylinders, tighten=0.03, drone_radius=0.0,
                            x_bounds=(-0.5, 4.5), y_bounds=(-1.0, 1.0),
                            z_bounds=(0.0, 1.0)):
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

    # # solid boundary lines
    # for yval in [ymin, ymax]:
    #     ax.axhline(yval, color="royalblue", linewidth=1.5, linestyle="-", alpha=0.6, zorder=4)
    # for xval in [xmin, xmax]:
    #     ax.axvline(xval, color="royalblue", linewidth=1.5, linestyle="-", alpha=0.6, zorder=4)

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

def choose_trajectory(actions_real, infos, strategy="first", prev_actions_real=None):
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

    if strategy == "minimum_projection_cost":
        if isinstance(infos, dict) and "projection_costs" in infos:
            costs_total = sum_projection_costs(infos["projection_costs"], K)
            return int(np.argmin(costs_total))

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
    K, H, D = a_candidates_real.shape
    traj_xyz = np.zeros((K, H + 1, 3), dtype=np.float32)
    traj_xyz[:, 0, :] = pos0_xyz[:3][None, :]
    traj_xyz[:, 1:, :] = pos0_xyz[:3][None, None, :] + np.cumsum(a_candidates_real[:, :, :3], axis=1)
    return traj_xyz

def build_obstacle_constraint_list(boxes, cylinders, spheres=None, x_bounds=None,
                                    y_bounds=None, z_bounds=None,
                                    corridor_halfspaces=None, z_halfspaces=None, tighten=0.0,
                                    cyl_extra_radius=0.0, drone_radius=0.0,
                                    sphere_radius=0.0, dynamic_cylinder_predictions=None,
                                    rate_limit=False, rate_v_max=RATE_LIMIT_V_MAX):
    """
    dynamic_cylinder_predictions: optional {cylinder_index: [(x,y), ...]} — one
    predicted (x,y) per horizon step (in order), from env.predict_cylinder_positions().
    Cylinders with an entry here get a per-timestep obstacle constraint (the
    projector avoids where the obstacle WILL be at each planned step); all others
    keep the usual single current-position constraint applied across the horizon.

    rate_limit: appends velocity dims [vx, vy, vz] (indices 3,4,5) to the state,
    tied to positions via 'deriv' dynamics constraints (x[t+1] = x[t] + dt*v[t],
    dt supplied separately via the Projector's own dt) and bounded by rate_v_max.
    This is what makes the projector's dt setting actually change the feasible
    set — caller must also build the Projector with transition_dim=6 to match.
    """
    constraint_list = []
    state_dim = 6 if rate_limit else 3

    # ---------------- bounds ----------------
    lb = np.full(state_dim, -np.inf, dtype=np.float32)
    ub = np.full(state_dim,  np.inf, dtype=np.float32)

    if x_bounds is not None:
        lb[0], ub[0] = float(x_bounds[0]), float(x_bounds[1])
    if y_bounds is not None:
        lb[1], ub[1] = float(y_bounds[0]), float(y_bounds[1])
    if z_bounds is not None:
        lb[2], ub[2] = float(z_bounds[0]), float(z_bounds[1])

    if rate_limit:
        # per-axis speed bound on [vx, vy, vz] — see RATE_LIMIT_V_MAX
        lb[3:6] = -float(rate_v_max)
        ub[3:6] =  float(rate_v_max)

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

    # for (x, y, z) in boxes:
    #     center = [float(x), float(y), float(z)]
    #     constraint_list.append(("sphere_outside", [0, 1, 2], center, box_sphere_r))

    # for (x, y, z) in cylinders:
    #     center = [float(x), float(y), float(z)]
    #     constraint_list.append(("sphere_outside", [0, 1, 2], center, cyl_sphere_r))

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

    # ---------------- dynamics (rate limit) ----------------
    # Ties x[t+1] = x[t] + dt*v[t] per axis (DynamicConstraints' existing 'deriv'
    # mechanism, same one used for pointmaze/antmaze/avoiding-d3il — see
    # diffuser/utils/constraints_helpers.py:formulate_dynamics_constraints).
    # Combined with the [vx,vy,vz] bounds above, this makes dt scale the maximum
    # per-step displacement the projector will allow.
    if rate_limit:
        constraint_list.append(("deriv", [0, 3]))  # x <-> vx
        constraint_list.append(("deriv", [1, 4]))  # y <-> vy
        constraint_list.append(("deriv", [2, 5]))  # z <-> vz

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
    K, H, _ = a_candidates_proj_real.shape
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
                             active_halfspaces=None, dynamic_cylinder_predictions=None,
                             rate_limit=False, rate_v_max=RATE_LIMIT_V_MAX):
    # We project POSITIONS, so we need horizon = H+1
    Hp1 = horizon_H + 1
    x_bounds = (-0.5, 4.5)
    y_bounds = (-0.95, 0.95)
    z_bounds = (0.0, 1.0)
    transition_dim = 6 if rate_limit else 3   # rate_limit appends [vx,vy,vz]

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
        rate_limit=rate_limit,
        rate_v_max=rate_v_max,
    )

    projector = Projector(
        horizon=Hp1,
        transition_dim=transition_dim,   # x,y,z positions (+ vx,vy,vz if rate_limit)
        action_dim=0,
        goal_dim=0,
        constraint_list=constraint_list,
        normalizer=normalizer,          # start without normalizer for debugging
        gradient=gradient,
        gradient_weights=[1, 0.5, 2],
        # dt drives the rate_limit 'deriv' constraint below — only zero it out
        # for the (non-rate-limit) generic "no Euler dynamics enforced" case.
        dt=dt if (use_dynamics or rate_limit) else 0.0,
        variant="states",
        skip_initial_state=True,
        diffusion_timestep_threshold=0.8,
        device=str(device),
        solver="scipy",           # your file uses scipy SLSQP path
        parallelize=True,         # candidates solve independently — run them concurrently
    )

    if rate_limit:
        # Close the first-step gap: skip_initial_state makes SafetyConstraints
        # strip block 0's row from every lb/ub constraint (block 0 is normally
        # just the pinned current state, so its own bounds don't matter) — but
        # here block 0's [vx,vy,vz] is the velocity that drives the very first
        # transition (pos0 -> first planned step) via the 'deriv' equality above,
        # and that first transition is exactly the action actually applied to the
        # drone each control step. Without this, block 0's velocity is bounded
        # only by the solver's generic +/-5 box instead of rate_v_max. Append the
        # missing rows directly (bypassing SafetyConstraints' skip-slicing) since
        # this only needs to touch block 0's velocity dims, not the position dims
        # every other variant relies on skip_initial_state to leave unconstrained.
        n_cols = transition_dim * projector.horizon
        extra_rows, extra_rhs = [], []
        for dim in (3, 4, 5):
            row = torch.zeros(n_cols, device=device)
            row[dim] = 1.0
            extra_rows.append(row.clone())
            extra_rhs.append(rate_v_max)
            row = torch.zeros(n_cols, device=device)
            row[dim] = -1.0
            extra_rows.append(row)
            extra_rhs.append(rate_v_max)
        projector.C = torch.cat([projector.C, torch.stack(extra_rows, dim=0)], dim=0)
        projector.d = torch.cat([projector.d, torch.tensor(extra_rhs, device=device)], dim=0)
        projector.add_numpy_constraints()   # refresh C_np/d_np used by project()

    return projector

def project_action_candidates_with_positions(projector, pos0, a_candidates_real, device, dt=None):
    """
    pos0: (3,) current position
    a_candidates_real: (K,H,3) delta-pos
    dt: pass the projector's dt when it was built with rate_limit=True (transition_dim=6,
        i.e. [x,y,z,vx,vy,vz]) — appends velocity dims derived from the deltas so the
        state matches what the 'deriv' dynamics constraint expects. None for the
        ordinary position-only (transition_dim=3) projector.
    Returns:
      a_proj_real: (K,H,3)
      proj_costs: (K,)
    """
    K, H, _ = a_candidates_real.shape

    # integrate deltas -> positions (K,H+1,3), all K candidates share pos0
    pos_traj = np.zeros((K, H + 1, 3), dtype=np.float32)
    pos_traj[:, 0] = pos0.astype(np.float32)
    pos_traj[:, 1:] = pos_traj[:, :1] + np.cumsum(a_candidates_real, axis=1)

    if dt is not None:
        vel_traj = np.zeros((K, H + 1, 3), dtype=np.float32)
        vel_traj[:, :H] = a_candidates_real / dt
        vel_traj[:, H] = vel_traj[:, H - 1]
        state_traj = np.concatenate([pos_traj, vel_traj], axis=-1)  # (K,H+1,6)
    else:
        state_traj = pos_traj

    state_t = torch.tensor(state_traj, dtype=torch.float32, device=device)  # (K,H+1,transition_dim)

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7,8])
    parser.add_argument("--max_steps", type=int, default=700)
    parser.add_argument("--action_scale", type=float, default=5.0)
    parser.add_argument("--episodes", type=int, default=1)   # how many episodes to roll
    parser.add_argument("--dynamic_obstacles", type=str, nargs="*", default=None, metavar="IDX:AXIS",
                        help="Enable sinusoidal cylinder movement. Omit this flag entirely to disable. "
                             "Pass with no values to move ALL cylinders laterally (axis 'y'). "
                             "Or give 'idx:axis' tokens (axis is 'x', 'y', or 'xy'; ':axis' optional, "
                             "defaults to 'y'), e.g. --dynamic_obstacles 0:y 2:x 4:xy")
    parser.add_argument("--floating_spheres", action="store_true", default=False,
                        help="Enable floating 3D sphere obstacles (planet mode). "
                             "Spheres move with independent x/y/z sinusoidal motion.")
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--num_candidates", type=int, default=4,
                        help="Override K (number of sampled candidates) for variants that select among multiple candidates")
    parser.add_argument("--use_halfspaces", action="store_true", default=False,
                        help="Enforce corridor halfspace constraints (from CORRIDOR_HALFSPACES "
                             "in config) in the projector and show them in XY plots.")
    parser.add_argument("--record_video", action="store_true", default=False,
                        help="Save an .mp4 of each recorded variant's episode to "
                             "<run_dir>/videos/eval_<variant>_<camera>.mp4")
    parser.add_argument("--camera", type=str, nargs="+", default=["spectator"],
                        choices=["spectator", "chase", "fpv"],
                        help="Which camera(s) to record from when --record_video is set "
                             "-- pass more than one (e.g. --camera spectator chase) to record "
                             "them simultaneously from the same episode, one .mp4 per camera: "
                             "'spectator' is the fixed environment-mounted outside view "
                             "(SPECTATOR_CAMERA_CFG), 'chase' is a third-person view mounted "
                             "on the drone (CHASE_CAMERA_CFG), 'fpv' is the same onboard "
                             "camera the policy sees (get_rgb()/FPV_CAMERA_CFG).")
    parser.add_argument("--record_variants", type=str, nargs="*", default=None,
                        help="Subset of variant names (from projection_variants) to record "
                             "when --record_video is set. Omit to record every variant run "
                             "this invocation.")
    parser.add_argument("--video_fps", type=int, default=20,
                        help="Playback fps of saved videos (independent of sim dt).")
    args, _unknown = parser.parse_known_args()

    device         = torch.device("cuda:0")
    drone_radius   = 0.08
    obs_amplitude  = 0.35
    obs_frequency  = 0.45
    sphere_amplitude = 0.20
    sphere_frequency = 0.20

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
                    parser.error(f"--dynamic_obstacles: invalid axis '{axis}' in '{tok}' (use x, y, or xy)")
                indices.append(int(idx_str))
                axes.append(axis)
            args.dynamic_cyl_indices = indices
            args.obs_axes = axes

    # ── resolve which seed run_dir(s) to evaluate ─────────────────────────
    # Default (no --seeds): behave exactly as before, args.run_dir IS the
    # seed directory. With --seeds, args.run_dir is the experiment directory
    # one level up, and we evaluate <run_dir>/<seed> for each seed in turn,
    # reusing a single Isaac Sim instance across all of them.
    if args.seeds:
        run_dirs = [os.path.join(args.run_dir, str(s)) for s in args.seeds]
    else:
        run_dirs = [args.run_dir]

    env = None
    shared_use_depth = None
    shared_eval_dt = None

    for run_dir in run_dirs:
        # ------------------ Load trained experiment ------------------
        print(f"\n[INFO] Loading run dir: {run_dir}")

        seedmodel = int(Path(run_dir).name)   # gives 9
        diff_exp = utils.load_diffusion(run_dir,epoch="best",device=str(device),)
        dataset = diff_exp.dataset
        diffusion = diff_exp.diffusion.to(device)
        diffusion.eval()

        # RGB vs RGBD is determined by the trained checkpoint, not a CLI flag here —
        # the dataset and model agree at train time (see traintransformer.py's
        # consistency assert), so trust the model's own in_chans/use_depth.
        use_depth = bool(getattr(diffusion.model, "use_depth", False))
        assert bool(getattr(dataset, "use_depth", False)) == use_depth, (
            f"Loaded checkpoint's model.use_depth={use_depth} but dataset.use_depth="
            f"{getattr(dataset, 'use_depth', False)} -- mismatched run dir?"
        )
        print(f"[INFO] Running evaluation in {'RGB-D' if use_depth else 'RGB'} mode "
              f"(use_depth={use_depth}, in_chans={getattr(diffusion.model, 'in_chans', 3)})")

        # ── derive eval dt from the trained dataset ───────────────────────────
        # a0_real is a displacement over `dataset.control_dt` seconds of sim time
        # (control_dt = stride * collection_dt). The eval env's sim dt must match
        # this or every action gets applied over the wrong amount of physical
        # time, making the drone fly too fast/slow relative to what the model
        # intends. Auto-derive it here instead of hardcoding it in env config.
        expected_dt = getattr(dataset, "control_dt", getattr(dataset, "dt", None))
        if args.dt is not None:
            if expected_dt is not None and not np.isclose(args.dt, expected_dt, rtol=1e-3):
                print(f"[WARN] --dt={args.dt} explicitly overrides dataset control_dt={expected_dt} "
                      "— running at a deliberately mismatched dt.")
            eval_dt = args.dt
        elif expected_dt is not None:
            eval_dt = expected_dt
        else:
            eval_dt = 0.005  # old run without stored stride/dt metadata (default sim dt)
        print(f"[INFO] Using eval dt={eval_dt} (dataset control_dt={expected_dt})")

        # ------------------ Create env (once — reused across seeds) ------------------
        # The sim instance is expensive to start, so with --seeds we build it once
        # for the first seed and reuse it for the rest. This assumes every seed in
        # the batch shares the same architecture/use_depth/dt, which holds as long
        # as they're all seeds of the same experiment folder.
        if env is None:
            # Propagate the detected mode to the FPV camera config before crazyflie_env_cfg
            # (which builds CrazyflieSceneCfg.FPV_CAMERA_CFG's data_types from cfg.USE_DEPTH
            # at import time) is imported, so the simulated camera actually has a depth
            # channel whenever the loaded checkpoint needs one.
            cfg.USE_DEPTH = use_depth
            from isaac.scripts.crazyflie_env import Crazyflie, CrazyflieEnvCfg

            env_cfg = CrazyflieEnvCfg(
                num_envs=1,
                device=str(device),
                dynamic_obstacles=args.dynamic_obstacles_enabled,
                obs_amplitude=obs_amplitude,
                obs_frequency=obs_frequency,
                dynamic_cyl_indices=args.dynamic_cyl_indices,
                obs_axes=args.obs_axes,
                floating_spheres=args.floating_spheres,
                sphere_amplitude=sphere_amplitude,
                sphere_frequency=sphere_frequency,
                dt=eval_dt,
                drone_radius=drone_radius,
            )
            env = Crazyflie(env_cfg)
            shared_use_depth = use_depth
            shared_eval_dt = eval_dt
        else:
            if use_depth != shared_use_depth:
                raise RuntimeError(
                    f"Seed {seedmodel}'s checkpoint use_depth={use_depth} doesn't match the "
                    f"already-running sim (use_depth={shared_use_depth}, set from an earlier seed "
                    "in this --seeds batch). Mixed RGB/RGB-D seeds can't share one sim instance -- "
                    "run this seed separately."
                )
            if not np.isclose(eval_dt, shared_eval_dt, rtol=1e-3):
                print(f"[WARN] Seed {seedmodel}'s derived eval_dt={eval_dt} differs from the "
                      f"running sim's dt={shared_eval_dt} (fixed when the sim was created for an "
                      "earlier seed) -- continuing at the sim's dt, not this seed's.")

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
        # path = os.path.join(run_dir, "results")
        # os.makedirs(path, exist_ok=True)
        # os.makedirs("plots", exist_ok=True)
        logger = MetricsLogger(                                             
            save_dir=os.path.join(run_dir, "results"),                               
            encoder_type=encoder_type,
            latent_dim=latent_dim,
            horizon=horizon,
            n_diffusion_steps=20,
            corridor_halfspaces  = corridor_halfspaces,
            boxes                = BOXES,
            cylinders            = CYLINDERS,
        )

        # ------------------ Video recording setup ------------------
        camera_fns = {
            "spectator": env.get_spectator_rgb,
            "chase": env.get_chase_rgb,
            "fpv": env.get_rgb,
        }
        video_dir = os.path.join(run_dir, "videos")
        if args.record_video:
            os.makedirs(video_dir, exist_ok=True)

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

            if vcfg["use_projection"]:
                # dt_ratio (dt-sweep variants only) is t_hat_s/t_s from the paper's
                # Table 2 — scale by the run's real control dt, don't use it as an
                # absolute seconds value.
                if vcfg.get("dt_ratio") is not None:
                    proj_dt = vcfg["dt_ratio"] * eval_dt
                else:
                    proj_dt = 0.1
                # All cylinders (static and dynamic) use the same base radius.
                # Dynamic ones get their actual positions via --dynamic_obstacles (see below).
                _init_spheres = SPHERES if args.floating_spheres else None
                _proj_cyls    = CYLINDERS
                pos_projector = build_position_projector(
                    horizon_H=horizon, gradient=gradient, device=device,
                    boxes=BOXES, cylinders=_proj_cyls,
                    spheres=_init_spheres,
                    normalizer=None, tighten=vcfg["tighten"], dt=proj_dt,
                    use_dynamics=vcfg.get("use_dynamics", True),
                    drone_radius=drone_radius,
                    sphere_radius=SPHERE_RADIUS if args.floating_spheres else 0.0,
                    active_halfspaces=active_halfspaces,
                    rate_limit=vcfg.get("rate_limit", False),
                )
                if vcfg["projection_mode"] == "sdpc":
                    # Enable proper in-loop SLSQP with obstacle constraints.
                    # pos0 is set per control step below (current drone position).
                    pos_projector.inloop_slsqp = True
                    pos_projector.action_normalizer = dataset.action_normalizer
                    pos_projector.pos0 = None  # filled in per step

            print(f"\n[INFO] ===== Episode {ep+1}/{args.episodes}_{variant_name} =====")
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

            # ── per-episode video writers, one per requested --camera, recorded
            # simultaneously from the same rollout (only for variants the caller asked for) ──
            record_this_episode = args.record_video and (
                args.record_variants is None or variant_name in args.record_variants
            )
            video_writers = {}
            video_paths = {}
            if record_this_episode:
                for cam_name in args.camera:
                    get_video_frame = camera_fns[cam_name]
                    frame0 = get_video_frame()
                    vh, vw = frame0.shape[:2]
                    video_path = os.path.join(video_dir, f"eval_{variant_name}_{cam_name}.mp4")
                    video_writers[cam_name] = cv2.VideoWriter(
                        video_path, cv2.VideoWriter_fourcc(*"mp4v"), args.video_fps, (vw, vh)
                    )
                    video_paths[cam_name] = video_path
                    print(f"[INFO] Recording '{cam_name}' camera video ({vw}x{vh} @ {args.video_fps}fps) -> {video_path}")

            def write_video_frame():
                for cam_name, writer in video_writers.items():
                    writer.write(cv2.cvtColor(camera_fns[cam_name](), cv2.COLOR_RGB2BGR))

            write_video_frame()


            # init rgb(d) history
            rgb0 = get_obs_frame_from_env(env, use_depth)
            rgb_hist = deque(maxlen=To)
            for _ in range(To):
                rgb_hist.append((rgb0[0].copy(), rgb0[1].copy()) if use_depth else rgb0.copy())

            traj_xyz = []
            actions_taken = []
            prev_actions_real = None
            cand_snapshots = []

            pos_init = env._pos_world().detach().cpu().numpy()[0]
            traj_xyz.append(pos_init.copy())

            for step in range(args.max_steps):
                # current position (for logging/command conversion)
                pos = env._pos_world().detach().cpu().numpy()[0]  # [x,y,z]

                # ── diffusion path ────────────────────────────────────────────
                # build condition
                obs_rgb_t = preprocess_obs_stack(rgb_hist, use_depth).to(device)  # (1,To,3or4,H,W)
                cond = {"obs_rgb": obs_rgb_t}
                # a_chunk = sample_action_chunk(diffusion=diffusion,cond=cond,horizon=horizon,action_dim=action_dim,device=device,)
                # a_chunk_real = dataset.action_normalizer.unnormalize(a_chunk)
                # a0_real = a_chunk_real[0]* float(args.action_scale)

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

                a_candidates_norm, infos = sample_action_candidates(diffusion=diffusion,cond=cond,horizon=horizon,action_dim=action_dim,
                                                                    num_candidates=vcfg["num_candidates"],projector=in_loop_projector)   # (K, H, D)

                # Unnormalize all candidates to real delta-pos, then apply action_scale
                # here -- before projection/selection -- so the projector validates
                # (and the selection scores) the trajectory that will actually be
                # executed, not an unscaled stand-in for it.
                a_candidates_real = dataset.action_normalizer.unnormalize(a_candidates_norm) * float(args.action_scale)   # (K, H, D)

                # ── plain (unguided) model output ──────────────────────────────
                # For "sdpc" mode, in_loop_projector steers the reverse-diffusion
                # sampling itself, so a_candidates_real above is NOT the plain
                # model output. Resample with the identical RNG state but
                # projector=None to get the true "before guidance" candidates for
                # the same underlying noise draw (doubles diffusion sampling cost
                # for sdpc variants). Every other mode already sampled unguided.
                if vcfg["projection_mode"] == "sdpc":
                    torch.set_rng_state(_rng_cpu)
                    if _rng_cuda is not None:
                        torch.cuda.set_rng_state_all(_rng_cuda)
                    a_plain_norm, _ = sample_action_candidates(diffusion=diffusion, cond=cond, horizon=horizon, action_dim=action_dim,
                                                               num_candidates=vcfg["num_candidates"], projector=None)
                    a_plain_real = dataset.action_normalizer.unnormalize(a_plain_norm) * float(args.action_scale)
                else:
                    a_plain_real = a_candidates_real
                # y_finals = a_candidates_real[:, :, 1].cumsum(axis=1)[:, -1]
                # print(f"Candidate final y positions: {y_finals}")
                # a0_real = a_candidates_real[0,0]* float(args.action_scale)

                # # apply projection and get costs
                # a_candidates_proj_real, proj_costs = project_action_candidates_with_positions(
                #     projector=pos_projector,
                #     pos0=pos[:3],
                #     a_candidates_real=a_candidates_real,
                #     device=device,
                # )
                # which = int(np.argmin(proj_costs))

                # ── per-step projector update ──────────
                # Gated on --dynamic_obstacles: with no dynamic obstacles,
                # get_cylinder_positions() returns the same rest positions every step, so
                # rebuilding the projector (full constraint-matrix reconstruction) would be
                # a no-op — skip it and keep reusing the projector built once above.
                _need_rebuild = (args.dynamic_obstacles_enabled or args.floating_spheres)
                if _need_rebuild and vcfg["use_projection"]:
                    # get_cylinder_positions() returns current positions for ALL cylinders
                    # (static ones at rest, dynamic ones at actual current position)
                    cyl_now = env.get_cylinder_positions()
                    sph_now = env.get_sphere_positions() if args.floating_spheres else None
                    # Predict each dynamic cylinder's exact future (x,y) at every planned
                    # horizon step (t = now + k*proj_dt), using the sim's own closed-form
                    # sinusoid — so the projector avoids where the obstacle WILL be, not
                    # just where it is right now. This is what makes `proj_dt` matter.
                    dyn_cyl_preds = (
                        env.predict_cylinder_positions([k * proj_dt for k in range(1, horizon + 1)])
                        if args.dynamic_obstacles_enabled else None
                    )
                    pos_projector = build_position_projector(
                        horizon_H=horizon, gradient=gradient, device=device,
                        boxes=BOXES, cylinders=cyl_now,
                        spheres=sph_now,
                        normalizer=None, tighten=vcfg["tighten"], dt=proj_dt,
                        use_dynamics=vcfg.get("use_dynamics", True),
                        obs_amplitude=0.0,  # exact positions — no extra radius needed
                        drone_radius=drone_radius,
                        sphere_radius=SPHERE_RADIUS if args.floating_spheres else 0.0,
                        active_halfspaces=active_halfspaces,
                        dynamic_cylinder_predictions=dyn_cyl_preds,
                        rate_limit=vcfg.get("rate_limit", False),
                    )
                    if vcfg["projection_mode"] == "sdpc":
                        pos_projector.inloop_slsqp = True
                        pos_projector.action_normalizer = dataset.action_normalizer
                        pos_projector.pos0 = pos[:3]

                # ── projection + selection  ---------
                if vcfg["use_projection"] and vcfg["projection_mode"] in ("post", "sdpc"):
                    a_candidates_proj_real, proj_costs = project_action_candidates_with_positions(
                        projector=pos_projector, pos0=pos[:3],
                        a_candidates_real=a_candidates_real, device=device,
                        dt=proj_dt if vcfg.get("rate_limit", False) else None,
                    )
                else:
                    a_candidates_proj_real = a_candidates_real
                    proj_costs = None

                if vcfg["selection"] == "minimum_projection_cost" and proj_costs is not None:
                    which = int(np.argmin(proj_costs))
                else:
                    which = choose_trajectory(
                        a_candidates_proj_real, infos=infos,
                        strategy=vcfg["selection"],
                        prev_actions_real=prev_actions_real,
                    )

                # ── bound verification (set to True to enable, False to silence) ──
                _VERIFY = False
                if _VERIFY and vcfg["use_projection"]:
                    verify_projection(
                        pos0=pos[:3],
                        a_candidates_proj_real=a_candidates_proj_real,
                        which=which,
                        x_bounds=(-0.5, 4.5), y_bounds=(-0.95, 0.95), z_bounds=(0.0, 1.0),
                        boxes=BOXES, cylinders=CYLINDERS,
                        tighten=vcfg["tighten"],
                        step=step,
                    )

                a0_real = a_candidates_proj_real[which, 0]  # action_scale already applied pre-projection
                prev_actions_real = a_candidates_proj_real[which:which+1]

            

                # print("[PROJ] costs:", proj_costs)
                # delta_change = np.linalg.norm(a_candidates_proj_real - a_candidates_real, axis=(1,2))  # (K,)
                # print("[PROJ] mean|proj-change|:", float(delta_change.mean()), "min:", float(delta_change.min()), "max:", float(delta_change.max()))

                # Convert delta-pos -> absolute command (your env seems to accept xyz setpoints)
                cmd_xyz = pos.copy()
                cmd_xyz[:action_dim] = cmd_xyz[:action_dim] + a0_real[:action_dim]
                # cmd_xyz[2] = 0.3 # If you want fixed altitude, uncomment:
                print(f"[MODEL OUTPUT] which={which} a0_real={a0_real}" f" cmd_xyz={cmd_xyz[:3]}")
                obs_next, rew, done_vec, info = env.step(cmd_xyz)

                # update rgb(d) history after step
                rgb = get_obs_frame_from_env(env, use_depth)
                rgb_hist.append(rgb)
                write_video_frame()
                # if step % 500 == 0:
                #     plt.imshow(rgb)
                #     plt.title("FPV Camera Frame")
                #     plt.axis("off")
                #     plt.show()

                # log
                # pos2 = env._pos_world().detach().cpu().numpy()[0]
                pos2 = obs_next[0] 
                # pos2 =cmd_xyz[:action_dim]
                traj_xyz.append(pos2.copy())
                actions_taken.append(a0_real.copy())

                logger.step(pos=pos, action=a0_real)

                done = bool(done_vec[0]) if isinstance(done_vec, (list, tuple, np.ndarray, torch.Tensor)) else bool(done_vec)
                print(f"step {step:04d} pos={pos2} done={done}")

                # xyz is a strict superset of xy (xyz[..., :2] == xy) -- integrate
                # once and slice for any xy-only consumer instead of integrating twice.
                cand_xyz = integrate_candidates_xyz(pos2, a_candidates_real)  # (K,H+1,3)
                # projected/safe version of the chosen candidate's horizon —
                # same data project_action_candidates_with_positions() already
                # computed above (a_candidates_proj_real), just also integrated
                # and kept instead of being discarded after action selection
                cand_xyz_proj = integrate_candidates_xyz(
                    pos2, a_candidates_proj_real[which:which + 1]
                )[0]  # (H+1,3)
                # plain/unguided model output for the same candidate index and
                # noise draw — see a_plain_real above (only differs from
                # a_candidates_real for "sdpc" mode's in-loop-guided sampling)
                cand_xyz_plain = integrate_candidates_xyz(
                    pos2, a_plain_real[which:which + 1]
                )[0]  # (H+1,3)
                cand_snapshots.append({
                    "step": step,
                    "pos": pos2.copy(),
                    # keys keep their original "traj_xy*" names for compatibility
                    # with old npz readers, but now carry (x,y,z) -- consumers
                    # slice [..., :2] for xy and check shape[-1] >= 3 for z.
                    "traj_xy": cand_xyz,
                    "traj_xy_proj": cand_xyz_proj,
                    "traj_xy_plain": cand_xyz_plain,
                    "chosen": int(which),
                    "cyl_xy":  env.get_cylinder_positions() if args.dynamic_obstacles_enabled else None,
                    "sph_xyz": env.get_sphere_positions()   if args.floating_spheres          else None,
                })

                if done:
                    print("[INFO] Done=True. Breaking episode loop.")
                    break

            for cam_name, writer in video_writers.items():
                writer.release()
                print(f"[INFO] Video saved -> {video_paths[cam_name]}")

            episode_wall_time_sec = time.time() - episode_start_time

            success = bool(info["success"][0])
            fell    = bool(info["fell"][0])
            logger.end_episode(success=success, fell=fell)
            if (ep + 1) % 5 == 0:
                logger.print_live_summary()
        
            # save raw trajectory and metadata for this episode
            traj_path = os.path.join(
                traj_dir,
                f"traj_{encoder_type}_L{latent_dim}_{variant_name}_seed{seedmodel}.npz"
            )
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
                boxes       = np.array(BOXES),           # (N, 2) box centers
                cylinders   = np.array(CYLINDERS),       # (N, 2) cylinder centers
                halfspaces  = np.array([[hs[0], hs[1]] for hs in active_halfspaces], dtype=object),
                hs_sides    = np.array([hs[2] for hs in active_halfspaces]),
                tighten     = vcfg.get("tighten", 0.0),
                use_projection = vcfg["use_projection"],
                projection_mode = vcfg["projection_mode"],
                num_candidates = vcfg["num_candidates"] if vcfg["num_candidates"] > 0 else 1,
                selection   = vcfg["selection"],

                # ── candidate snapshots (for replaying plan viz) ──
                # save full episode length (all steps), not just a truncated prefix
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

                # ── floating sphere metadata + per-step positions ──
                floating_spheres  = bool(args.floating_spheres),
                sphere_positions  = np.array(SPHERES, dtype=np.float32) if SPHERES
                                    else np.zeros((0, 3), dtype=np.float32),
                sphere_radius     = float(SPHERE_RADIUS),
                sphere_amplitude  = float(sphere_amplitude),
                sphere_frequency  = float(sphere_frequency),
                sph_xyz_traj = np.array(
                    [s["sph_xyz"] for s in cand_snapshots if s.get("sph_xyz") is not None],
                    dtype=np.float32,
                ) if any(s.get("sph_xyz") is not None for s in cand_snapshots)
                  else np.zeros((0, max(len(SPHERES), 1), 3), dtype=np.float32),
            )
            print(f"[TRAJ] saved: {traj_path}")

            # ------------------ Plot episode ------------------
            if len(traj_xyz) > 0:
                traj_xyz_np = np.stack(traj_xyz, axis=0).astype(np.float32)

                # Z vs t
                plt.figure(figsize=(7, 4))
                plt.plot(traj_xyz_np[:, 2])
                if args.floating_spheres and SPHERES:
                    _sr  = SPHERE_RADIUS
                    _amp = sphere_amplitude
                    for (_, _, _sz) in SPHERES:
                        plt.axhspan(_sz - _sr - _amp, _sz + _sr + _amp,
                                    color="#4a9de0", alpha=0.12)
                        plt.axhspan(_sz - _sr, _sz + _sr,
                                    color="#4a9de0", alpha=0.22)
                plt.ylim(-0.05, 1.15)   # fix to flight envelope [0, 1] with small padding
                plt.axhline(0.0, color="#888888", linewidth=0.8, linestyle=":")   # floor
                plt.axhline(1.0, color="#888888", linewidth=0.8, linestyle=":")   # ceiling
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
                add_obstacles_xy(ax, BOXES, [], box_size_xy=0.20, cyl_radius=0.06)  # boxes only
                if args.dynamic_obstacles_enabled:
                    dyn_idx = args.dynamic_cyl_indices if args.dynamic_cyl_indices is not None \
                              else list(range(len(CYLINDERS)))
                    _raw_axes = args.obs_axes if args.obs_axes is not None else ["y"]
                    dyn_axes_resolved = (_raw_axes * len(dyn_idx))[:len(dyn_idx)] if len(_raw_axes) == 1 else _raw_axes
                    dyn_set = set(dyn_idx)
                    static_cyls = [CYLINDERS[i] for i in range(len(CYLINDERS)) if i not in dyn_set]
                    dyn_cyls    = [CYLINDERS[i] for i in dyn_idx]
                    add_obstacles_xy(ax, [], static_cyls, box_size_xy=0.20, cyl_radius=0.06)
                    add_dynamic_cylinders_xy(
                        ax, dyn_cyls, dyn_axes_resolved, cand_snapshots,
                        obs_amplitude=obs_amplitude, cyl_radius=0.06,
                        drone_radius=drone_radius,
                    )
                else:
                    add_obstacles_xy(ax, [], CYLINDERS, box_size_xy=0.20, cyl_radius=0.06)

                # Sphere obstacles overlay
                if args.floating_spheres and SPHERES:
                    from matplotlib.patches import Circle as _Circle
                    _sr  = SPHERE_RADIUS
                    _amp = sphere_amplitude
                    for (sx, sy, _sz) in SPHERES:
                        ax.add_patch(_Circle(
                            (sx, sy), _sr,
                            linewidth=0.8, edgecolor="#1f6fbf", facecolor="#4a9de0",
                            alpha=0.40, zorder=2,
                        ))
                        ax.add_patch(_Circle(
                            (sx, sy), _sr + _amp,
                            linewidth=0.8, edgecolor="#4a9de0", facecolor="none",
                            linestyle=":", alpha=0.30, zorder=1,
                        ))
                ax.set_xlim(-0.5, 4.5)
                ax.set_ylim(-1.0, 1.0)
                if active_halfspaces:
                    plot_halfspace_constraints_xy(ax, active_halfspaces, (-0.5, 4.5), (-1.0, 1.0))

                # ---- blue constraint margin overlay ----
                tighten_val = vcfg.get("tighten", 0.0)
                constraint_handles = []
                if vcfg["use_projection"]:
                    if args.dynamic_obstacles_enabled:
                        _dyn_set_plot = set(args.dynamic_cyl_indices) if args.dynamic_cyl_indices is not None \
                                        else set(range(len(CYLINDERS)))
                        _static_c = [CYLINDERS[i] for i in range(len(CYLINDERS)) if i not in _dyn_set_plot]
                        _dyn_c    = [CYLINDERS[i] for i in range(len(CYLINDERS)) if i in     _dyn_set_plot]
                        # dynamic cylinders: no blue circle — orange swept band already shows range
                        constraint_handles = plot_constraint_overlay(
                            ax, BOXES, _static_c,
                            tighten=tighten_val, drone_radius=drone_radius,
                            x_bounds=(-0.5, 4.5), y_bounds=(-1.0, 1.0),
                        )
                    else:
                        constraint_handles = plot_constraint_overlay(
                            ax, BOXES, CYLINDERS,
                            tighten=tighten_val,
                            drone_radius=drone_radius,
                            x_bounds=(-0.5, 4.5),
                            y_bounds=(-1.0, 1.0),
                        )
                # --------------------------------------------

                # Overlay candidate rollouts snapshots
                for snap in cand_snapshots:
                    traj_xy = snap["traj_xy"]      # (K,H+1,2)
                    chosen = snap["chosen"]

                    K = traj_xy.shape[0]
                    for k in range(K):
                        xy = traj_xy[k]

                        if k == chosen:
                            # ax.plot(xy[:, 0], xy[:, 1], linewidth=3.5, alpha=0.9) #for chosen trajectory
                            pass
                        else:
                            # ax.plot(xy[:, 0], xy[:, 1], linewidth=1.5, alpha=0.25) #for unchosen trajectories
                            pass

                    # Mark snapshot start position
                    ax.scatter(snap["pos"][0], snap["pos"][1], s=12, alpha=0.4)

                print("[DEBUG] cand_snapshots count:", len(cand_snapshots))
                if len(cand_snapshots) > 0:
                    print("[DEBUG] first snapshot traj_xy shape:", cand_snapshots[0]["traj_xy"].shape)

                ax.set_xlabel("x")
                ax.set_ylabel("y")
                ax.set_title(f"XY trajectory with{encoder_type}, {latent_dim} and {variant_name}")  #and {variant_name}
                ax.set_aspect("equal", adjustable="box")  # equal scale without expanding y beyond corridor
                ax.set_xlim(-0.5, 4.5)
                ax.set_ylim(-1.25, 1.25)
                ax.grid(True, alpha=0.3)
                ax.legend()

                # merge legend handles
                handles, labels = ax.get_legend_handles_labels()
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

    env.close()
    os._exit(0)


if __name__ == "__main__":
    main()
