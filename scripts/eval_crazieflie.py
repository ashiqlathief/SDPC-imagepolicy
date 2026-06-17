import argparse
import os
from collections import deque
from pathlib import Path
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import re 
from matplotlib.patches import Rectangle, Circle

import diffuser.utils as utils
from diffuser.sampling.projection import Projector
from metrics_logger import MetricsLogger   

BOXES = [
    #experiment1
    (3.0,  0.2),
    (2.0, -0.2),
    (1.1,  0.2),
    (3.2, -0.3), #(3.7, -0.3),
    (2.3, -0.2),
    (1.6, -0.3),

#   #experiment2
    # (3.5, -0.4),
    # (2.2, -0.4),
    # (1.6, -0.4),
    # (3.7, -0.4),
    # (2.8, -0.4),
    # (1.0, -0.4),
    # (3.5, 0.7),
    # (2.2, 0.7),
    # (1.6, 0.7),
    # (3.7, 0.7),
    # (2.8, 0.7),
    # (1.0, 0.7),

    #experiment3
    # (3.0,  0.2),
    # (2.0, -0.2),
    # (1.1,  0.2),
    # (0.5, -0.7), #(3.7, -0.3),
    # (0.5, 0.7),
    # (1.6, -0.3),
]

CYLINDERS = [
    #experiment1
    (1.0, -0.2),
    (2.3,  0.3),
    (1.7,  0.4),
    (3.0,  0.4),
    (1.5, 0.4),#(2.0,  0.4), --- IGNORE ---
    (2.0,  0.3),

#   #experiment2
    # (1.4, 0.4),
    # (2.8, 0.3),
    # (1.2, 0.4),
    # (3.5, 0.3),
    # (2.0, 0.4),
    # (2.5, 0.4),
    # (1.4, -0.7),
    # (2.8, -0.8),
    # (1.2, -0.7),
    # (3.5, -0.8),
    # (3.2, -0.7),
    # (2.5, -0.7),

    # experiment3
    # (1.0, -0.2),
    # (2.3,  0.3),
    # (1.7,  0.4),
    # (3.0,  0.4),
    # (1.5, 0.4), #(2.0,  0.4), --- IGNORE ---
    # (2.0,  0.3),
]

corridor_halfspaces = [
# [[-0.2, -0.2], [1.1, 0.0], 'above'],   # hard (\ shape)
# [[-0.2, 0.2], [1.2, 0.0], 'below'],     # hard (/ shape) toprighthard
# [[-0.2, -0.2], [1.1, 0.0], 'above'],   # easier (\ shape)
# [[-0.2, 0.2], [1.2, 0.0], 'below'],   # easier (/ shape)
# [[-0.2, -0.2], [1.1, 0.0], 'below'],   # easier (\ shape)
# [[-0.2, 0.2], [1.2, 0.0], 'below'],   # easier (/ shape)
]

# Z flight-envelope halfspaces.
# Format: (normal_3d, rhs)  →  normal · [x, y, z] <= rhs
# tighten shrinks the envelope from both ends (ceiling down, floor up).
WALL_HEIGHT = 1.0   # metres — obstacle / wall tops in the corridor
z_halfspaces = [
    ([0.0, 0.0,  1.0], WALL_HEIGHT),   # z <=  1.0 m  — drone cannot fly above wall/ceiling
    ([0.0, 0.0, -1.0], 0.0),           # z >= 0.0 m   — drone cannot go underground
]

projection_variants = [
#   'dpcc-r', 
#   'dpcc-r-tightened',
  'dpcc-c',
  'dpcc-c-tightened',
  'dpcc-t',
  'dpcc-t-tightened',
  'diffuser',
  'gradient',
  'gradient-tightened',
  'post_processing',
  'post_processing-tightened',
#   'model_free',
#   'model_free-tightened',
  'dpcc-c-tightened-dt0p25',
  'dpcc-c-tightened-dt0p5',
  'dpcc-c-tightened-dt2p0',
  'dpcc-c-tightened-dt4p0',
]

def variant_cfg(name: str):
    cfg = dict(
        num_candidates=1,
        selection="first",
        use_projection=False,
        projection_mode="none",    # "none" | "post" | "gradient"
        tighten=0.0,
        dt=None,
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

    elif name == "model_free":
        # baseline: no diffusion, just go straight to target (simple P controller)
        cfg.update(num_candidates=1, selection="first", use_projection=False, use_dynamics=False)

    elif name == "model_free-tightened":
        cfg.update(num_candidates=1, selection="first", use_projection=False, use_dynamics=False)

    elif name == "dpcc-c":
        cfg.update(num_candidates=8, selection="minimum_projection_cost", use_projection=True, projection_mode="dpcc", tighten=0.0)

    elif name == "dpcc-c-tightened":
        cfg.update(num_candidates=8, selection="minimum_projection_cost", use_projection=True, projection_mode="dpcc", tighten=0.05)

    elif name == "dpcc-t":
        cfg.update(num_candidates=8, selection="temporal_consistency", use_projection=True, projection_mode="dpcc", tighten=0.0)

    elif name == "dpcc-t-tightened":
        cfg.update(num_candidates=8, selection="temporal_consistency", use_projection=True, projection_mode="dpcc", tighten=0.05)

    elif name == "dpcc-r":
        # dpcc-r often means single sample with projection (repair)
        cfg.update(num_candidates=1, selection="first", use_projection=True, projection_mode="dpcc", tighten=0.0)

    elif name == "dpcc-r-tightened":
        cfg.update(num_candidates=1, selection="first", use_projection=True, projection_mode="dpcc", tighten=0.05)

    # dt sweeps (Table 2)
    elif name.startswith("dpcc-c-tightened-dt"):
        # parse dt from string, e.g. dt0p25 -> 0.25
        dt_str = name.split("dt")[-1].replace("p", ".")
        cfg.update(num_candidates=8, selection="minimum_projection_cost", use_projection=True, projection_mode="dpcc", tighten=0.05, dt=float(dt_str))

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

def add_dynamic_cylinders_xy(ax, cyl_rest, cand_snapshots, obs_amplitude,
                              cyl_radius=0.06, drone_radius=0.0):
    """
    Draw dynamic cylinder visualisation on an XY axes:
      - orange band  = full oscillation sweep of the exclusion zone
                       (cyl_radius + drone_radius wide, y0 ± amplitude tall)
      - dashed circle = exclusion zone at rest position
      - faded dots    = physical cylinder position sampled every few steps
    """
    excl_r = cyl_radius + drone_radius   # total exclusion radius

    for (x0, y0) in cyl_rest:
        ylo = max(-0.85, y0 ) #- obs_amplitude
        yhi = min( 0.85, y0 ) #+ obs_amplitude
        # swept exclusion band
        ax.add_patch(Rectangle(
            (x0 - excl_r, ylo), 2 * excl_r, yhi - ylo,
            facecolor="tab:orange", alpha=0.15, zorder=1,
        ))
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
        dists = np.linalg.norm(
            actions_real[:, :-1, :] - prev_actions_real[:, 1:, :],
            axis=(1, 2)
        )
        return int(np.argmin(dists))

    if strategy == "minimum_projection_cost":
        if isinstance(infos, dict) and "projection_costs" in infos:
            costs_total = np.zeros(K, dtype=np.float32)
            for _, cost in infos["projection_costs"].items():
                costs_total += np.asarray(cost, dtype=np.float32)
            return int(np.argmin(costs_total))

    return 0

def integrate_candidates_xy(pos0_xyz, a_candidates_real):
    """
    pos0_xyz: (3,)
    a_candidates_real: (K,H,3)
    Returns: traj_xy (K, H+1, 2)
    """
    K, H, D = a_candidates_real.shape
    traj_xy = np.zeros((K, H + 1, 2), dtype=np.float32)
    traj_xy[:, 0, :] = pos0_xyz[:2][None, :]
    traj_xy[:, 1:, :] = pos0_xyz[:2][None, None, :] + np.cumsum(a_candidates_real[:, :, :2], axis=1)
    return traj_xy

def build_obstacle_constraint_list(boxes, cylinders, x_bounds=None, y_bounds=None, z_bounds=None,
                                    corridor_halfspaces=None, z_halfspaces=None, tighten=0.0,
                                    cyl_extra_radius=0.0, drone_radius=0.0):
    constraint_list = []

    # ---------------- bounds ----------------
    lb = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float32)
    ub = np.array([ np.inf,  np.inf,  np.inf], dtype=np.float32)

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

    for (x, y) in cylinders:
        center = [float(x), float(y)]
        constraint_list.append(("sphere_outside", [0, 1], center,
                                 0.06 + _dr + float(tighten) + float(cyl_extra_radius)))

    # for (x, y, z) in boxes:
    #     center = [float(x), float(y), float(z)]
    #     constraint_list.append(("sphere_outside", [0, 1, 2], center, box_sphere_r))

    # for (x, y, z) in cylinders:
    #     center = [float(x), float(y), float(z)]
    #     constraint_list.append(("sphere_outside", [0, 1, 2], center, cyl_sphere_r))

    # ---------------- halfspaces in XY ----------------
    # trajectory_dim = 3
    # act_obs_indices = {"x": 0, "y": 1, "z": 2}
    # for hs in corridor_halfspaces:
    #     C_row, d = utils.formulate_halfspace_constraints(
    #         hs,
    #         enlarge_constraints=0.025 + float(tighten),     # your margin
    #         trajectory_dim=trajectory_dim,
    #         act_obs_indices=act_obs_indices,
    #     )
    #     constraint_list.append(("ineq", (C_row.astype(np.float32), float(d))))

    # ---------------- z halfspace envelope ----------------
    # Each entry: (normal_3d, rhs)  →  normal · [x, y, z] <= rhs - tighten
    # Ceiling: [0,0,1]·p <= WALL_HEIGHT - tighten  (shrinks ceiling down)
    # Floor:   [0,0,-1]·p <= 0.0 - tighten        (shrinks floor up — use tighten=0 here)
    if z_halfspaces is not None:
        for (normal, rhs) in z_halfspaces:
            C_row = np.array(normal, dtype=np.float32)
            constraint_list.append(("ineq", (C_row, float(rhs) - float(tighten))))

    return constraint_list

def diagnose_projector(projector, device,
                       x_bounds=(-0.5, 4.5), y_bounds=(-0.95, 0.95), z_bounds=(0.0, 1.0)):
    """
    Two-part diagnostic:

    Part 1 — C matrix inspection
        Scans every row of projector.C_np to find rows that enforce each
        x/y/z bound.  If a bound is missing from C, it was never wired in.

    Part 2 — live test projection
        Builds a trajectory where every future position is far outside every
        bound, then projects it and checks whether the result is inside bounds.
        This is the ground-truth answer to "does the bound actually work?".
    """
    H   = projector.horizon
    dim = projector.transition_dim
    C   = projector.C_np   # (n_rows, H*dim)
    d   = projector.d_np   # (n_rows,)

    print("\n" + "="*60)
    print(f"[DIAGNOSE] C shape: {C.shape}  →  {C.shape[0]} constraint rows")
    print(f"           d shape: {d.shape}")

    # ── Part 1: scan C rows for each bound ──────────────────────────────────
    dim_names = ["x", "y", "z"]
    bounds = [x_bounds, y_bounds, z_bounds]

    for dim_idx in range(dim):
        lb, ub = bounds[dim_idx]
        name   = dim_names[dim_idx]

        # lb row:  C[row, t*dim+dim_idx] = -1,  d[row] = -lb  → -x <= -lb → x >= lb
        # ub row:  C[row, t*dim+dim_idx] = +1,  d[row] = +ub  → x <= ub
        lb_rows, ub_rows = [], []
        for row in range(C.shape[0]):
            active_cols = np.where(np.abs(C[row]) > 1e-8)[0]
            if len(active_cols) == 1:
                col = active_cols[0]
                if col % dim == dim_idx:          # this row touches our dimension
                    val = C[row, col]
                    rhs = d[row]
                    if val < 0:                   # -1 * x_dim <= -lb  →  x_dim >= lb
                        lb_rows.append((row, -rhs))
                    else:                          # +1 * x_dim <= ub
                        ub_rows.append((row, rhs))

        lb_ok = len(lb_rows) > 0
        ub_ok = len(ub_rows) > 0
        print(f"\n  {name}_bounds = ({lb}, {ub})")
        print(f"    lb rows found: {len(lb_rows)}  {'✓' if lb_ok else '✗ MISSING — lb NOT enforced!'}")
        print(f"    ub rows found: {len(ub_rows)}  {'✓' if ub_ok else '✗ MISSING — ub NOT enforced!'}")
        if lb_rows:
            sample_rhs = [f"{v:.3f}" for _, v in lb_rows[:3]]
            print(f"    sample lb rhs values (should all be {lb}): {sample_rhs}")
        if ub_rows:
            sample_rhs = [f"{v:.3f}" for _, v in ub_rows[:3]]
            print(f"    sample ub rhs values (should all be {ub}): {sample_rhs}")

    # ── Part 2: live projection test ─────────────────────────────────────────
    print("\n  [TEST] Projecting a trajectory that violates ALL bounds ...")

    # pos0 = valid position, all future steps far outside every bound
    test_np = np.zeros((1, H, dim), dtype=np.float32)
    test_np[0, 0] = [2.0, 0.0, 0.5]    # t=0: inside bounds (skip_initial_state)
    for t in range(1, H):
        test_np[0, t] = [9.0, 3.0, 3.0]  # t>0: clearly outside x, y, z

    test_t     = torch.tensor(test_np, device=device)
    proj_t, _  = projector.project(test_t)
    proj_np    = proj_t.squeeze(0).detach().cpu().numpy()   # (H, dim)

    all_ok = True
    for t in range(1, H):
        x, y, z = proj_np[t]
        x_ok = x_bounds[0] - 1e-3 <= x <= x_bounds[1] + 1e-3
        y_ok = y_bounds[0] - 1e-3 <= y <= y_bounds[1] + 1e-3
        z_ok = z_bounds[0] - 1e-3 <= z <= z_bounds[1] + 1e-3
        status = "OK" if (x_ok and y_ok and z_ok) else "FAIL"
        if status == "FAIL":
            all_ok = False
            print(f"    t={t}: x={x:.4f} y={y:.4f} z={z:.4f}  ← {status}")

    if all_ok:
        print(f"    All {H-1} future steps projected inside bounds  ✓")
    else:
        print(f"    Some steps still outside bounds after projection  ✗")
        print(f"    (SLSQP may have hit maxiter=1000 without converging)")

    print("="*60 + "\n")


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


def build_position_projector(horizon_H, gradient, device, boxes, cylinders, normalizer=None,
                             tighten=0.0, dt=0.1, use_dynamics=True, obs_amplitude=0.0,
                             drone_radius=0.0):
    # We project POSITIONS, so we need horizon = H+1
    Hp1 = horizon_H + 1
    x_bounds = (-0.5, 4.5)
    y_bounds = (-0.95, 0.95)
    z_bounds = (0.0, 1.0)

    constraint_list = build_obstacle_constraint_list(
        boxes=boxes,
        cylinders=cylinders,
        x_bounds=x_bounds,
        y_bounds=y_bounds,
        z_bounds=z_bounds,
        corridor_halfspaces=corridor_halfspaces,
        z_halfspaces=z_halfspaces,
        tighten=tighten,
        cyl_extra_radius=obs_amplitude,
        drone_radius=drone_radius,
    )

    projector = Projector(
        horizon=Hp1,
        transition_dim=3,         # x,y,z positions
        action_dim=0,
        goal_dim=0,
        constraint_list=constraint_list,
        normalizer=normalizer,          # start without normalizer for debugging
        gradient=gradient,
        gradient_weights=[1, 0.5, 2], dt=dt if use_dynamics else 0.0,   # 0.0 = no Euler dynamics enforced
        variant="states",
        skip_initial_state=True,
        diffusion_timestep_threshold=0.8,
        device=str(device),
        solver="scipy",           # your file uses scipy SLSQP path
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
    proj_costs = np.zeros(K, dtype=np.float32)
    a_proj_real = np.zeros_like(a_candidates_real, dtype=np.float32)

    for k in range(K):
        # integrate deltas -> positions (H+1,3)
        pos_traj = np.zeros((H + 1, 3), dtype=np.float32)
        pos_traj[0] = pos0.astype(np.float32)
        pos_traj[1:] = pos_traj[0] + np.cumsum(a_candidates_real[k], axis=0)

        pos_t = torch.tensor(pos_traj[None, :, :], dtype=torch.float32, device=device)  # (1,H+1,3)

        pos_proj_t, cost = projector.project(pos_t)  # (1,H+1,3), cost shape (1,)
        pos_proj = pos_proj_t.squeeze(0).detach().cpu().numpy()  # (H+1,3)

        # convert back to deltas (H,3)
        a_proj = pos_proj[1:] - pos_proj[:-1]

        a_proj_real[k] = a_proj
        proj_costs[k] = float(cost[0])

    return a_proj_real, proj_costs

def build_inloop_projector(horizon_H, device, tighten=0.0, dt=0.1):
    """
    Projector that runs INSIDE the diffusion denoising loop.
    Operates on normalized action sequences (K, H, 3) — NOT positions.
    Uses only action bounds (no obstacle geometry, since x is in normalized action space).
    """
    # Action bounds: clamp normalized actions to [-1, 1] per dim
    lb = np.array([-1.0, -1.0, -1.0], dtype=np.float32)
    ub = np.array([ 1.0,  1.0,  1.0], dtype=np.float32)
    constraint_list = [("lb", lb), ("ub", ub)]

    projector = Projector(
        horizon=horizon_H,        # H, not H+1
        transition_dim=3,
        action_dim=3,
        goal_dim=0,
        constraint_list=constraint_list,
        normalizer=None,
        gradient=False,
        variant="actions",        # operating on actions, not states
        skip_initial_state=False,
        dt=dt,
        diffusion_timestep_threshold=0.8,
        device=str(device),
        solver="scipy",
    )
    return projector

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--ckpt", type=str, default="state_best.pt")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--action_scale", type=float, default=10.0)
    parser.add_argument("--projection_yaml", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=1)   # how many episodes to roll
    parser.add_argument("--num_candidates", type=int, default=4)
    parser.add_argument("--trajectory_selection",type=str,default="first",choices=["first", "temporal_consistency", "minimum_projection_cost"],)
    parser.add_argument("--dynamic_obstacles", action="store_true", default=False,
                        help="Enable sinusoidal cylinder movement during evaluation")
    parser.add_argument("--obs_amplitude", type=float, default=0.35,
                        help="Obstacle oscillation amplitude in metres")
    parser.add_argument("--obs_frequency", type=float, default=0.25,
                        help="Obstacle oscillation frequency in Hz")
    parser.add_argument("--dynamic_proj_update", action="store_true", default=False,
                        help="Rebuild projector every step with actual cylinder positions "
                             "(accurate but slower). Default: worst-case radius expansion.")
    parser.add_argument("--drone_radius", type=float, default=0.10,
                        help="Drone bounding circle radius in metres (Minkowski expansion "
                             "applied to all XY obstacle constraints). Default 0.10m for Crazyflie.")
    parser.add_argument("--dynamic_cyl_indices", type=int, nargs="*", default=None,
                        help="Indices of cylinders to move (0-based). Omit to move all cylinders. "
                             "Example: --dynamic_cyl_indices 0 2 4")
    args, _unknown = parser.parse_known_args()
    from isaac.scripts.crazyflie_env import Crazyflie, CrazyflieEnvCfg
    device = torch.device(args.device)

    # ------------------ Create env ------------------
    cfg = CrazyflieEnvCfg(
        num_envs=1,
        device=str(device),
        dynamic_obstacles=args.dynamic_obstacles,
        obs_amplitude=args.obs_amplitude,
        obs_frequency=args.obs_frequency,
        dynamic_cyl_indices=args.dynamic_cyl_indices,
    )
    env = Crazyflie(cfg)

    # ------------------ Load trained experiment ------------------
    print(f"[INFO] Loading run dir: {args.run_dir}")

    seedmodel = int(Path(args.run_dir).name)   # gives 9
    diff_exp = utils.load_diffusion(args.run_dir,epoch="best",device=str(device),)
    dataset = diff_exp.dataset
    diffusion = diff_exp.diffusion.to(device)
    diffusion.eval()

    run_name = Path(args.run_dir).parent.name  # e.g. "H16_K20_ENCvit_LAT256"
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
    # path = os.path.join(args.run_dir, "results")
    # os.makedirs(path, exist_ok=True)
    # os.makedirs("plots", exist_ok=True)
    logger = MetricsLogger(                                             
        save_dir=os.path.join(args.run_dir, "results"),                               
        encoder_type=encoder_type,
        latent_dim=latent_dim,
        horizon=horizon,
        n_diffusion_steps=20,
        num_candidates=args.num_candidates,
        corridor_halfspaces  = corridor_halfspaces,
        boxes                = BOXES,
        cylinders            = CYLINDERS,              
    )    
    # ------------------ Episodes ------------------
    # for ep in range(args.episodes):
    for ep, variant_name in enumerate(projection_variants):
        vcfg = variant_cfg(variant_name)
        args.num_candidates = vcfg["num_candidates"] if vcfg["num_candidates"] > 0 else args.num_candidates
        # args.trajectory_selection = vcfg["selection"]
        # ------------------ Optional projection ------------------
        projection_mode=vcfg["projection_mode"]
        gradient = (projection_mode == "gradient")
        pos_projector    = None   # post-processing: operates on positions (H+1)
        inloop_projector = None   # dpcc in-loop: operates on actions (H)

        if vcfg["use_projection"]:
            proj_dt = vcfg["dt"] if vcfg["dt"] is not None else 0.1
            # All cylinders (static and dynamic) use the same base radius.
            # Dynamic ones get their actual positions via --dynamic_proj_update.
            pos_projector = build_position_projector(
                horizon_H=horizon, gradient=gradient, device=device,
                boxes=BOXES, cylinders=CYLINDERS,
                normalizer=None, tighten=vcfg["tighten"], dt=proj_dt,
                use_dynamics=vcfg.get("use_dynamics", True),
                drone_radius=args.drone_radius,
            )
            # ── run once per variant to confirm bounds are wired correctly ──
            diagnose_projector(pos_projector, device,
                               x_bounds=(-0.5, 4.5), y_bounds=(-0.95, 0.95), z_bounds=(0.0, 1.0))
            if vcfg["projection_mode"] == "dpcc":
                inloop_projector = build_inloop_projector(
                    horizon_H=horizon, device=device,
                    tighten=vcfg["tighten"], dt=proj_dt,
                )

        print(f"\n[INFO] ===== Episode {ep+1}/{args.episodes}_{variant_name} =====")
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
        

        # init rgb history
        rgb0 = get_rgb_from_env(env)
        rgb_hist = deque(maxlen=To)
        for _ in range(To):
            rgb_hist.append(rgb0.copy())

        traj_xyz = []
        actions_taken = []
        prev_actions_real = None
        cand_snapshots = []

        pos_init = env._pos_world().detach().cpu().numpy()[0]
        traj_xyz.append(pos_init.copy())
        # pos = env._pos_world().detach().cpu().numpy()[0]  # [x,y,z]
        # target = env.target_pos[0].detach().cpu().numpy() if hasattr(env, "target_pos") else np.array([np.nan, np.nan, np.nan])
        # cmd_xyz = pos.copy()
        for step in range(args.max_steps):
            # current position (for logging/command conversion)
            pos = env._pos_world().detach().cpu().numpy()[0]  # [x,y,z]
            target = env.target_pos[0].detach().cpu().numpy() if hasattr(env, "target_pos") else np.array([np.nan, np.nan, np.nan])
            
            # ── model_free: bypass diffusion entirely ─────────────────────
            if variant_name.startswith("model_free"):
                gain = 0.3
                cmd_xyz = pos.copy()
                cmd_xyz[:3] = pos[:3] + gain * (target[:3] - pos[:3])
                obs_next, rew, done_vec, info = env.step(cmd_xyz)
                rgb = get_rgb_from_env(env)
                rgb_hist.append(rgb)
                pos2 = obs_next[0]
                traj_xyz.append(pos2.copy())
                actions_taken.append(cmd_xyz[:3] - pos[:3])
                logger.step(pos=pos, action=cmd_xyz[:3] - pos[:3])
                done = bool(done_vec[0]) if isinstance(done_vec, (list, tuple, np.ndarray, torch.Tensor)) else bool(done_vec)
                if done:
                    break
                continue   # skip the diffusion path below

            # ── diffusion path ────────────────────────────────────────────
            # build condition
            obs_rgb_t = preprocess_rgb_stack(rgb_hist).to(device)  # (1,To,3,H,W)
            cond = {"obs_rgb": obs_rgb_t}
            # a_chunk = sample_action_chunk(diffusion=diffusion,cond=cond,horizon=horizon,action_dim=action_dim,device=device,)
            # a_chunk_real = dataset.action_normalizer.unnormalize(a_chunk)
            # a0_real = a_chunk_real[0]* float(args.action_scale)

            # -------------------------------------------------
            # Sample K candidate chunks (normalized action space)
            # -------------------------------------------------
            in_loop_projector = inloop_projector if vcfg["projection_mode"] == "dpcc" else None

            a_candidates_norm, infos = sample_action_candidates(diffusion=diffusion,cond=cond,horizon=horizon,action_dim=action_dim,
                                                                num_candidates=args.num_candidates,projector=in_loop_projector)   # (K, H, D)

            # Unnormalize all candidates to real delta-pos
            a_candidates_real = dataset.action_normalizer.unnormalize(a_candidates_norm)   # (K, H, D)
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

            # ── per-step projector update for dynamic obstacles ──────────
            if args.dynamic_obstacles and args.dynamic_proj_update and vcfg["use_projection"]:
                # get_cylinder_positions() returns current positions for ALL cylinders
                # (static ones at rest, dynamic ones at actual current position)
                cyl_now = env.get_cylinder_positions()
                pos_projector = build_position_projector(
                    horizon_H=horizon, gradient=gradient, device=device,
                    boxes=BOXES, cylinders=cyl_now,
                    normalizer=None, tighten=vcfg["tighten"], dt=proj_dt,
                    use_dynamics=vcfg.get("use_dynamics", True),
                    obs_amplitude=0.0,  # exact positions — no extra radius needed
                    drone_radius=args.drone_radius,
                )

            # ── projection + selection  ---------
            if vcfg["use_projection"] and vcfg["projection_mode"] in ("post", "dpcc"):
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

            a0_real = a_candidates_proj_real[which, 0] * float(args.action_scale)
            prev_actions_real = a_candidates_proj_real[which:which+1]

            # print("[PROJ] costs:", proj_costs)
            # delta_change = np.linalg.norm(a_candidates_proj_real - a_candidates_real, axis=(1,2))  # (K,)
            # print("[PROJ] mean|proj-change|:", float(delta_change.mean()), "min:", float(delta_change.min()), "max:", float(delta_change.max()))

            # Convert delta-pos -> absolute command (your env seems to accept xyz setpoints)
            cmd_xyz = pos.copy()
            cmd_xyz[:action_dim] = cmd_xyz[:action_dim] + a0_real[:action_dim]
            # cmd_xyz[2] = 0.3 # If you want fixed altitude, uncomment:

            obs_next, rew, done_vec, info = env.step(cmd_xyz)

            # update rgb history after step
            rgb = get_rgb_from_env(env)
            rgb_hist.append(rgb)
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
            print(f"step {step:04d} pos={pos2} target={target} done={done}")

            traj_xy = integrate_candidates_xy(pos2, a_candidates_real)  # (K,H+1,2)
            cand_snapshots.append({
                "step": step,
                "pos": pos2.copy(),
                "traj_xy": traj_xy,
                "chosen": int(which),
                "cyl_xy": env.get_cylinder_positions() if args.dynamic_obstacles else None,
            })

            if done:
                print("[INFO] Done=True. Breaking episode loop.")
                break
        success = bool(info["success"][0])                             
        fell    = bool(info["fell"][0])                                
        logger.end_episode(success=success, fell=fell)
        if (ep + 1) % 5 == 0:
            logger.print_live_summary()
        
        # save raw trajectory and metadata for this episode
        traj_dir = os.path.join(args.run_dir, "trajectories")
        os.makedirs(traj_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        traj_path = os.path.join(
            traj_dir, 
            f"traj_{encoder_type}_L{latent_dim}_{variant_name}.npz"
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
            boxes       = np.array(BOXES),           # (N, 2) box centers
            cylinders   = np.array(CYLINDERS),       # (N, 2) cylinder centers
            halfspaces  = np.array([[hs[0], hs[1]] for hs in corridor_halfspaces],dtype=object),
            hs_sides    = np.array([hs[2] for hs in corridor_halfspaces]),
            tighten     = vcfg.get("tighten", 0.0),
            use_projection = vcfg["use_projection"],
            projection_mode = vcfg["projection_mode"],
            num_candidates = vcfg["num_candidates"] if vcfg["num_candidates"] > 0 else 1,
            selection   = vcfg["selection"],

            # ── candidate snapshots (for replaying plan viz) ──
            # save first N snapshots to keep file size reasonable
            snap_pos    = np.array([s["pos"]    for s in cand_snapshots[:50]]),
            snap_chosen = np.array([s["chosen"] for s in cand_snapshots[:50]]),
        )
        print(f"[TRAJ] saved: {traj_path}")

        # ------------------ Plot episode ------------------
        if len(traj_xyz) > 0:
            traj_xyz_np = np.stack(traj_xyz, axis=0).astype(np.float32)

            # Z vs t
            plt.figure(figsize=(7, 4))
            plt.plot(traj_xyz_np[:, 2])
            if np.isfinite(target).all():
                plt.axhline(target[2], linestyle="--")
            plt.xlabel("timestep")
            plt.ylabel("z")
            plt.title(f"Z over time with {encoder_type}, {latent_dim} and {variant_name}") # and {variant_name}
            plt.tight_layout()
            plot_dir = os.path.join(args.run_dir, "plots", f"{encoder_type}_{latent_dim}")
            os.makedirs(plot_dir, exist_ok=True)

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

            # Target
            ax.scatter(target[0], target[1], marker="*", s=180, label="target")

            # Obstacles overlay
            add_obstacles_xy(ax, BOXES, [], box_size_xy=0.20, cyl_radius=0.06)  # boxes only
            if args.dynamic_obstacles:
                dyn_idx = args.dynamic_cyl_indices if args.dynamic_cyl_indices is not None \
                          else list(range(len(CYLINDERS)))
                dyn_set = set(dyn_idx)
                static_cyls = [CYLINDERS[i] for i in range(len(CYLINDERS)) if i not in dyn_set]
                dyn_cyls    = [CYLINDERS[i] for i in dyn_idx]
                add_obstacles_xy(ax, [], static_cyls, box_size_xy=0.20, cyl_radius=0.06)
                add_dynamic_cylinders_xy(
                    ax, dyn_cyls, cand_snapshots,
                    obs_amplitude=args.obs_amplitude, cyl_radius=0.06,
                    drone_radius=args.drone_radius,
                )
            else:
                add_obstacles_xy(ax, [], CYLINDERS, box_size_xy=0.20, cyl_radius=0.06)
            ax.set_xlim(-0.5, 4.5)
            ax.set_ylim(-1.0, 1.0)
            # plot_halfspace_constraints_xy(ax, corridor_halfspaces, (-0.5, 4.5), (-1.0, 1.0))

            # ---- blue constraint margin overlay ----
            tighten_val = vcfg.get("tighten", 0.0)
            constraint_handles = []
            if vcfg["use_projection"]:
                if args.dynamic_obstacles:
                    _dyn_set_plot = set(args.dynamic_cyl_indices) if args.dynamic_cyl_indices is not None \
                                    else set(range(len(CYLINDERS)))
                    _static_c = [CYLINDERS[i] for i in range(len(CYLINDERS)) if i not in _dyn_set_plot]
                    _dyn_c    = [CYLINDERS[i] for i in range(len(CYLINDERS)) if i in     _dyn_set_plot]
                    # dynamic cylinders: no blue circle — orange swept band already shows range
                    constraint_handles = plot_constraint_overlay(
                        ax, BOXES, _static_c,
                        tighten=tighten_val, drone_radius=args.drone_radius,
                        x_bounds=(-0.5, 4.5), y_bounds=(-1.0, 1.0),
                    )
                else:
                    constraint_handles = plot_constraint_overlay(
                        ax, BOXES, CYLINDERS,
                        tighten=tighten_val,
                        drone_radius=args.drone_radius,
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
            ax.axis("equal")
            ax.grid(True, alpha=0.3)
            ax.legend()

            # merge legend handles
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles=handles + constraint_handles,
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


if __name__ == "__main__":
    main()
