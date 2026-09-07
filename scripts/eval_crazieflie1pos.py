import importlib
import os
from collections import deque
from pathlib import Path
import time
import numpy as np
np.set_printoptions(precision=3, suppress=True)
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

import diffuser.utils as utils
import diffuser.sampling.projection as projection_mod
from diffuser.sampling.projection import Projector
from diffuser.sampling.policies import temporal_consistency_distances
projection_mod.DEBUG_SLSQP = False
from metrics_logger import MetricsLogger
import depth_obstacle_estimator as detect_mod
from depth_obstacle_estimator import camera_world_pose, quat_apply, detect_obstacles_umap
detect_mod.DEBUG_DETECT = False

cfg = importlib.import_module("config.avoiding-crazyflie")
CYLINDERS = cfg.CYLINDERS
KEEPOUT_ZONES = getattr(cfg, 'KEEPOUT_ZONES', [])   # (x, y, radius) world-frame, planner-only  see config/avoiding-crazyflie.py

CYL_RADIUS = 0.20
RUN_DIR = "isaac/logs/avoiding-crazyflie/diffusion/H8_K20_Dmodels.ImagePoseCondUNet1DTemporalCondModel_Evitp_L384"
# "isaac/logs/avoiding-crazyflie/diffusion/H8_K20_Dmodels.ImagePoseCondTransformer1DModel_Eraw_pixels_L27648"
# "isaac/logs/avoiding-crazyflie/diffusion/H8_K20_Dmodels.ImagePoseCondUNet1DTemporalCondModel_Evitp_L384"
SEEDS = [7]
MAX_STEPS = 700
TARGET_X = 3.00
TARGET_Y = -1.50
TARGET_Z = 1.5

RANDOMIZE_SPAWN_TARGET = True
SPAWN_X_RANGE = (-5.5, -4.5)
SPAWN_Y_RANGE = (-0.5, 0.5)
SPAWN_Z = 0.75
TARGET_X_RANGE = (3.5, 4.0)
TARGET_Y_RANGE = (-1.0, 1.0)

# ── Obstacle-aware projection (in-loop SLSQP) ────────────────────────────────────
VARIANTS = ["sdpc-r", "sdpc-c", "sdpc-t", 
            "diffuser"]*5
VARIANT_CFG = {
    "sdpc-r": dict(num_candidates=1, selection="first", use_projection=True),
    "sdpc-c": dict(num_candidates=2, selection="minimum_projection_cost", use_projection=True),
    "sdpc-t": dict(num_candidates=2, selection="temporal_consistency", use_projection=True),
    "diffuser": dict(num_candidates=1, selection="first", use_projection=False),
}
OBSTACLE_SOURCE = "depth"  # "ground_truth" = env.get_cylinder_positions() "depth" = the camera-based detect_depth_obstacles() path below.
CYL_PHYS_RADIUS = 0.2
DEPTH_OBSTACLE_RADIUS = 0.3
MAX_DEPTH_OBSTACLES = 5
UMAP_MAX_RANGE = 3.0
UMAP_BIN_SIZE = 100
UMAP_T_POI = 500.0
UMAP_T_THO = 1800.0
UMAP_MIN_PIXEL_COUNT = 50
PROJ_TIGHTEN = 0.15
PROJ_DT = 0.1
FLIGHT_Z_MIN = 0.02
FLIGHT_Z_MAX = 1.5
_Z_HALFSPACES = [
    ([0.0, 0.0, 1.0], FLIGHT_Z_MAX),    # z <= FLIGHT_Z_MAX
    ([0.0, 0.0, -1.0], FLIGHT_Z_MIN),   # z >= FLIGHT_Z_MIN
]


def preprocess_rgb_stack(rgb_hist):
    """rgb_hist: list/deque of To frames, each (H,W,3) uint8.
    Returns torch tensor (1, To, 3, H, W) in [0,1] -- the model's only input, "obs_rgb"
    (matches CrazyflieImageDataset.__getitem__)."""
    arr = np.stack(rgb_hist, axis=0).astype(np.float32) / 255.0  # (To,H,W,3)
    arr = np.transpose(arr, (0, 3, 1, 2))  # (To,3,H,W)
    return torch.from_numpy(arr).unsqueeze(0)  # (1,To,3,H,W)


def sample_action(diffusion, cond, horizon, action_dim):
    """Single unconditioned (no projector) diffusion sample.
    Returns a0_norm: (action_dim,) numpy -- just the first predicted step,
    receding-horizon style (resampled fresh every control step)."""
    with torch.no_grad():
        x, _ = diffusion.conditional_sample(cond, horizon=horizon, projector=None)  # (1,H,D)
    return x[0, 0, :action_dim].detach().cpu().numpy()


def sample_action_horizon(diffusion, cond, horizon, action_dim, projector=None, num_candidates=1):
    """Same as sample_action() but returns the FULL predicted horizon for K candidate
    samples, not just step 0 of one, and accepts an in-loop projector: passed straight
    through to conditional_sample() so the SLSQP correction happens progressively
    during denoising (once past projector.diffusion_timestep_threshold), same as
    eval_crazieflie1.py's "sdpc" variants -- not a post-hoc correction after the fact.
    Returns (K, H, action_dim); K=1 (default) is the sdpc-r case, just batch-of-one."""
    obs_rgb = cond["obs_rgb"]  # (1,To,3or4,H,W)
    cond_k = {"obs_rgb": obs_rgb.repeat(num_candidates, 1, 1, 1, 1)}
    if "goal_rel" in cond:
        cond_k["goal_rel"] = cond["goal_rel"].repeat(num_candidates, 1)
    with torch.no_grad():
        x, _ = diffusion.conditional_sample(cond_k, horizon=horizon, projector=projector)  # (K,H,D)
    return x[:, :, :action_dim].detach().cpu().numpy()  # (K, H, action_dim)


def get_ground_truth_obstacles(env):
    return [(float(x), float(y), CYL_PHYS_RADIUS) for (x, y) in env.get_cylinder_positions()]


def detect_depth_obstacles(env, depth_fx, depth_fy, depth_cx, depth_cy):
    """Every obstacle detect_obstacles_umap() finds in the current depth frame
    (U-disparity-map + contour method -- see depth_obstacle_estimator.py's module
    docstring), up to MAX_DEPTH_OBSTACLES, world-frame. Returns [(x, y, radius), ...]."""
    depth = env.get_depth()
    depth_2d = depth[..., 0] if depth.ndim == 3 else depth
    root = env.robot.data.root_state_w[0].detach().cpu().numpy()
    pos_body_w, quat_body_w = root[0:3], root[3:7]
    pos_cam_w, quat_cam_w = camera_world_pose(pos_body_w, quat_body_w)

    detections = detect_obstacles_umap(
        depth_2d, depth_fx, depth_fy, depth_cx, depth_cy,
        max_range_m=UMAP_MAX_RANGE, bin_size=UMAP_BIN_SIZE,
        t_poi=UMAP_T_POI, t_tho=UMAP_T_THO, min_pixel_count=UMAP_MIN_PIXEL_COUNT,
        max_obstacles=MAX_DEPTH_OBSTACLES,
    )
    points = []
    for pos_cam, half_w, _half_h in detections:
        world_xyz = pos_cam_w + quat_apply(quat_cam_w, pos_cam)
        radius = max(DEPTH_OBSTACLE_RADIUS, half_w)
        points.append((float(world_xyz[0]), float(world_xyz[1]), float(radius)))
    return points


def build_projector(horizon_H, device, static_points, drone_radius=0.0,
                     pos0=None, action_normalizer=None, keepout_zones=None):
    lb = np.array([-6.5, -1.95, FLIGHT_Z_MIN], dtype=np.float32)
    ub = np.array([4.5, 1.95, FLIGHT_Z_MAX], dtype=np.float32)
    constraint_list = [("lb", lb), ("ub", ub)]

    for (x, y, r) in static_points:
        radius = 0.2 + drone_radius + PROJ_TIGHTEN
        constraint_list.append(("sphere_outside", [0, 1], [float(x), float(y)], float(radius)))
    for (x, y, zone_radius) in (keepout_zones or []):
        radius = float(zone_radius) + drone_radius + PROJ_TIGHTEN
        constraint_list.append(("sphere_outside", [0, 1], [float(x), float(y)], radius))
    for normal, rhs in _Z_HALFSPACES:
        constraint_list.append(("ineq", (np.array(normal, dtype=np.float32), float(rhs))))

    projector = Projector(
        horizon=horizon_H + 1, transition_dim=3, action_dim=0, goal_dim=0,
        constraint_list=constraint_list, normalizer=None, gradient=False,
        gradient_weights=[1, 0.5, 2], dt=PROJ_DT, variant="states",
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


def add_obstacles_xy(ax, cylinders, cyl_radius=CYL_RADIUS):
    for x, y in cylinders:
        ax.add_patch(Circle((x, y), cyl_radius, linewidth=1.0,
                             edgecolor="black", facecolor="tab:orange", alpha=0.30))


def main():
    run_start_time = time.time()
    if RUN_DIR is None:
        raise ValueError("Set RUN_DIR at the top of this file before running.")

    device = torch.device("cuda:0")

    drone_radius = 0.15

    run_dirs = [os.path.join(RUN_DIR, str(s)) for s in SEEDS] if SEEDS else [RUN_DIR]
    env = None

    for run_dir in run_dirs:
        print(f"\n[INFO] Loading run dir: {run_dir}")
        seedmodel = int(Path(run_dir).name)
        diff_exp = utils.load_diffusion(run_dir, epoch="best", device=str(device))
        dataset = diff_exp.dataset
        diffusion = diff_exp.diffusion.to(device)
        diffusion.eval()

        cfg.USE_DEPTH = any(VARIANT_CFG[v]["use_projection"] for v in VARIANTS)

        from isaac.scripts.crazyflie_envpos import Crazyflie, CrazyflieEnvCfg

        use_pose_cond = bool(getattr(dataset, "use_pose_cond", False))
        pose_target_world = np.array([TARGET_X, TARGET_Y, TARGET_Z], dtype=np.float32)

        env_cfg = CrazyflieEnvCfg(
            num_envs=1, device=str(device),
            drone_radius=drone_radius,
            goal_pos=tuple(pose_target_world.tolist()),
        )
        env = Crazyflie(env_cfg)
        env.reset()
        _pos_warmup = env._pos_world().detach().cpu().numpy()[0]
        env.step(_pos_warmup[None, :3])
        _K = env.cam.data.intrinsic_matrices[0].detach().cpu().numpy()
        depth_fx, depth_fy, depth_cx, depth_cy = float(_K[0, 0]), float(_K[1, 1]), float(_K[0, 2]), float(_K[1, 2])
        print(f"[INFO] Real camera intrinsics (from IsaacSim camera): "
              f"fx={depth_fx:.2f} fy={depth_fy:.2f} cx={depth_cx:.2f} cy={depth_cy:.2f}")

        run_name = Path(run_dir).parent.name
        horizon = int(getattr(diffusion, "horizon", 16))
        action_dim = int(getattr(diffusion, "action_dim", 3))
        To = int(getattr(dataset, "n_obs_steps", 2))
        print(f"[INFO] Online eval started. run={run_name} To={To} H={horizon} action_dim={action_dim}")
        print(f"[INFO] Env goal (success radius {env_cfg.success_radius}m) = {pose_target_world.tolist()}")

        logger = MetricsLogger(
            save_dir=os.path.join(run_dir, "results"),
            encoder_type=getattr(diffusion.model, "encoder_type", "unknown"),
            latent_dim=int(getattr(diffusion.model, "image_cond_dim", 0)),
            horizon=horizon, n_diffusion_steps=20,
            corridor_halfspaces=[], cylinders=CYLINDERS,
        )

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
        print(f"[INFO] trajectories -> {traj_dir}")
        print(f"[INFO] plots        -> {plot_dir}")

        # ------------------ Episodes (one per variant) ------------------
        for ep, variant_name in enumerate(VARIANTS):
            num_candidates = VARIANT_CFG[variant_name]["num_candidates"]
            selection_strategy = VARIANT_CFG[variant_name]["selection"]
            use_projection = VARIANT_CFG[variant_name]["use_projection"]
            print(f"\n[INFO] ===== Episode {ep + 1}/{len(VARIANTS)}: {variant_name} "
                  f"(num_candidates={num_candidates}, selection={selection_strategy}, "
                  f"use_projection={use_projection}) =====")
            episode_start_time = time.time()

            spawn_pos = None
            if RANDOMIZE_SPAWN_TARGET:
                rng = np.random.default_rng(ep)
                spawn_pos = (float(rng.uniform(*SPAWN_X_RANGE)), float(rng.uniform(*SPAWN_Y_RANGE)), SPAWN_Z)
                pose_target_world = np.array([
                    rng.uniform(*TARGET_X_RANGE), rng.uniform(*TARGET_Y_RANGE), TARGET_Z,
                ], dtype=np.float32)
                env.goal_pos = torch.tensor(pose_target_world, dtype=torch.float32, device=device)
                print(f"[INFO] Randomized spawn={spawn_pos} target={pose_target_world.tolist()}")

            print(f"[TARGET] ep={ep} pos={pose_target_world.tolist()}")
            _ = env.reset(seed=ep, pos=spawn_pos)
            logger.begin_episode(variant_name, episode=ep, seed=ep)
            pos0 = env._pos_world().detach().cpu().numpy()[0]
            hold_action = np.tile(pos0[:action_dim].astype(np.float32), (env.num_envs, 1))
            for _ in range(3):
                try:
                    env.step(hold_action)
                except Exception:
                    pass

            rgb0 = env.get_rgb()
            rgb_hist = deque(maxlen=To)
            for _ in range(To):
                rgb_hist.append(rgb0.copy())

            traj_xyz = []
            actions_taken = []
            inference_times = []  # per-step diffusion sampling wall time (seconds)
            prev_actions_real = None  # previous step's executed (H,3) chunk, for
                                       # SELECTION_STRATEGY="temporal_consistency"
            depth_obstacle_accum = {}  # {(vx,vy): (x,y)} -- every depth detection this
                                        # episode, deduped by voxel cell, for the XY plot

            pos_init = env._pos_world().detach().cpu().numpy()[0]
            traj_xyz.append(pos_init.copy())

            for step in range(MAX_STEPS):
                pos = env._pos_world().detach().cpu().numpy()[0]
                _elapsed = time.strftime('%M:%S', time.gmtime(time.time() - episode_start_time))
                obs_rgb_t = preprocess_rgb_stack(rgb_hist).to(device)
                cond = {"obs_rgb": obs_rgb_t}
                goal_rel = (pose_target_world - pos[:3]).astype(np.float32)
                cond["goal_rel"] = torch.from_numpy(goal_rel).float().unsqueeze(0).to(device)

                if use_projection:
                    if OBSTACLE_SOURCE == "ground_truth":
                        static_pts = get_ground_truth_obstacles(env)
                    else:
                        static_pts = detect_depth_obstacles(env, depth_fx, depth_fy, depth_cx, depth_cy)
                    print(f"[OBSTACLES] source={OBSTACLE_SOURCE} static_pts={static_pts}")
                    for (x, y, _r) in static_pts:
                        depth_obstacle_accum[(round(x / 0.05), round(y / 0.05))] = (x, y)
                    projector = build_projector(horizon, device, static_pts, drone_radius,
                                                 pos0=pos[:3], action_normalizer=dataset.action_normalizer,
                                                 keepout_zones=KEEPOUT_ZONES)

                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    _infer_start = time.time()
                    a_horizon_norm = sample_action_horizon(diffusion, cond, horizon, action_dim,
                                                            projector=projector, num_candidates=num_candidates)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    inference_time = time.time() - _infer_start

                    a_horizon_real = dataset.action_normalizer.unnormalize(a_horizon_norm)  # (K,H,D)
                    proj_costs = None
                    if projector is not None:
                        a_horizon_real[:, :, :3], proj_costs = project_deltas_from_pos(
                            projector, pos[:3], a_horizon_real[:, :, :3], device
                        )
                    choice = choose_candidate(a_horizon_real, proj_costs, prev_actions_real, selection_strategy)
                    a0_real = a_horizon_real[choice, 0]
                    prev_actions_real = a_horizon_real[choice]
                else:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    _infer_start = time.time()
                    a0_norm = sample_action(diffusion, cond, horizon, action_dim)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    inference_time = time.time() - _infer_start
                    a0_real = dataset.action_normalizer.unnormalize(a0_norm)

                inference_times.append(inference_time)
                print(f"[INFERENCE] step={step} time={inference_time * 1000:.1f}ms")
                print(f"[MODEL OUTPUT] a0_real={a0_real}")
                cmd_xyz = pos.copy()
                cmd_xyz[:action_dim] = pos + a0_real[:action_dim]
                # cmd_xyz[2] = np.clip(cmd_xyz[2], FLIGHT_Z_MIN, FLIGHT_Z_MAX)
                obs_next, _rew, done_vec, info = env.step(cmd_xyz[None, :])  # (action_dim,) -> (1, action_dim)

                rgb = env.get_rgb()
                rgb_hist.append(rgb)

                pos2 = obs_next[0]
                traj_xyz.append(pos2.copy())
                actions_taken.append(a0_real.copy())
                logger.step(pos=pos, action=a0_real)

                done = bool(done_vec[0]) if isinstance(done_vec, (list, tuple, np.ndarray, torch.Tensor)) else bool(done_vec)
                print(f"{_elapsed} step {step:04d} pos={pos2} done={done}")

                if done:
                    print("[INFO] Done=True. Breaking episode loop.")
                    break

            episode_wall_time_sec = time.time() - episode_start_time
            print(f"[INFO] Episode (seed {seedmodel}) took "
                  f"{episode_wall_time_sec / 60.0:.2f} minutes ({episode_wall_time_sec:.1f}s)")

            inference_times_arr = np.array(inference_times, dtype=np.float32)
            if len(inference_times_arr) > 0:
                print(f"[INFO] Inference time: mean={inference_times_arr.mean() * 1000:.1f}ms "
                      f"min={inference_times_arr.min() * 1000:.1f}ms max={inference_times_arr.max() * 1000:.1f}ms "
                      f"total={inference_times_arr.sum():.2f}s over {len(inference_times_arr)} steps")

            success = bool(info["success"][0])
            fell = bool(info["fell"][0])
            logger.end_episode(success=success, fell=fell, wall_time_sec=episode_wall_time_sec)

            traj_path = os.path.join(traj_dir, f"traj_pos_{variant_name}_seed{seedmodel}_ep{ep}.npz")
            np.savez(
                traj_path,
                xyz=np.array(traj_xyz),
                actions=np.array(actions_taken),
                success=success, fell=fell, episode=ep,
                episode_wall_time_sec=float(episode_wall_time_sec),
                inference_times=inference_times_arr,
                cylinders=np.array(CYLINDERS),
            )
            print(f"[TRAJ] saved: {traj_path}")

            # ------------------ Plot episode ------------------
            if len(traj_xyz) > 0:
                traj_xyz_np = np.stack(traj_xyz, axis=0).astype(np.float32)

                plt.figure(figsize=(7, 4))
                plt.plot(traj_xyz_np[:, 2])
                plt.ylim(-0.05, 2.15)
                plt.axhline(0.0, color="#888888", linewidth=0.8, linestyle=":")
                plt.axhline(2.0, color="#888888", linewidth=0.8, linestyle=":")
                plt.xlabel("timestep")
                plt.ylabel("z  (m)")
                plt.title(f"Z over time ({variant_name}, seed {seedmodel}, ep {ep})")
                plt.tight_layout()
                z_path = os.path.join(plot_dir, f"{variant_name}_z_ep{ep}.pdf")
                plt.savefig(z_path)
                plt.close()
                print(f"[PLOT] saved: {z_path}")

                xy_exec = traj_xyz_np[:, :2]
                fig, ax = plt.subplots(figsize=(8, 7))
                ax.plot(xy_exec[:, 0], xy_exec[:, 1], linewidth=2.5, marker="o", markersize=3, label="executed")
                ax.scatter(pos_init[0], pos_init[1], marker="o", s=70, color="green", zorder=5, label="start")
                ax.scatter(xy_exec[-1, 0], xy_exec[-1, 1], marker="x", s=60, label="end")

                add_obstacles_xy(ax, CYLINDERS, cyl_radius=CYL_RADIUS)

                if use_projection and depth_obstacle_accum:
                    det_xy = np.array(list(depth_obstacle_accum.values()))
                    ax.scatter(det_xy[:, 0], det_xy[:, 1], s=15, color="crimson", alpha=0.5,
                               zorder=3, label="depth-detected obstacle")

                # Keep-out zones overlay (virtual, planner-only  see KEEPOUT_ZONES
                # in config/avoiding-crazyflie.py)
                for (kx, ky, kr) in KEEPOUT_ZONES:
                    ax.add_patch(Circle(
                        (kx, ky), kr,
                        linewidth=1.2, edgecolor="crimson", facecolor="crimson",
                        alpha=0.15, linestyle="--", zorder=2,
                    ))

                ax.set_xlabel("x")
                ax.set_ylabel("y")
                ax.set_title(f"XY trajectory ({variant_name}, seed {seedmodel}, ep {ep})")
                ax.set_aspect("equal", adjustable="box")
                ax.set_xlim(-6.5, 4.5)
                ax.set_ylim(-2.25, 2.25)
                ax.grid(True, alpha=0.3)
                ax.legend(loc="upper left", fontsize=8)

                out_path = os.path.join(plot_dir, f"{variant_name}_xy_ep{ep}.pdf")
                fig.tight_layout()
                fig.savefig(out_path)
                plt.close(fig)
                print("[PLOT] saved:", out_path)

            env.reset()
        print(f"\n[INFO] ===== Aggregate over {', '.join(VARIANTS)} =====")
        logger.print_live_summary()
        logger.save()

    total_wall_time_sec = time.time() - run_start_time
    print(f"[INFO] Total eval run time: {total_wall_time_sec / 60.0:.2f} minutes "
          f"({total_wall_time_sec:.1f}s)")

    env.close()
    os._exit(0)


if __name__ == "__main__":
    main()
