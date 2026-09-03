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
from matplotlib.patches import Circle, Rectangle, Polygon

import diffuser.utils as utils
from metrics_logger import MetricsLogger

cfg = importlib.import_module("config.avoiding-crazyflie")
CYLINDERS = cfg.CYLINDERS
DEPTH_NEAR = cfg.DEPTH_NEAR
DEPTH_FAR = cfg.DEPTH_FAR

CYL_RADIUS = 0.30

# ── Run configuration -- edit these directly instead of passing CLI flags ──────
RUN_DIR = "isaac/logs/avoiding-crazyflie/diffusion/H8_K20_Dmodels.ImagePoseCondUNet1DTemporalCondModel_Evitp_L384_DEPTHFalse"
SEEDS = [7]
ACTION_SCALE = 1.0
NUM_EPISODES = 1  # episodes per seed
DYNAMIC_OBSTACLES = None  # None = disabled. [] = move ALL cylinders laterally (axis
                          # 'y'). Or 'idx:axis' tokens (axis 'x'/'y'/'xy', ':axis'
                          # optional, defaults to 'y'), e.g. ["0:y", "2:x", "4:xy"].
SAVE_FRAMES = False
MAX_STEPS = 700
TARGET_Y = -1.00
POSE_TARGET_SANITY_HOLD = False  # debug: if True, pose_target is overwritten every step
                                  # with the drone's OWN current position instead of the
                                  # fixed goal below. If pose conditioning actually steers
                                  # the policy, this should make it roughly hold still
                                  # (near-zero velocity) instead of flying its usual path --
                                  # isolates whether pose_target has any effect at all on
                                  # this checkpoint.


def get_rgb_from_env(env):
    """Fetch RGB frame only from env.get_rgb(). Returns uint8 (H,W,3)."""
    if not hasattr(env, "get_rgb"):
        raise RuntimeError("Env does not have get_rgb() method.")
    return env.get_rgb()


def get_obs_frame_from_env(env, use_depth):
    """One observation frame for the policy's history buffer.
    Returns rgb: (H,W,3) uint8, or (rgb, depth) if use_depth=True."""
    rgb = get_rgb_from_env(env)
    if not use_depth:
        return rgb
    if not hasattr(env, "get_depth"):
        raise RuntimeError("use_depth=True but env does not have get_depth() method.")
    return rgb, env.get_depth()


def preprocess_rgb_stack(rgb_hist):
    """rgb_hist: list/deque of To frames, each (H,W,3) uint8.
    Returns torch tensor (1, To, 3, H, W) in [0,1]."""
    arr = np.stack(rgb_hist, axis=0).astype(np.float32) / 255.0  # (To,H,W,3)
    arr = np.transpose(arr, (0, 3, 1, 2))  # (To,3,H,W)
    return torch.from_numpy(arr).unsqueeze(0)  # (1,To,3,H,W)


def preprocess_obs_stack(obs_hist, use_depth):
    """Matches CrazyflieImageDataset.__getitem__'s "obs_rgb" (same
    DEPTH_NEAR/DEPTH_FAR clip-and-minmax normalization)."""
    if not use_depth:
        return preprocess_rgb_stack(obs_hist)

    rgb_ten = preprocess_rgb_stack([frame[0] for frame in obs_hist])
    depth = np.stack([frame[1] for frame in obs_hist], axis=0)  # (To,H,W,1)
    depth = np.squeeze(depth, axis=-1)
    non_finite = ~np.isfinite(depth)
    if non_finite.any():
        depth = depth.copy()
        depth[non_finite] = DEPTH_FAR
    depth = np.clip(depth, DEPTH_NEAR, DEPTH_FAR)
    depth = (depth - DEPTH_NEAR) / (DEPTH_FAR - DEPTH_NEAR)
    depth_ten = torch.from_numpy(depth[None, :, None, :, :].astype(np.float32))  # (1,To,1,H,W)
    return torch.cat([rgb_ten, depth_ten], dim=2)  # (1,To,4,H,W)


def sample_action(diffusion, cond, horizon, action_dim):
    """Single unconditioned (no projector) diffusion sample.
    Returns a0_norm: (action_dim,) numpy -- just the first predicted step,
    receding-horizon style (resampled fresh every control step)."""
    with torch.no_grad():
        x, _ = diffusion.conditional_sample(cond, horizon=horizon, projector=None)  # (1,H,D)
    return x[0, 0, :action_dim].detach().cpu().numpy()


def add_obstacles_xy(ax, cylinders, cyl_radius=CYL_RADIUS):
    for x, y in cylinders:
        ax.add_patch(Circle((x, y), cyl_radius, linewidth=1.0,
                             edgecolor="black", facecolor="tab:orange", alpha=0.30))


def add_dynamic_cylinders_xy(ax, cyl_rest, cyl_axes, cand_snapshots, obs_amplitude,
                              cyl_radius=CYL_RADIUS, drone_radius=0.0,
                              x_clamp=(0.3, 4.7), y_clamp=(-0.85, 0.85)):
    """orange band = oscillation sweep, dashed circle = exclusion zone at rest,
    faded dots = physical cylinder position sampled every few steps."""
    excl_r = cyl_radius + drone_radius
    for (x0, y0), axis in zip(cyl_rest, cyl_axes):
        if axis == "y":
            ylo, yhi = max(y_clamp[0], y0 - obs_amplitude), min(y_clamp[1], y0 + obs_amplitude)
            band = Rectangle((x0 - excl_r, ylo), 2 * excl_r, yhi - ylo, facecolor="tab:orange", alpha=0.15, zorder=1)
        elif axis == "x":
            xlo, xhi = max(x_clamp[0], x0 - obs_amplitude), min(x_clamp[1], x0 + obs_amplitude)
            band = Rectangle((xlo, y0 - excl_r), xhi - xlo, 2 * excl_r, facecolor="tab:orange", alpha=0.15, zorder=1)
        else:  # "xy"
            x_lo, x_hi = max(x_clamp[0], x0 - obs_amplitude), min(x_clamp[1], x0 + obs_amplitude)
            y_lo, y_hi = max(y_clamp[0], y0 - obs_amplitude), min(y_clamp[1], y0 + obs_amplitude)
            p1, p2 = np.array([x_lo, y_lo]), np.array([x_hi, y_hi])
            seg = p2 - p1
            d_hat = seg / (np.linalg.norm(seg) + 1e-8)
            n_hat = np.array([-d_hat[1], d_hat[0]])
            corners = np.array([p1 - excl_r * n_hat, p1 + excl_r * n_hat, p2 + excl_r * n_hat, p2 - excl_r * n_hat])
            band = Polygon(corners, facecolor="tab:orange", alpha=0.15, zorder=1)
        ax.add_patch(band)
        ax.add_patch(Circle((x0, y0), excl_r, linewidth=1.2, linestyle="--",
                             edgecolor="tab:orange", facecolor="none", alpha=0.5, zorder=2))

    stride = max(1, len(cand_snapshots) // 30)
    for snap in cand_snapshots[::stride]:
        if snap.get("cyl_xy") is None:
            continue
        for (cx, cy) in snap["cyl_xy"]:
            ax.add_patch(Circle((cx, cy), cyl_radius, facecolor="tab:orange", alpha=0.10, zorder=2))


def main():
    run_start_time = time.time()
    if RUN_DIR is None:
        raise ValueError("Set RUN_DIR at the top of this file before running.")

    device = torch.device("cuda:0")

    # ── parse DYNAMIC_OBSTACLES tokens (same convention as eval_crazieflie1.py) ──
    if DYNAMIC_OBSTACLES is None:
        dynamic_obstacles_enabled = False
        dynamic_cyl_indices = None
        obs_axes = None
    else:
        dynamic_obstacles_enabled = True
        if len(DYNAMIC_OBSTACLES) == 0:
            dynamic_cyl_indices = None
            obs_axes = None
        elif len(DYNAMIC_OBSTACLES) == 1 and DYNAMIC_OBSTACLES[0] in ("x", "y", "xy"):
            dynamic_cyl_indices = None
            obs_axes = [DYNAMIC_OBSTACLES[0]]
        else:
            indices, axes = [], []
            for tok in DYNAMIC_OBSTACLES:
                idx_str, _, axis = tok.partition(":")
                axis = axis or "y"
                if axis not in ("x", "y", "xy"):
                    raise ValueError(f"DYNAMIC_OBSTACLES: invalid axis '{axis}' in '{tok}' (use x, y, or xy)")
                indices.append(int(idx_str))
                axes.append(axis)
            dynamic_cyl_indices = indices
            obs_axes = axes
    obs_amplitude = 0.35
    obs_frequency = 0.25
    drone_radius = 0.1

    run_dirs = [os.path.join(RUN_DIR, str(s)) for s in SEEDS] if SEEDS else [RUN_DIR]
    env = None
    shared_use_depth = None

    for run_dir in run_dirs:
        print(f"\n[INFO] Loading run dir: {run_dir}")
        seedmodel = int(Path(run_dir).name)
        diff_exp = utils.load_diffusion(run_dir, epoch="best", device=str(device))
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
        cfg.USE_DEPTH = use_depth

        from isaac.scripts.crazyflie_envvel import Crazyflie, CrazyflieEnvCfg

        if env is None:
            env_cfg = CrazyflieEnvCfg(
                num_envs=1, device=str(device),
                dynamic_obstacles=dynamic_obstacles_enabled,
                obs_amplitude=obs_amplitude, obs_frequency=obs_frequency,
                dynamic_cyl_indices=dynamic_cyl_indices, obs_axes=obs_axes,
                drone_radius=drone_radius,
            )
            env = Crazyflie(env_cfg)
            shared_use_depth = use_depth
        elif use_depth != shared_use_depth:
            raise RuntimeError(
                f"run_dir {run_dir}'s checkpoint has use_depth={use_depth}, but that doesn't match "
                f"the already-running sim (use_depth={shared_use_depth}). Run this seed separately."
            )

        run_name = Path(run_dir).parent.name
        horizon = int(getattr(diffusion, "horizon", 16))
        action_dim = int(getattr(diffusion, "action_dim", 3))
        To = int(getattr(dataset, "n_obs_steps", 2))
        print(f"[INFO] Online eval started. run={run_name} To={To} H={horizon} action_dim={action_dim}")

        use_pose_cond = bool(getattr(dataset, "use_pose_cond", False))
        pose_target_world = None
        if use_pose_cond:
            pose_target_world = np.array([4.0, TARGET_Y, 1.5], dtype=np.float32)
            if POSE_TARGET_SANITY_HOLD:
                print("[INFO] POSE_TARGET_SANITY_HOLD=True: pose_target will track current pos every step (goal below unused).")
            else:
                print(f"[INFO] Pose-conditioned model: fixed goal for this run = {pose_target_world.tolist()}")

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

        # ------------------ Episodes ------------------
        for ep in range(NUM_EPISODES):
            print(f"\n[INFO] ===== Episode {ep + 1}/{NUM_EPISODES} =====")
            episode_start_time = time.time()
            _ = env.reset(seed=ep)
            logger.begin_episode("vxz_yawrate", episode=ep, seed=ep)

            # Camera warm-up: hold zero velocity command (important for Isaac/Replicator)
            # crazyflie_envvel.Crazyflie.step() indexes action as act[:, 0] etc, so it
            # needs a (num_envs, action_dim) array, not a bare (action_dim,) vector.
            for _ in range(3):
                try:
                    env.step(np.zeros((env.num_envs, action_dim), dtype=np.float32))
                except Exception:
                    pass

            rgb0 = get_obs_frame_from_env(env, use_depth)
            rgb_hist = deque(maxlen=To)
            for _ in range(To):
                rgb_hist.append((rgb0[0].copy(), rgb0[1].copy()) if use_depth else rgb0.copy())

            traj_xyz = []
            actions_taken = []
            frames_taken = []
            cand_snapshots = []  # kept only for the dynamic-cylinder plot overlay

            pos_init = env._pos_world().detach().cpu().numpy()[0]
            traj_xyz.append(pos_init.copy())
            if SAVE_FRAMES:
                frames_taken.append(rgb0[0].copy() if use_depth else rgb0.copy())

            for step in range(MAX_STEPS):
                pos = env._pos_world().detach().cpu().numpy()[0]
                _elapsed = time.strftime('%M:%S', time.gmtime(time.time() - episode_start_time))

                obs_rgb_t = preprocess_obs_stack(rgb_hist, use_depth).to(device)
                cond = {"obs_rgb": obs_rgb_t}
                if use_pose_cond:
                    pose_now_norm = dataset.pose_normalizer.normalize(pos[:3].astype(np.float32))
                    step_target = pos[:3].astype(np.float32) if POSE_TARGET_SANITY_HOLD else pose_target_world
                    pose_target_norm = dataset.pose_normalizer.normalize(step_target)
                    cond["pose_now"] = torch.from_numpy(pose_now_norm).float().unsqueeze(0).to(device)
                    cond["pose_target"] = torch.from_numpy(pose_target_norm).float().unsqueeze(0).to(device)

                a0_norm = sample_action(diffusion, cond, horizon, action_dim)
                a0_real = dataset.action_normalizer.unnormalize(a0_norm) * float(ACTION_SCALE)  # [vx,vz,yaw_rate]

                print(f"[MODEL OUTPUT] a0_real={a0_real}")
                obs_next, _rew, done_vec, info = env.step(a0_real[None, :])  # (action_dim,) -> (1, action_dim)

                rgb = get_obs_frame_from_env(env, use_depth)
                rgb_hist.append(rgb)
                if SAVE_FRAMES:
                    frames_taken.append(rgb[0].copy() if use_depth else rgb.copy())

                pos2 = obs_next[0]
                traj_xyz.append(pos2.copy())
                actions_taken.append(a0_real.copy())
                logger.step(pos=pos, action=a0_real)

                done = bool(done_vec[0]) if isinstance(done_vec, (list, tuple, np.ndarray, torch.Tensor)) else bool(done_vec)
                print(f"{_elapsed} step {step:04d} pos={pos2} done={done}")
                cand_snapshots.append({
                    "pos": pos2.copy(),
                    "cyl_xy": env.get_cylinder_positions() if dynamic_obstacles_enabled else None,
                })

                if done:
                    print("[INFO] Done=True. Breaking episode loop.")
                    break

            episode_wall_time_sec = time.time() - episode_start_time
            print(f"[INFO] Episode (seed {seedmodel}) took "
                  f"{episode_wall_time_sec / 60.0:.2f} minutes ({episode_wall_time_sec:.1f}s)")

            success = bool(info["success"][0])
            fell = bool(info["fell"][0])
            logger.end_episode(success=success, fell=fell, wall_time_sec=episode_wall_time_sec)
            if (ep + 1) % 5 == 0:
                logger.print_live_summary()

            traj_path = os.path.join(traj_dir, f"traj_vxz_yawrate_seed{seedmodel}_ep{ep}.npz")
            if SAVE_FRAMES:
                frames_path = os.path.join(traj_dir, f"frames_vxz_yawrate_seed{seedmodel}_ep{ep}.npy")
                np.save(frames_path, np.array(frames_taken, dtype=np.uint8))
                print(f"[INFO] Saved {len(frames_taken)} frames -> {frames_path}")
            np.savez(
                traj_path,
                xyz=np.array(traj_xyz),
                actions=np.array(actions_taken),
                success=success, fell=fell, episode=ep,
                episode_wall_time_sec=float(episode_wall_time_sec),
                cylinders=np.array(CYLINDERS),
                dynamic_obstacles=bool(dynamic_obstacles_enabled),
                obs_amplitude=float(obs_amplitude), obs_frequency=float(obs_frequency),
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
                plt.title(f"Z over time (seed {seedmodel}, ep {ep})")
                plt.tight_layout()
                z_path = os.path.join(plot_dir, f"z_ep{ep}.pdf")
                plt.savefig(z_path)
                plt.close()
                print(f"[PLOT] saved: {z_path}")

                xy_exec = traj_xyz_np[:, :2]
                fig, ax = plt.subplots(figsize=(8, 7))
                ax.plot(xy_exec[:, 0], xy_exec[:, 1], linewidth=2.5, label="executed")
                ax.scatter(pos_init[0], pos_init[1], marker="o", s=70, color="green", zorder=5, label="start")
                ax.scatter(xy_exec[-1, 0], xy_exec[-1, 1], marker="x", s=60, label="end")

                if dynamic_obstacles_enabled:
                    dyn_idx = dynamic_cyl_indices if dynamic_cyl_indices is not None else list(range(len(CYLINDERS)))
                    _raw_axes = obs_axes if obs_axes is not None else ["y"]
                    dyn_axes_resolved = (_raw_axes * len(dyn_idx))[:len(dyn_idx)] if len(_raw_axes) == 1 else _raw_axes
                    dyn_set = set(dyn_idx)
                    static_cyls = [CYLINDERS[i] for i in range(len(CYLINDERS)) if i not in dyn_set]
                    dyn_cyls = [CYLINDERS[i] for i in dyn_idx]
                    add_obstacles_xy(ax, static_cyls, cyl_radius=CYL_RADIUS)
                    add_dynamic_cylinders_xy(ax, dyn_cyls, dyn_axes_resolved, cand_snapshots,
                                              obs_amplitude=obs_amplitude, cyl_radius=CYL_RADIUS,
                                              drone_radius=drone_radius)
                else:
                    add_obstacles_xy(ax, CYLINDERS, cyl_radius=CYL_RADIUS)

                ax.set_xlabel("x")
                ax.set_ylabel("y")
                ax.set_title(f"XY trajectory (seed {seedmodel}, ep {ep})")
                ax.set_aspect("equal", adjustable="box")
                ax.set_xlim(-6.5, 4.5)
                ax.set_ylim(-2.25, 2.25)
                ax.grid(True, alpha=0.3)
                ax.legend(loc="upper left", fontsize=8)

                out_path = os.path.join(plot_dir, f"xy_ep{ep}.pdf")
                fig.tight_layout()
                fig.savefig(out_path)
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
