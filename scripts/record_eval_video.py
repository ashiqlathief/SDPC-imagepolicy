"""
Standalone video recorder for a single closed-loop Crazyflie evaluation
episode (real trained diffusion policy + projector, same logic as
eval_crazieflie.py), saved as an .mp4.

Three camera options (--camera):
  - "spectator" (default): FIXED, environment-mounted outside view
    (SPECTATOR_CAMERA_CFG in isaac/scripts/crazyflie_env_cfg.py, 960x540)
    -- the drone flies through a static frame, like a human spectator
    watching the corridor. Best for showing someone the task itself.
  - "chase": third-person view MOUNTED ON the drone, follows it around
    (CHASE_CAMERA_CFG, 854x480, wide FOV).
  - "fpv": the same 96x96 onboard camera the policy itself sees
    (get_rgb()/FPV_CAMERA_CFG) -- to show exactly what the policy saw.

Usage:
    python scripts/record_eval_video.py \
        --run_dir isaac/logs/avoiding-crazyflie/diffusion/<run>/9 \
        --variant dpcc-c-tightened \
        --camera spectator \
        --out videos/eval_dpcc-c-tightened.mp4

Run from the repo root with the env_isaaclab conda environment active.
"""
import argparse
import os
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

import diffuser.utils as utils

# Same-directory import (eval_crazieflie.py lives in this same scripts/ folder).
import eval_crazieflie as ec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--variant", type=str, default="dpcc-c-tightened",
                         help="Any name accepted by eval_crazieflie.variant_cfg(), "
                              "e.g. diffuser, post_processing-tightened, dpcc-c-tightened.")
    parser.add_argument("--max_steps", type=int, default=50)
    parser.add_argument("--num_candidates", type=int, default=4)
    parser.add_argument("--action_scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dynamic_obstacles", action="store_true", default=False)
    parser.add_argument("--obs_amplitude", type=float, default=0.35)
    parser.add_argument("--obs_frequency", type=float, default=0.25)
    parser.add_argument("--dynamic_proj_update", action="store_true", default=False)
    parser.add_argument("--dynamic_cyl_indices", type=int, nargs="*", default=None)
    parser.add_argument("--drone_radius", type=float, default=0.10)
    parser.add_argument("--dt", type=float, default=None,
                         help="Override sim/control dt; default auto-derived from the "
                              "trained dataset's control_dt, same as eval_crazieflie.py.")
    parser.add_argument("--out", type=str, default=None,
                         help="Output .mp4 path. Default: <run_dir>/videos/eval_<variant>.mp4")
    parser.add_argument("--video_fps", type=int, default=20,
                         help="Playback fps of the saved video (independent of sim dt).")
    parser.add_argument("--camera", type=str, default="spectator",
                         choices=["spectator", "chase", "fpv"],
                         help="Which camera to record from: 'spectator' is a FIXED, "
                              "environment-mounted outside view (SPECTATOR_CAMERA_CFG, "
                              "960x540) -- the drone flies through the frame, camera doesn't "
                              "move; 'chase' is a third-person view MOUNTED ON the drone "
                              "(CHASE_CAMERA_CFG, 854x480, follows the drone); 'fpv' is the "
                              "same 96x96 onboard camera the policy itself sees "
                              "(get_rgb()/FPV_CAMERA_CFG).")
    args = parser.parse_args()

    from isaac.scripts.crazyflie_env import Crazyflie, CrazyflieEnvCfg
    device = torch.device(args.device)

    # ------------------ Load trained experiment ------------------
    print(f"[INFO] Loading run dir: {args.run_dir}")
    diff_exp = utils.load_diffusion(args.run_dir, epoch="best", device=str(device))
    dataset = diff_exp.dataset
    diffusion = diff_exp.diffusion.to(device)
    diffusion.eval()

    horizon = int(getattr(diffusion, "horizon", 16))
    action_dim = int(getattr(diffusion, "action_dim", 3))
    To = int(getattr(dataset, "n_obs_steps", 2))
    print(f"[INFO] To={To}, H={horizon}, action_dim={action_dim}")

    # ------------------ dt auto-derivation (same as eval_crazieflie.py) ----
    expected_dt = getattr(dataset, "control_dt", getattr(dataset, "dt", None))
    if args.dt is not None:
        eval_dt = args.dt
    elif expected_dt is not None:
        eval_dt = expected_dt
    else:
        eval_dt = CrazyflieEnvCfg.dt
    print(f"[INFO] Using eval dt={eval_dt} (dataset control_dt={expected_dt})")

    # ------------------ Create env ------------------
    cfg = CrazyflieEnvCfg(
        num_envs=1,
        device=str(device),
        dynamic_obstacles=args.dynamic_obstacles,
        obs_amplitude=args.obs_amplitude,
        obs_frequency=args.obs_frequency,
        dynamic_cyl_indices=args.dynamic_cyl_indices,
        dt=eval_dt,
    )
    env = Crazyflie(cfg)

    # ------------------ Variant / projector setup (mirrors eval_crazieflie.py) --
    vcfg = ec.variant_cfg(args.variant)
    num_candidates = vcfg["num_candidates"] if vcfg["num_candidates"] > 0 else args.num_candidates
    projection_mode = vcfg["projection_mode"]
    gradient = (projection_mode == "gradient")
    pos_projector = None
    inloop_projector = None
    proj_dt = vcfg["dt"] if vcfg["dt"] is not None else 0.1

    if vcfg["use_projection"]:
        pos_projector = ec.build_position_projector(
            horizon_H=horizon, gradient=gradient, device=device,
            boxes=ec.BOXES, cylinders=ec.CYLINDERS,
            normalizer=None, tighten=vcfg["tighten"], dt=proj_dt,
            use_dynamics=vcfg.get("use_dynamics", True),
            drone_radius=args.drone_radius,
        )
        if projection_mode == "dpcc":
            inloop_projector = ec.build_inloop_projector(
                horizon_H=horizon, device=device,
                tighten=vcfg["tighten"], dt=proj_dt,
            )

    # ------------------ Video writer ------------------
    camera_fns = {
        "spectator": env.get_spectator_rgb,
        "chase": env.get_chase_rgb,
        "fpv": env.get_rgb,
    }
    get_video_frame = camera_fns[args.camera]

    out_path = args.out
    if out_path is None:
        out_dir = os.path.join(args.run_dir, "videos")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"eval_{args.variant}_{args.camera}.mp4")
    else:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    frame0 = get_video_frame()  # (H, W, 3) uint8, RGB
    h, w = frame0.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, args.video_fps, (w, h))
    print(f"[INFO] Recording '{args.camera}' camera video ({w}x{h} @ {args.video_fps}fps) -> {out_path}")

    def write_frame():
        frame_rgb = get_video_frame()
        writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))

    # ------------------ Episode rollout (mirrors eval_crazieflie.py main loop) --
    print(f"[INFO] ===== Recording variant '{args.variant}' =====")
    _ = env.reset(seed=args.seed)

    for _ in range(3):  # camera warm-up
        pos = env._pos_world().detach().cpu().numpy()[0]
        try:
            env.step(pos.copy())
        except Exception:
            pass
    write_frame()

    rgb0 = ec.get_rgb_from_env(env)
    rgb_hist = deque(maxlen=To)
    for _ in range(To):
        rgb_hist.append(rgb0.copy())

    prev_actions_real = None
    info = {}

    for step in range(args.max_steps):
        pos = env._pos_world().detach().cpu().numpy()[0]
        target = env.target_pos[0].detach().cpu().numpy() if hasattr(env, "target_pos") else np.array([np.nan] * 3)

        if args.variant.startswith("model_free"):
            gain = 0.3
            cmd_xyz = pos.copy()
            cmd_xyz[:3] = pos[:3] + gain * (target[:3] - pos[:3])
            obs_next, rew, done_vec, info = env.step(cmd_xyz)
            rgb_hist.append(ec.get_rgb_from_env(env))
            write_frame()
            done = bool(done_vec[0]) if isinstance(done_vec, (list, tuple, np.ndarray, torch.Tensor)) else bool(done_vec)
            if done:
                break
            continue

        obs_rgb_t = ec.preprocess_rgb_stack(rgb_hist).to(device)
        cond = {"obs_rgb": obs_rgb_t}

        in_loop_projector = inloop_projector if projection_mode == "dpcc" else None
        a_candidates_norm, infos = ec.sample_action_candidates(
            diffusion=diffusion, cond=cond, horizon=horizon, action_dim=action_dim,
            num_candidates=num_candidates, projector=in_loop_projector,
        )
        a_candidates_real = dataset.action_normalizer.unnormalize(a_candidates_norm)

        if args.dynamic_obstacles and args.dynamic_proj_update and vcfg["use_projection"]:
            cyl_now = env.get_cylinder_positions()
            pos_projector = ec.build_position_projector(
                horizon_H=horizon, gradient=gradient, device=device,
                boxes=ec.BOXES, cylinders=cyl_now,
                normalizer=None, tighten=vcfg["tighten"], dt=proj_dt,
                use_dynamics=vcfg.get("use_dynamics", True),
                obs_amplitude=0.0,
                drone_radius=args.drone_radius,
            )

        if vcfg["use_projection"] and projection_mode in ("post", "dpcc"):
            a_candidates_proj_real, proj_costs = ec.project_action_candidates_with_positions(
                projector=pos_projector, pos0=pos[:3],
                a_candidates_real=a_candidates_real, device=device,
            )
        else:
            a_candidates_proj_real, proj_costs = a_candidates_real, None

        if vcfg["selection"] == "minimum_projection_cost" and proj_costs is not None:
            which = int(np.argmin(proj_costs))
        else:
            which = ec.choose_trajectory(
                a_candidates_proj_real, infos=infos,
                strategy=vcfg["selection"], prev_actions_real=prev_actions_real,
            )

        a0_real = a_candidates_proj_real[which, 0] * float(args.action_scale)
        prev_actions_real = a_candidates_proj_real[which:which + 1]

        cmd_xyz = pos.copy()
        cmd_xyz[:action_dim] = cmd_xyz[:action_dim] + a0_real[:action_dim]
        obs_next, rew, done_vec, info = env.step(cmd_xyz)

        rgb_hist.append(ec.get_rgb_from_env(env))
        write_frame()

        done = bool(done_vec[0]) if isinstance(done_vec, (list, tuple, np.ndarray, torch.Tensor)) else bool(done_vec)
        print(f"step {step:04d} pos={obs_next[0]} target={target} done={done}")
        if done:
            print("[INFO] Done=True. Breaking episode loop.")
            break

    writer.release()
    success = bool(info.get("success", [False])[0]) if info else None
    fell = bool(info.get("fell", [False])[0]) if info else None
    print(f"[INFO] Episode finished. success={success} fell={fell}")
    print(f"[INFO] Video saved -> {out_path}")

    env.close()


if __name__ == "__main__":
    main()
