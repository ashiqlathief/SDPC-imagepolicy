import argparse
import os
import pickle
from collections import namedtuple
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.sim import SimulationContext
from isaaclab_assets import CRAZYFLIE_CFG


# ============================================================
# Minimal diffusion loading
# ============================================================
DiffusionExperiment = namedtuple("Diffusion", "model diffusion epoch action_normalizer")


def load_config(path: str):
    with open(path, "rb") as f:
        config = pickle.load(f)
    print(f"[load] Loaded config → {path}")
    return config


def load_diffusion_minimal(run_dir: str, device: str = "cuda:0"):
    run_dir = Path(run_dir)
    print(f"\n[load] Loading diffusion from {run_dir}\n")

    # 1. Model
    model_config = load_config(run_dir / "model_config.pkl")
    model = model_config().to(device)

    # 2. Diffusion
    diffusion_config = load_config(run_dir / "diffusion_config.pkl")
    diffusion = diffusion_config(model).to(device)

    # 3. Weights
    ckpt_path = run_dir / "state_best.pt"
    if not ckpt_path.is_file():
        candidates = sorted(run_dir.glob("state_*.pt"))
        if not candidates:
            raise FileNotFoundError(f"No checkpoint found in {run_dir}")
        ckpt_path = candidates[-1]
        print(f"[load] state_best.pt not found → using {ckpt_path.name}")

    ckpt = torch.load(ckpt_path, map_location=device)
    print(f"[load] Loaded weights from {ckpt_path.name}")

    if isinstance(ckpt, dict):
        if "ema" in ckpt:
            state_dict = ckpt["ema"]
            print("[load] Using EMA weights")
        elif "model" in ckpt:
            state_dict = ckpt["model"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    # Strip "model." prefix
    cleaned = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            cleaned[k[len("model."):]] = v

    if not cleaned:
        cleaned = state_dict
        print("[load] Warning: no 'model.' prefix found – loading raw state_dict")

    if hasattr(diffusion, "model"):
        missing, unexpected = diffusion.model.load_state_dict(cleaned, strict=False)
        if missing:
            print(f"[load] Missing keys (first 5): {missing[:5]}")
        if unexpected:
            print(f"[load] Unexpected keys (first 5): {unexpected[:5]}")
        print("[load] Model weights loaded (strict=False)")
    else:
        model.load_state_dict(cleaned, strict=False)

    diffusion.eval()

    # 4. Action normalizer
    action_normalizer = None
    try:
        dataset_config = load_config(run_dir / "dataset_config.pkl")
        try:
            ds = dataset_config()
            action_normalizer = getattr(ds, "action_normalizer", None)
            if action_normalizer is None and hasattr(ds, "normalizer"):
                action_normalizer = ds.normalizer
            print("[load] Got action_normalizer from dataset")
        except Exception as e:
            print(f"[load] Could not instantiate dataset ({e})")
            if hasattr(dataset_config, "action_normalizer"):
                action_normalizer = dataset_config.action_normalizer
            elif hasattr(dataset_config, "normalizer"):
                action_normalizer = dataset_config.normalizer
    except Exception as e:
        print(f"[load] Warning: could not load dataset_config → {e}")

    return DiffusionExperiment(
        model=model,
        diffusion=diffusion,
        epoch="best",
        action_normalizer=action_normalizer,
    )


def unnormalize_actions(actions: np.ndarray, normalizer) -> np.ndarray:
    """Try to unnormalize. Returns input unchanged if normalizer is None or identity."""
    if normalizer is None:
        return actions

    # DatasetNormalizer style
    if hasattr(normalizer, "unnormalize") and "key" in normalizer.unnormalize.__code__.co_varnames:
        return normalizer.unnormalize(actions, key="actions")

    # Single-field normalizer
    if hasattr(normalizer, "unnormalize"):
        return normalizer.unnormalize(actions)

    # Nested
    if hasattr(normalizer, "normalizers") and "actions" in normalizer.normalizers:
        return normalizer.normalizers["actions"].unnormalize(actions)

    return actions


# ============================================================
# Load model
# ============================================================
RUN_DIR = "/home/huyen-admin/spear_upb_ws/IsaacLab/scripts/demos/diffusion_models/H8_K20_Dmodels.ImagePoseCondUNet1DTemporalCondModel_Evitp_L384/7/"
print(f"\n[INFO] Loading diffusion run: {RUN_DIR}")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
diff_exp = load_diffusion_minimal(RUN_DIR, device=str(device))

diffusion = diff_exp.diffusion
action_normalizer = diff_exp.action_normalizer

horizon    = int(getattr(diffusion, "horizon", 8))
action_dim = int(getattr(diffusion, "action_dim", 3))
To         = 2
img_size   = 96

print(f"[INFO] horizon={horizon}, action_dim={action_dim}, To={To}, img_size={img_size}")

# ===================== NORMALIZER DEBUG =====================
print("\n========== NORMALIZER DEBUG ==========")
print("action_normalizer is None?", action_normalizer is None)
if action_normalizer is not None:
    print("type:", type(action_normalizer))
    if hasattr(action_normalizer, "normalizers") and "actions" in getattr(action_normalizer, "normalizers", {}):
        n = action_normalizer.normalizers["actions"]
        print("→ nested actions normalizer:", type(n))
        if hasattr(n, "mins"):
            print("  mins:", n.mins)
            print("  maxs:", n.maxs)
        if hasattr(n, "means"):
            print("  means:", n.means)
            print("  stds :", n.stds)
    elif hasattr(action_normalizer, "mins"):
        print("mins:", action_normalizer.mins)
        print("maxs:", action_normalizer.maxs)
    elif hasattr(action_normalizer, "means"):
        print("means:", action_normalizer.means)
        print("stds :", action_normalizer.stds)
    else:
        print("Normalizer attributes:", [x for x in dir(action_normalizer) if not x.startswith("_")])
else:
    print("→ No normalizer loaded. Will apply manual scaling.")
print("======================================\n")


def preprocess_rgb(rgb_tensor: torch.Tensor) -> torch.Tensor:
    if rgb_tensor.dtype == torch.uint8:
        rgb_tensor = rgb_tensor.float() / 255.0
    img = rgb_tensor.permute(0, 3, 1, 2)
    img = TF.resize(img, [img_size, img_size], antialias=True)
    return img


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[0.0, -6.0, 3.0], target=[0.0, 0.0, 1.0])

    # Start & Goal
    start_pos = torch.tensor([-4.0, 0.0, 1.0], device=args_cli.device)
    goal_pos  = torch.tensor([ 2.0, 1.0, 1.0], device=args_cli.device)
    desired_pos = start_pos.clone()

    # Ground + light
    sim_utils.GroundPlaneCfg().func("/World/defaultGroundPlane", sim_utils.GroundPlaneCfg())
    sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)).func(
        "/World/Light", sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # Robot
    robot_cfg = CRAZYFLIE_CFG.replace(prim_path="/World/Crazyflie")
    robot_cfg.init_state.pos = (start_pos[0].item(), start_pos[1].item(), start_pos[2].item())
    robot_cfg.spawn.func("/World/Crazyflie", robot_cfg.spawn, translation=robot_cfg.init_state.pos)
    robot = Articulation(robot_cfg)

    # Camera
    camera_cfg = CameraCfg(
        prim_path="/World/Crazyflie/body/front_camera",
        offset=CameraCfg.OffsetCfg(
            pos=(0.05, 0.0, 0.02),
            rot=(0.5, -0.5, 0.5, -0.5),
            convention="ros",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 20.0),
        ),
        update_period=0.02,
        width=96,
        height=96,
    )
    camera = Camera(cfg=camera_cfg)

    sim.reset()
    print("[INFO] Simulation + camera + diffusion ready.")
    print(f"[INFO] Start: {start_pos.cpu().numpy()}  →  Goal: {goal_pos.cpu().numpy()}")

    obs_history = []
    sim_dt = sim.get_physics_dt()
    count = 0
    debug_print_every = 20

    # -------------------- Manual scaling parameters --------------------
    # Because the network currently outputs huge values and the normalizer
    # is identity / None, we force a reasonable step size.
    MAX_STEP = 0.12          # maximum meters the drone may move per diffusion call
    MANUAL_SCALE = 0.1 # 0.00015   # extra scale if the raw values are still large

    while simulation_app.is_running():
        # -------- Reset --------
        if count % 3000 == 0:
            count = 0
            robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)
            robot.reset()
            obs_history.clear()
            desired_pos = start_pos.clone()
            print(">>>>>>>> Reset to start position!")

        # -------- Camera --------
        camera.update(dt=sim_dt)
        rgb = camera.data.output["rgb"]
        img = preprocess_rgb(rgb).to(device)
        obs_history.append(img)
        if len(obs_history) > To:
            obs_history.pop(0)

        # -------- Diffusion (every step) --------
        if len(obs_history) == To:
            obs_rgb = torch.cat(obs_history, dim=0).unsqueeze(0)

            current_pos = robot.data.root_pos_w[0]
            goal_rel = (goal_pos - current_pos).unsqueeze(0)

            cond = {
                "obs_rgb": obs_rgb,
                "goal_rel": goal_rel,
            }

            with torch.no_grad():
                x, _ = diffusion.conditional_sample(cond, horizon=horizon, projector=None)

            actions_norm = x[0, :, :action_dim].detach().cpu().numpy()  # (H, 3)

            # Try the loaded normalizer (currently does nothing)
            deltas = unnormalize_actions(actions_norm, action_normalizer)
            raw_delta = deltas[0].copy()

            # --------------- CORRECT / SAFE NORMALIZATION ---------------
            # 1. Apply a strong manual scale (because network outputs are ~100-800)
            delta = raw_delta * MANUAL_SCALE

            # 2. Limit the maximum step length
            norm = np.linalg.norm(delta)
            if norm > MAX_STEP:
                delta = delta / (norm + 1e-8) * MAX_STEP

            # 3. Compute new position
            current_np = current_pos.detach().cpu().numpy()
            new_pos = current_np + delta

            desired_pos = torch.tensor(new_pos, device=args_cli.device, dtype=torch.float32)

            # Height safety
            desired_pos[2] = torch.clamp(desired_pos[2], 0.4, 2.0)

            # -------------------- Debug --------------------
            if count % debug_print_every == 0:
                print("\n---------- Diffusion Debug (step {}) ----------".format(count))
                print(f"current_pos      : {np.round(current_np, 4)}")
                print(f"goal_rel         : {np.round(goal_rel[0].cpu().numpy(), 4)}")
                print(f"raw network out  : {np.round(actions_norm[0], 4)}")
                print(f"after unnorm     : {np.round(raw_delta, 4)}")
                print(f"after manual scale + clamp : {np.round(delta, 4)}")
                print(f"new desired_pos  : {np.round(desired_pos.cpu().numpy(), 4)}")
                print("------------------------------------------------\n")

        # -------- Write to sim --------
        root_pose = robot.data.default_root_state[:, :7].clone()
        root_pose[:, 0:3] = desired_pos
        robot.write_root_pose_to_sim(root_pose)
        robot.write_root_velocity_to_sim(torch.zeros_like(robot.data.default_root_state[:, 7:]))

        robot.write_data_to_sim()
        sim.step()
        robot.update(sim_dt)

        count += 1


if __name__ == "__main__":
    main()
    simulation_app.close()