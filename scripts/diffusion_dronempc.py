import argparse

import numpy as np
np.set_printoptions(precision=3, suppress=True)
import torch
import torchvision.transforms.functional as TF

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.sim.spawners.materials import PreviewSurfaceCfg

from isaac.scripts.arl_robot_1_cfg import ARL_ROBOT_1_CFG
import diffuser.utils as utils
import diffuser.sampling.projection as projection_mod
from diffuser.sampling.projection import Projector
from diffuser.sampling.policies import temporal_consistency_distances
projection_mod.DEBUG_SLSQP = False

CORRIDOR_LENGTH = 11        # x direction
CORRIDOR_WIDTH = 5.0        # y direction (clearance between walls)
WALL_THICKNESS = 0.10
WALL_HEIGHT = 3.0
CORRIDOR_X_OFFSET = -CORRIDOR_LENGTH / 2.0

_WALL_ASSET_DIR    = "Environments/Simple_Warehouse/Props"
_WALL_6M           = f"{_WALL_ASSET_DIR}/SM_WallA_6M.usd"
_WALL_3M           = f"{_WALL_ASSET_DIR}/SM_WallA_3M.usd"
_WALL_NATIVE_HEIGHT = 3.1
_WALL_NATIVE_LEN_6M = 6.0
_WALL_NATIVE_LEN_3M = 3.0
_WALL_Z_SCALE      = WALL_HEIGHT / _WALL_NATIVE_HEIGHT   # squash 3.1 m -> WALL_HEIGHT

_ROT_YAW_0   = (1.0,    0.0, 0.0,  0.0)      # identity
_ROT_YAW_P90 = (0.7071, 0.0, 0.0,  0.7071)   # +90 deg about Z: local +X -> world +Y
_ROT_YAW_N90 = (0.7071, 0.0, 0.0, -0.7071)   # -90 deg about Z: local +X -> world -Y

_CORR_X0 = CORRIDOR_X_OFFSET                        # open end of the corridor
_CORR_X1 = CORRIDOR_LENGTH + CORRIDOR_X_OFFSET       # closed end of the corridor
_CORR_Y  = CORRIDOR_WIDTH / 2.0                      # inner-face y of each side wall
_SEG_6M_LEN       = _WALL_NATIVE_LEN_6M
_SEG_FILL_LEN     = CORRIDOR_LENGTH - _WALL_NATIVE_LEN_3M - _SEG_6M_LEN   # = 2.0 for CORRIDOR_LENGTH = 11
_SEG_FILL_SCALE_Y = _SEG_FILL_LEN / _WALL_NATIVE_LEN_3M
_SEG_END_LEN      = CORRIDOR_LENGTH - _SEG_6M_LEN - _SEG_FILL_LEN         # = 3.0, closes the run at _CORR_X1

_END_WALL_SPAN       = CORRIDOR_WIDTH + 2 * WALL_THICKNESS   # overlaps a bit into each side wall
_END_WALL_SCALE_Y    = _END_WALL_SPAN / _WALL_NATIVE_LEN_6M

RED_MAT = PreviewSurfaceCfg(diffuse_color=(0.85, 0.10, 0.10))

WALLS_MODULAR = [
    # ---- left wall (y = -CORR_Y), full corridor length ----
    (_WALL_6M, _CORR_X0 + _SEG_6M_LEN / 2.0,                                -_CORR_Y, 0.0, _ROT_YAW_N90, (1.0, 1.0, _WALL_Z_SCALE)),
    (_WALL_3M, _CORR_X0 + _SEG_6M_LEN + _SEG_FILL_LEN / 2.0,                -_CORR_Y, 0.0, _ROT_YAW_N90, (1.0, _SEG_FILL_SCALE_Y, _WALL_Z_SCALE)),
    (_WALL_3M, _CORR_X0 + _SEG_6M_LEN + _SEG_FILL_LEN + _SEG_END_LEN / 2.0, -_CORR_Y, 0.0, _ROT_YAW_N90, (1.0, 1.0, _WALL_Z_SCALE)),

    # ---- right wall (y = +CORR_Y), full corridor length ----
    (_WALL_6M, _CORR_X0 + _SEG_6M_LEN / 2.0,                                _CORR_Y, 0.0, _ROT_YAW_P90, (1.0, 1.0, _WALL_Z_SCALE)),
    (_WALL_3M, _CORR_X0 + _SEG_6M_LEN + _SEG_FILL_LEN / 2.0,                _CORR_Y, 0.0, _ROT_YAW_P90, (1.0, _SEG_FILL_SCALE_Y, _WALL_Z_SCALE)),
    (_WALL_3M, _CORR_X0 + _SEG_6M_LEN + _SEG_FILL_LEN + _SEG_END_LEN / 2.0, _CORR_Y, 0.0, _ROT_YAW_P90, (1.0, 1.0, _WALL_Z_SCALE)),

    # ---- end wall, closing the corridor across its width at _CORR_X1 ----
    (_WALL_6M, _CORR_X1, 0.0, 0.0, _ROT_YAW_0, (1.0, _END_WALL_SCALE_Y, _WALL_Z_SCALE)),
]

DRONE_RADIUS = 0.26   # ~half the ARL "lmf2" body's ~0.52m width -- retune if it clips
PROJ_TIGHTEN = 0.15
PROJ_DT = 0.1
FLIGHT_Z_MIN = 0.4
FLIGHT_Z_MAX = 2.0
_Z_HALFSPACES = [
    ([0.0, 0.0, 1.0], FLIGHT_Z_MAX),    # z <= FLIGHT_Z_MAX
    ([0.0, 0.0, -1.0], FLIGHT_Z_MIN),   # z >= FLIGHT_Z_MIN
]
_FLIGHT_MARGIN = DRONE_RADIUS + PROJ_TIGHTEN
FLIGHT_LB = np.array([_CORR_X0 + _FLIGHT_MARGIN, -_CORR_Y + _FLIGHT_MARGIN, FLIGHT_Z_MIN], dtype=np.float32)
FLIGHT_UB = np.array([_CORR_X1 - _FLIGHT_MARGIN,  _CORR_Y - _FLIGHT_MARGIN, FLIGHT_Z_MAX], dtype=np.float32)
STATIC_OBSTACLES = [    # (2, 0.7),
    (2.5, 0.5),
    # (2, 0.0),
    (2.5, -0.5),
    # (2, -0.8),
    (1, 0.8),
    # # (0.5, 0.4),
    (1, 0.0),
    # # (0.5, -0.4),
    (1, -0.9),]   # [(x, y), ...]
KEEPOUT_ZONES = []      # [(x, y, radius), ...] planner-only virtual keep-outs

CYL_RADIUS = 0.15   # matches CylinderCfg radius in env_cfg.py / crazyflie_env_cfg1.py
CYL_HEIGHT = 2.0

MPC_NUM_CANDIDATES = 2
MPC_SELECTION = "minimum_projection_cost"   # "first" | "minimum_projection_cost" | "temporal_consistency"


def build_projector(horizon_H, device, static_points, drone_radius, pos0, action_normalizer, keepout_zones=None):
    constraint_list = [("lb", FLIGHT_LB), ("ub", FLIGHT_UB)]
    for (x, y) in static_points:
        radius = 0.2 + drone_radius + PROJ_TIGHTEN
        constraint_list.append(("sphere_outside", [0, 1], [float(x), float(y)], radius))
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
        device=str(device), solver="scipy", parallelize=False, goal_pull_weight=0.0,
    )
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


def sample_action_horizon(diffusion, cond, horizon, action_dim, projector, num_candidates):
    """Full predicted horizon for K candidate samples (receding-horizon MPC style:
    resampled fresh every control step), with the in-loop SLSQP projector passed
    straight through to conditional_sample() -- the correction happens progressively
    during denoising, same as eval_crazieflie1pos.py's "sdpc" variants."""
    obs_rgb = cond["obs_rgb"]  # (1,To,3,H,W)
    cond_k = {"obs_rgb": obs_rgb.repeat(num_candidates, 1, 1, 1, 1)}
    if "goal_rel" in cond:
        cond_k["goal_rel"] = cond["goal_rel"].repeat(num_candidates, 1)
    with torch.no_grad():
        x, _ = diffusion.conditional_sample(cond_k, horizon=horizon, projector=projector)  # (K,H,D)
    return x[:, :, :action_dim].detach().cpu().numpy()


RUN_DIR = "isaac/logs/avoiding-crazyflie/diffusion/H8_K20_Dmodels.ImagePoseCondUNet1DTemporalCondModel_Evitp_L384/7"
print(f"\n[INFO] Loading diffusion run: {RUN_DIR}")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
diff_exp = utils.load_diffusion(RUN_DIR, epoch="best", device=str(device))
dataset = diff_exp.dataset
diffusion = diff_exp.diffusion.to(device)
diffusion.eval()
action_normalizer = dataset.action_normalizer

horizon    = int(getattr(diffusion, "horizon", 16))
action_dim = int(getattr(diffusion, "action_dim", 3))
To         = int(getattr(dataset, "n_obs_steps", 2))
img_size   = int(getattr(dataset, "img_size", 96))

print(f"[INFO] horizon={horizon}, action_dim={action_dim}, To={To}, img_size={img_size}")
print(f"[INFO] action_normalizer: {type(action_normalizer).__name__} "
      f"mins={getattr(action_normalizer, 'mins', None)} maxs={getattr(action_normalizer, 'maxs', None)}")
print(f"[INFO] MPC: num_candidates={MPC_NUM_CANDIDATES} selection={MPC_SELECTION} "
      f"drone_radius={DRONE_RADIUS} flight_lb={FLIGHT_LB} flight_ub={FLIGHT_UB}")


def preprocess_rgb(rgb_tensor: torch.Tensor) -> torch.Tensor:
    if rgb_tensor.dtype == torch.uint8:
        rgb_tensor = rgb_tensor.float() / 255.0
    img = rgb_tensor.permute(0, 3, 1, 2)
    img = TF.resize(img, [img_size, img_size], antialias=True)
    return img


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=1/30, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[-10.48034, -1.2583, 4.51853], target=[0.0, 0.0, 1.0])

    # Start & Goal
    start_pos = torch.tensor([-4.0, 0.0, 1.0], device=args_cli.device)
    goal_pos  = torch.tensor([ 2.0, 1.0, 1.0], device=args_cli.device)
    desired_pos = start_pos.clone()

    # Ground + light
    sim_utils.GroundPlaneCfg().func("/World/defaultGroundPlane", sim_utils.GroundPlaneCfg(color=(0.5, 0.5, 0.5)))
    sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)).func(
        "/World/Light", sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    for i, (usd, x, y, z, rot, scale) in enumerate(WALLS_MODULAR):
        wall_cfg = sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/{usd}",
            scale=scale,
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        )
        wall_cfg.func(f"/World/Walls/WallModular{i:02d}", wall_cfg, translation=(x, y, z), orientation=rot)

    for i, (x, y) in enumerate(STATIC_OBSTACLES):
        cyl_cfg = sim_utils.CylinderCfg(
            visual_material=RED_MAT,
            radius=CYL_RADIUS,
            height=CYL_HEIGHT,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        )
        cyl_cfg.func(f"/World/Obstacles/Cyl{i:02d}", cyl_cfg, translation=(x, y, CYL_HEIGHT / 2.0))

    robot_cfg = ARL_ROBOT_1_CFG.replace(prim_path="/World/ArlRobot")
    robot_cfg.init_state.pos = (start_pos[0].item(), start_pos[1].item(), start_pos[2].item())
    robot_cfg.spawn.func("/World/ArlRobot", robot_cfg.spawn, translation=robot_cfg.init_state.pos)
    robot = Articulation(robot_cfg)

    camera_cfg = CameraCfg(
        prim_path="/World/ArlRobot/base_link/front_camera",
        offset=CameraCfg.OffsetCfg(
            pos=(0.30, 0.0, 0.0),
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
    print("[INFO] Simulation + camera + diffusion + MPC ready.")
    print(f"[INFO] Start: {start_pos.cpu().numpy()}  →  Goal: {goal_pos.cpu().numpy()}")

    obs_history = []
    sim_dt = sim.get_physics_dt()
    count = 0
    debug_print_every = 1

    MAX_STEP = 0.2  # metres per control step; retune if it clips legitimate motion
    prev_actions_real = None  # previous step's executed (H,3) chunk, for
                               # MPC_SELECTION="temporal_consistency"

    projector = build_projector(
        horizon, device, STATIC_OBSTACLES, DRONE_RADIUS,
        start_pos.detach().cpu().numpy(), action_normalizer, keepout_zones=KEEPOUT_ZONES,
    )

    while simulation_app.is_running():
        # -------- Reset --------
        if count % 300 == 0:
            count = 0
            robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)
            robot.reset()
            obs_history.clear()
            desired_pos = start_pos.clone()
            prev_actions_real = None
            print(">>>>>>>> Reset to start position!")

        # -------- Camera --------
        camera.update(dt=sim_dt)
        rgb = camera.data.output["rgb"]
        img = preprocess_rgb(rgb).to(device)
        obs_history.append(img)
        if len(obs_history) > To:
            obs_history.pop(0)

        # -------- Diffusion + MPC (every step) --------
        if len(obs_history) == To:
            obs_rgb = torch.cat(obs_history, dim=0).unsqueeze(0)

            current_pos = robot.data.root_pos_w[0]
            current_np = current_pos.detach().cpu().numpy()
            goal_rel = (goal_pos - current_pos).unsqueeze(0)

            cond = {
                "obs_rgb": obs_rgb,
                "goal_rel": goal_rel,
            }

            projector.pos0 = current_np

            a_horizon_norm = sample_action_horizon(
                diffusion, cond, horizon, action_dim,
                projector=projector, num_candidates=MPC_NUM_CANDIDATES,
            )  # (K, H, 3), normalized [-1,1]
            a_horizon_real = action_normalizer.unnormalize(a_horizon_norm)  # (K, H, 3), metres
            a_horizon_real[:, :, :3], proj_costs = project_deltas_from_pos(
                projector, current_np, a_horizon_real[:, :, :3], device
            )
            choice = choose_candidate(a_horizon_real, proj_costs, prev_actions_real, MPC_SELECTION)
            prev_actions_real = a_horizon_real[choice]

            raw_delta = a_horizon_real[choice, 0].copy()
            delta = raw_delta.copy()
            norm = np.linalg.norm(delta)
            if norm > MAX_STEP:
                delta = delta / (norm + 1e-8) * MAX_STEP

            # Compute new position
            new_pos = current_np + delta
            desired_pos = torch.tensor(new_pos, device=args_cli.device, dtype=torch.float32)

            # Height safety
            desired_pos[2] = torch.clamp(desired_pos[2], 0.4, 2.0)

            # -------------------- Debug --------------------
            if count % debug_print_every == 0:
                print("\n---------- MPC Debug (step {}) ----------".format(count))
                print(f"current_pos      : {np.round(current_np, 4)}")
                print(f"goal_rel         : {np.round(goal_rel[0].cpu().numpy(), 4)}")
                print(f"proj_costs       : {proj_costs}  choice={choice}")
                print(f"raw network out  : {np.round(a_horizon_norm[choice, 0], 4)}")
                print(f"after unnorm/proj: {np.round(raw_delta, 4)}")
                print(f"after safety cap : {np.round(delta, 4)}")
                print(f"new desired_pos  : {np.round(desired_pos.cpu().numpy(), 4)}")

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
