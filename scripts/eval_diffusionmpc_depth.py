from pathlib import Path
import numpy as np
np.set_printoptions(precision=3, suppress=True)
import torch

import diffuser.utils as utils
import diffuser.sampling.projection as projection_mod
from diffuser.sampling.projection import Projector
from diffuser.sampling.policies import temporal_consistency_distances
projection_mod.DEBUG_SLSQP = False

import depth_obstacle_estimator as detect_mod
from depth_obstacle_estimator import (
    camera_world_pose, quat_apply, detect_obstacles_umap,
    DEPTH_WIDTH, DEPTH_HEIGHT, DEPTH_FX, DEPTH_FY, DEPTH_CX, DEPTH_CY,
)
detect_mod.DEBUG_DETECT = False

KEEPOUT_ZONES = [
    # (-2.0, 0.0, 0.5),
]

RUN_DIR = "isaac/logs/avoiding-crazyflie/diffusion/H8_K20_Dmodels.ImagePoseCondUNet1DTemporalCondModel_Evitp_L384/7"

PROJ_TIGHTEN = 0.15
PROJ_DT = 0.1
FLIGHT_Z_MIN = 0.02
FLIGHT_Z_MAX = 1.5
_Z_HALFSPACES = [
    ([0.0, 0.0, 1.0], FLIGHT_Z_MAX),    # z <= FLIGHT_Z_MAX
    ([0.0, 0.0, -1.0], FLIGHT_Z_MIN),   # z >= FLIGHT_Z_MIN
]
DRONE_RADIUS = 0.15
START_POS = np.array([-5.0, 0.0, 0.75], dtype=np.float32)   # dummy fixed drone position
START_QUAT = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # (w,x,y,z) identity -- facing +x

MAX_DEPTH_OBSTACLES = 5
UMAP_MAX_RANGE = 3.0
UMAP_BIN_SIZE = 100
UMAP_T_POI = 500.0
UMAP_T_THO = 1800.0
UMAP_MIN_PIXEL_COUNT = 50
DEPTH_OBSTACLE_RADIUS = 0.3

VARIANT_CFG = {
    "sdpc-r": dict(num_candidates=1, selection="first"),
    "sdpc-c": dict(num_candidates=2, selection="minimum_projection_cost"),
    "sdpc-t": dict(num_candidates=2, selection="temporal_consistency"),
}


def make_synthetic_depth_frame():
    """(DEPTH_HEIGHT, DEPTH_WIDTH) depth-in-meters frame: far background
    (dropped by detect_obstacles_umap's max-range histogram, so it never
    competes for "largest contour") plus one near rectangular patch standing
    in for a single obstacle directly ahead of the camera."""
    depth = np.full((DEPTH_HEIGHT, DEPTH_WIDTH), UMAP_MAX_RANGE * 2.0, dtype=np.float32)
    depth[150:250, 350:500] = 1.2  # ~1.2m-away "obstacle" patch
    return depth


def detect_depth_obstacles_dummy(pos_body_w, quat_body_w):
    """Same math as eval_crazieflie1pos.py's detect_depth_obstacles(), just fed
    a synthetic depth frame and a dummy drone pose instead of a live env."""
    depth_2d = make_synthetic_depth_frame()
    pos_cam_w, quat_cam_w = camera_world_pose(pos_body_w, quat_body_w)

    detections = detect_obstacles_umap(
        depth_2d, DEPTH_FX, DEPTH_FY, DEPTH_CX, DEPTH_CY,
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


def build_projector(horizon_H, device, static_points, drone_radius, pos0, action_normalizer):
    lb = np.array([-6.5, -1.95, FLIGHT_Z_MIN], dtype=np.float32)
    ub = np.array([4.5, 1.95, FLIGHT_Z_MAX], dtype=np.float32)
    constraint_list = [("lb", lb), ("ub", ub)]

    for (x, y, radius) in static_points:
        r = radius + drone_radius + PROJ_TIGHTEN
        constraint_list.append(("sphere_outside", [0, 1], [float(x), float(y)], float(r)))
    for (x, y, zone_radius) in KEEPOUT_ZONES:
        r = float(zone_radius) + drone_radius + PROJ_TIGHTEN
        constraint_list.append(("sphere_outside", [0, 1], [float(x), float(y)], r))
    for normal, rhs in _Z_HALFSPACES:
        constraint_list.append(("ineq", (np.array(normal, dtype=np.float32), float(rhs))))

    projector = Projector(
        horizon=horizon_H + 1, transition_dim=3, action_dim=0, goal_dim=0,
        constraint_list=constraint_list, normalizer=None, gradient=False,
        gradient_weights=[1, 0.5, 2], dt=PROJ_DT, variant="states",
        skip_initial_state=True, diffusion_timestep_threshold=0.8,
        device=str(device), solver="scipy", parallelize=True, goal_pull_weight=0.0,
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
    obs_rgb = cond["obs_rgb"]  # (1,To,3,H,W)
    cond_k = {"obs_rgb": obs_rgb.repeat(num_candidates, 1, 1, 1, 1)}
    if "goal_rel" in cond:
        cond_k["goal_rel"] = cond["goal_rel"].repeat(num_candidates, 1)
    with torch.no_grad():
        x, _ = diffusion.conditional_sample(cond_k, horizon=horizon, projector=projector)  # (K,H,D)
    return x[:, :, :action_dim].detach().cpu().numpy()


print(f"\n[INFO] Loading run dir: {RUN_DIR}")
device = torch.device("cuda:0")

seedmodel = int(Path(RUN_DIR).name)
diff_exp = utils.load_diffusion(RUN_DIR, epoch="best", device=str(device))
dataset = diff_exp.dataset
diffusion = diff_exp.diffusion.to(device)
diffusion.eval()
horizon = int(getattr(diffusion, "horizon", 16))
action_dim = int(getattr(diffusion, "action_dim", 3))
To = int(getattr(dataset, "n_obs_steps", 2))
img_size = int(getattr(dataset, "img_size", 96))

obs_rgb = torch.rand(1, To, 3, img_size, img_size, device=device)
goal_rel = torch.zeros(1, 3, device=device)
cond = {"obs_rgb": obs_rgb, "goal_rel": goal_rel}

static_pts = detect_depth_obstacles_dummy(START_POS, START_QUAT)
print(f"[INFO] depth-detected obstacles (synthetic frame): {static_pts}")

for variant_name, vcfg in VARIANT_CFG.items():
    projector = build_projector(horizon, device, static_pts, DRONE_RADIUS,
                                 START_POS, dataset.action_normalizer)

    a_horizon_norm = sample_action_horizon(diffusion, cond, horizon, action_dim,
                                            projector=projector, num_candidates=vcfg["num_candidates"])
    a_horizon_real = dataset.action_normalizer.unnormalize(a_horizon_norm)
    a_horizon_real[:, :, :3], proj_costs = project_deltas_from_pos(
        projector, START_POS, a_horizon_real[:, :, :3], device
    )
    choice = choose_candidate(a_horizon_real, proj_costs, None, vcfg["selection"])
    a0_real = a_horizon_real[choice, 0]

    print(f"\n[{variant_name}] num_candidates={vcfg['num_candidates']} selection={vcfg['selection']} "
          f"chosen={choice} proj_costs={proj_costs}")
    print(f"[{variant_name}] a0_real={a0_real}")
    print(f"[{variant_name}] full horizon (real units):")
    print(a_horizon_real[choice])
