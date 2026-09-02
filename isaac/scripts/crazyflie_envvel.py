from dataclasses import dataclass
import argparse
import os
from isaaclab.app import AppLauncher
import torch
import numpy as np
import gymnasium as gym
import math

_parser = argparse.ArgumentParser(add_help=False)
AppLauncher.add_app_launcher_args(_parser)
_app_args, _ = _parser.parse_known_args()
_app_args.headless = os.environ.get("CRAZYFLIE_ENV_HEADLESS", "1") != "0"
_app_args.enable_cameras = True
app_launcher = AppLauncher(_app_args)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene
from isaaclab.utils.math import quat_apply

PROPELLER_JOINT_NAMES = ("m1_joint", "m2_joint", "m3_joint", "m4_joint")
PROPELLER_VELOCITY_TARGETS = (-130.0, 130.0, -130.0, 130.0)


from .crazyflie_env_cfg1 import (CrazyflieSceneCfg, CYLINDERS, CORRIDOR_LENGTH,
                                DEPTH_FAR)

@dataclass
class CrazyflieEnvCfg:
    num_envs: int = 1
    env_spacing: float = 2.0
    dt: float = 1.0 / 50.0  # matches exp2vla's execise_01_c.py (this run's data collector):
                             # sim.dt=1/50, decimation=2 -> 0.04s/control-step. self.count
                             # below must stay in sync so count*dt reproduces that exactly.
    device: str = "cuda:0"
    gate_x_min: float = 3.95
    gate_x_max: float =  4.0
    gate_y: float = 1.0
    gate_y_tol: float = 0.005   # how close to y=1 counts
    min_z: float = 0.02
    max_z: float = 2.0
    reset_on_fail: bool = False # if True, env auto-resets inside step()
    success_radius: float = 0.2
    drone_radius: float = 0.10   # Crazyflie body radius (m) for collision checks
    dynamic_obstacles: bool = False
    obs_amplitude: float = 0.25   # sinusoid amplitude in metres
    obs_frequency: float = 0.25   # oscillation frequency in Hz
    dynamic_cyl_indices: list | None = None  # which cylinders move; None = all
    obs_axes: list | None = None  # per-dynamic-cylinder motion axis: "x", "y", or "xy".
                                   # None = all "y" (legacy lateral-only behaviour).
                                   # Must match length of dynamic_cyl_indices (or len(CYLINDERS) if that's None).

    goal_y = 2.0
    
class Crazyflie(gym.Env):
    metadata = {"render_modes": ["human", "none"]}

    def __init__(
        self,
        cfg: CrazyflieEnvCfg,
        device: str = "cuda:0",
    ):
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        super().__init__()

        self.num_envs = cfg.num_envs

        sim_cfg = sim_utils.SimulationCfg(dt=cfg.dt, device = cfg.device)
        self._sim = sim_utils.SimulationContext(sim_cfg)
        self._sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])

        scene_cfg = CrazyflieSceneCfg(num_envs=cfg.num_envs, env_spacing=cfg.env_spacing)
        scene_cfg.crazyflie.spawn.rigid_props.disable_gravity = True
        _spawn_x = scene_cfg.crazyflie.init_state.pos[0]
        scene_cfg.crazyflie.init_state = scene_cfg.crazyflie.init_state.replace(pos=(_spawn_x, 0.0, 0.75))
        self.scene = InteractiveScene(scene_cfg)
        self.robot = self.scene["crazyflie"]
        self.cam = self.scene["FPV_CAMERA_CFG"]

        self._sim.reset()
        self.robot.update(self._sim.get_physics_dt())

        # Save initial state
        self.initial_root_state = self.robot.data.root_state_w.clone()  # (N,13) world
        self.env_origins = self.initial_root_state[:, 0:3].clone().to(self.device)

        self.num_envs = self.robot.num_instances

        print("prop names:", self.robot.body_names)

        # propeller joints -- visual/dynamics spin only, see step() below
        self.propeller_indices = self.robot.find_joints(list(PROPELLER_JOINT_NAMES))[0]
        self._rotor_target_vel = torch.zeros(self.num_envs, self.robot.num_joints, device=self.device)

        self.count = 2  # see cfg.dt's comment -- count*dt = 2 * 1/50 = 0.04s, matching
                        # execise_01_c.py's decimation=2 @ sim.dt=1/50 control cadence
        self._cyl_phys_radius = 0.15   # matches CylinderCfg radius in scene

        # episode bookkeeping
        self.success_acc = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.fell_acc    = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # dynamic obstacle state (disabled unless cfg.dynamic_obstacles=True)
        self._dynamic_obs = cfg.dynamic_obstacles
        if self._dynamic_obs:
            dyn_idx = list(cfg.dynamic_cyl_indices) if cfg.dynamic_cyl_indices is not None \
                      else list(range(len(CYLINDERS)))
            self._dyn_indices = dyn_idx
            self._dyn_cyls   = [self.scene[f"cyl_{i:02d}"] for i in dyn_idx]
            self._cyl_x0     = [float(CYLINDERS[i][0]) for i in dyn_idx]
            self._cyl_y0     = [float(CYLINDERS[i][1]) for i in dyn_idx]
            self._cyl_z0     = [float(CYLINDERS[i][2]) for i in dyn_idx]
            self._obs_phases  = [2.0 * math.pi * k / len(dyn_idx) for k in range(len(dyn_idx))]
            self._obs_amplitude = cfg.obs_amplitude
            self._obs_frequency = cfg.obs_frequency

            if cfg.obs_axes is None:
                axes = ["y"] * len(dyn_idx)
            elif len(cfg.obs_axes) == 1:
                axes = list(cfg.obs_axes) * len(dyn_idx)   # broadcast single axis to all
            else:
                axes = list(cfg.obs_axes)
            if len(axes) != len(dyn_idx):
                raise ValueError(
                    f"obs_axes length ({len(axes)}) must match dynamic_cyl_indices length ({len(dyn_idx)})"
                )
            self._obs_axes = axes
            # Keep oscillation within the playable corridor regardless of axis.
            self._x_clamp = (0.3 , CORRIDOR_LENGTH - 0.3 )
            self._y_clamp = (-0.85, 0.85)
        self._obs_t = 0.0

    # ------------------------------------------------------------------
    # Dynamic obstacle helpers
    # ------------------------------------------------------------------
    def _oscillate(self, axis: str, x0: float, y0: float, delta: float) -> tuple:
        """Apply a sinusoidal offset `delta` to x and/or y depending on `axis`
        ("x", "y", or "xy"), clamped to stay inside the corridor."""
        new_x, new_y = x0, y0
        if "x" in axis:
            new_x = max(self._x_clamp[0], min(self._x_clamp[1], x0 + delta))
        if "y" in axis:
            new_y = max(self._y_clamp[0], min(self._y_clamp[1], y0 + delta))
        return new_x, new_y

    def _step_dynamic_obstacles(self, dt: float) -> None:
        """Sinusoidal oscillation for all kinematic cylinders, along the axis
        configured per-cylinder via obs_axes. Called every physics sub-step
        inside step() when dynamic_obstacles=True."""
        for i, cyl_obj in enumerate(self._dyn_cyls):
            delta = self._obs_amplitude * math.sin(
                2.0 * math.pi * self._obs_frequency * self._obs_t + self._obs_phases[i]
            )
            new_x, new_y = self._oscillate(self._obs_axes[i], self._cyl_x0[i], self._cyl_y0[i], delta)
            pose = torch.zeros(self.num_envs, 7, device=self.device)
            pose[:, 0] = new_x
            pose[:, 1] = new_y
            pose[:, 2] = self._cyl_z0[i]
            pose[:, 3] = 1.0  # quaternion w=1 (identity rotation)
            cyl_obj.write_root_pose_to_sim(pose)
        self._obs_t += dt

    def _reset_dynamic_obstacles(self) -> None:
        """Return all kinematic cylinders to their rest positions."""
        for i, cyl_obj in enumerate(self._dyn_cyls):
            pose = torch.zeros(self.num_envs, 7, device=self.device)
            pose[:, 0] = self._cyl_x0[i]
            pose[:, 1] = self._cyl_y0[i]
            pose[:, 2] = self._cyl_z0[i]
            pose[:, 3] = 1.0
            cyl_obj.write_root_pose_to_sim(pose)


    def get_cylinder_positions(self) -> list:
        """Returns current (x, y) of every cylinder.
        Static cylinders return their rest position; only dynamic ones oscillate."""
        if not self._dynamic_obs:
            return [(float(CYLINDERS[i][0]), float(CYLINDERS[i][1]))
                    for i in range(len(CYLINDERS))]
        dyn_set = {idx: k for k, idx in enumerate(self._dyn_indices)}
        positions = []
        for i in range(len(CYLINDERS)):
            x0 = float(CYLINDERS[i][0])
            y0 = float(CYLINDERS[i][1])
            if i in dyn_set:
                k = dyn_set[i]
                delta = self._obs_amplitude * math.sin(
                    2.0 * math.pi * self._obs_frequency * self._obs_t + self._obs_phases[k]
                )
                x0, y0 = self._oscillate(self._obs_axes[k], x0, y0, delta)
            positions.append((x0, y0))
        return positions

    def predict_cylinder_positions(self, offsets: list) -> dict:
        """Exact future (x, y) for every dynamic cylinder at t = self._obs_t + offset,
        for each offset in `offsets`. Uses the same closed-form sinusoid as
        get_cylinder_positions()/_step_dynamic_obstacles(), just evaluated ahead of
        time — the motion law is deterministic, so this is exact, not an estimate.

        Returns {cylinder_index: [(x, y), ...]} (one entry per offset, same order),
        keyed by the index into CYLINDERS. Static cylinders are omitted (callers
        should keep using their fixed rest position for those).
        """
        if not self._dynamic_obs:
            return {}
        preds = {}
        for k, idx in enumerate(self._dyn_indices):
            x0 = self._cyl_x0[k]
            y0 = self._cyl_y0[k]
            traj = []
            for off in offsets:
                t = self._obs_t + off
                delta = self._obs_amplitude * math.sin(
                    2.0 * math.pi * self._obs_frequency * t + self._obs_phases[k]
                )
                traj.append(self._oscillate(self._obs_axes[k], x0, y0, delta))
            preds[idx] = traj
        return preds

    def reset(self, *, seed: int | None = None):
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # reset joints
        joint_pos, joint_vel = self.robot.data.default_joint_pos, self.robot.data.default_joint_vel
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel)

        # reset root pose/vel to initial
        self.robot.write_root_pose_to_sim(self.initial_root_state[:, :7])
        zero_root_vel = torch.zeros_like(self.initial_root_state[:, 7:])
        self.robot.write_root_velocity_to_sim(zero_root_vel)
        self.robot.reset()

        self.success_acc = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.fell_acc = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # reset dynamic obstacles to rest positions
        if self._dynamic_obs:
            self._obs_t = 0.0
            self._reset_dynamic_obstacles()
       
        # one sim step to settle
        self._sim.step()
        self.robot.update(self._sim.get_physics_dt())
        self.scene.update(self._sim.get_physics_dt()) 

        print("Environment reset")
        return self.get_observation()
    
    def step(self, action):

        act = torch.as_tensor(action, device=self.device, dtype=torch.float32)
        if self._dynamic_obs:
            self._step_dynamic_obstacles(self._sim.get_physics_dt())
        _cyl_list = self.get_cylinder_positions()   # current positions (dynamic or static)
        if _cyl_list:
            _cyl_xy = torch.tensor(
                [[p[0], p[1]] for p in _cyl_list], dtype=torch.float32, device=self.device
            )  # (N_cyl, 2)
            _cyl_collision_r = self._cyl_phys_radius + self.cfg.drone_radius
        else:
            _cyl_xy = None

        lin_vel_b = torch.stack([act[:, 0], torch.zeros_like(act[:, 0]), act[:, 1]], dim=1)
        ang_vel_b = torch.stack(
            [torch.zeros_like(act[:, 2]), torch.zeros_like(act[:, 2]), act[:, 2]], dim=1
        )
        for idx, vel in zip(self.propeller_indices, PROPELLER_VELOCITY_TARGETS):
            self._rotor_target_vel[:, idx] = vel

        quat = self.robot.data.root_quat_w
        vel_cmd = torch.cat(
            [quat_apply(quat, lin_vel_b), quat_apply(quat, ang_vel_b)], dim=1
        )
        self.robot.write_root_velocity_to_sim(vel_cmd)
        self.robot.set_joint_velocity_target(self._rotor_target_vel)

        self.robot.write_data_to_sim()
        self._sim.step()
        self.robot.update(self._sim.get_physics_dt())
        self.scene.update(self._sim.get_physics_dt())

        pos_world = self._pos_world()
        x = pos_world[:, 0]
        y = pos_world[:, 1]
        z = pos_world[:, 2]

        # goal
        self.success_acc |= (x >= self.cfg.gate_x_max)

        # floor / walls / ceiling
        self.fell_acc |= (z < self.cfg.min_z)                   # hit ground
        self.fell_acc |= (y >=  self.cfg.goal_y)                 # hit left wall
        self.fell_acc |= (y <= -self.cfg.goal_y)                 # hit right wall
        self.fell_acc |= (z > self.cfg.max_z)                    # hit ceiling

        # cylinder collision: drone centre within (cyl_radius + drone_radius) of any cyl
        if _cyl_xy is not None:
            drone_xy = pos_world[:, :2].unsqueeze(1)             # (N_env, 1, 2)
            dists_cyl = torch.norm(drone_xy - _cyl_xy.unsqueeze(0), dim=-1)  # (N_env, N_cyl)
            self.fell_acc |= (dists_cyl < _cyl_collision_r).any(dim=-1)

        done = self.success_acc | self.fell_acc

        success = self.success_acc & ~self.fell_acc
        fell = self.fell_acc
        
        obs = self.get_observation()
        reward = self.get_reward()

        info = {
            "success": success.detach().cpu().numpy(),
            "fell": fell.detach().cpu().numpy(),
            "pos": self._pos_world().detach().cpu().numpy(),
        }
        
        return obs, reward, done.detach().cpu().numpy(), info
    
    def get_rgb(self):
        rgb = self.cam.data.output["rgb"]  # (N,H,W,3), float [0,1]
        rgb = rgb[0].detach().cpu().numpy().astype(np.uint8)  # convert to uint8 [0,255]
        self._last_rgb = rgb
        return rgb

    def get_depth(self):
        """Depth channel for the FPV camera. Requires the camera to be configured
        with "distance_to_camera" in data_types (see USE_DEPTH in crazyflie_env_cfg.py)."""
        if "distance_to_camera" not in self.cam.data.output:
            raise RuntimeError(
                "FPV camera was not configured for depth output. "
                "Set USE_DEPTH=True in crazyflie_env_cfg.py to enable distance_to_camera."
            )
        depth = self.cam.data.output["distance_to_camera"]  # (N,H,W,1), float meters
        depth = depth[0].detach().cpu().numpy().astype(np.float32)
        non_finite = ~np.isfinite(depth)
        if non_finite.any():
            depth[non_finite] = DEPTH_FAR
        self._last_depth = depth
        return depth

    def get_rgbd(self):
        """Concatenated RGB + depth for the FPV camera, as (H,W,4): uint8 RGB, float32 depth."""
        rgb = self.get_rgb()
        depth = self.get_depth()
        return rgb, depth
    
    def get_observation(self) -> np.ndarray:
        root = self.robot.data.root_state_w  # (N,13)
        pos = root[:, 0:3]
        obs = torch.cat([pos], dim=-1)
        return obs.detach().cpu().numpy()

    def get_reward(self):
        ...
    
    def render(self):
        return None

    def close(self):
        print("Environment close")
        self._sim.stop()
        global simulation_app
        simulation_app.close()

    def ee_pose_world(self) -> np.ndarray:
        """
        For a drone, treat 'ee pose' as the root pose.
        returns array (num_envs, 7) = [x,y,z, qx,qy,qz,qw] in world
        """
        root = self.robot.data.root_state_w
        pose = root[:, 0:7]
        return pose.detach().cpu().numpy()
    
    def robot_state(self) -> np.ndarray:
        """
        returns (num_envs, 13) where position is localized by env_origins.
        """
        root = self.robot.data.root_state_w.clone()
        root[:, 0:3] -= self.env_origins
        return root.detach().cpu().numpy()
    
    def _pos_world(self) -> torch.Tensor:
        root = self.robot.data.root_state_w
        return root[:, 0:3]

