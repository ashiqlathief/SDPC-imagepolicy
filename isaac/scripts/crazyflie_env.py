from dataclasses import dataclass
from time import perf_counter
import argparse
import os
from isaaclab.app import AppLauncher
from warp import pos
_parser = argparse.ArgumentParser(add_help=False)
AppLauncher.add_app_launcher_args(_parser)
_app_args, _ = _parser.parse_known_args()
# Headless by default (every existing caller relies on this); set
# CRAZYFLIE_ENV_HEADLESS=0 before importing this module to pop up the actual
# Isaac Sim GUI instead -- see --source isaac in depth_camera_live_test.py.
_app_args.headless = os.environ.get("CRAZYFLIE_ENV_HEADLESS", "0") != "0"
_app_args.enable_cameras = True
app_launcher = AppLauncher(_app_args)
simulation_app = app_launcher.app

import torch
import numpy as np
import torch
import gymnasium as gym
from typing import Tuple
import math

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene
from isaaclab.scene import InteractiveScene


from .crazyflie_env_cfg import (CrazyflieSceneCfg, CYLINDERS, CORRIDOR_LENGTH, DEPTH_FAR,
                                _SPHERES_XYZ as SPHERES_XYZ, SPHERE_RADIUS)

def quat_to_yaw(quat_xyzw: torch.Tensor) -> torch.Tensor:
    """quat_xyzw: (...,4) -> yaw (...,)"""
    x, y, z, w = quat_xyzw.unbind(-1)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return torch.atan2(siny_cosp, cosy_cosp)

def quat_to_euler_xyzw(q_xyzw: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert (x,y,z,w) quat to roll,pitch,yaw. q_xyzw shape (N,4)."""
    x, y, z, w = q_xyzw.unbind(-1)

    t0 = 2 * (w * x + y * z)
    t1 = 1 - 2 * (x * x + y * y)
    roll = torch.atan2(t0, t1)

    t2 = 2 * (w * y - z * x)
    t2 = torch.clamp(t2, -1.0, 1.0)
    pitch = torch.asin(t2)

    t3 = 2 * (w * z + x * y)
    t4 = 1 - 2 * (y * y + z * z)
    yaw = torch.atan2(t3, t4)

    return roll, pitch, yaw

def controller_motor_forces(
    root_state,
    target_pos,
    env_origins,
    sim_dt,
    robot_mass,
    gravity,
    num_envs,
    sim,
):
    
    device = sim.device

    if (not hasattr(controller_motor_forces, "vel_int")) or (controller_motor_forces.vel_int.shape[0] != num_envs):
        # ========= Controller gains (to tune) =========
        controller_motor_forces.Kp_pos = torch.tensor([0.8, 0.8, 1.5], device=device)
        controller_motor_forces.Kd_pos = torch.tensor([0.8, 0.8, 0.8], device=device)

        controller_motor_forces.Kp_vel = torch.tensor([1.0, 1.0, 2.0], device=device)
        controller_motor_forces.Ki_vel = torch.tensor([0.1, 0.1, 0.3], device=device)
        controller_motor_forces.Kd_vel = torch.tensor([0.2, 0.2, 0.5], device=device)

        controller_motor_forces.Kp_att = torch.tensor([4.0, 4.0, 1.5], device=device)
        controller_motor_forces.Kd_att = torch.tensor([0.5, 0.5, 0.3], device=device)
        controller_motor_forces.Ki_att = torch.tensor([0.1, 0.1, 0.0], device=device)

        # Mixer scaling
        controller_motor_forces.MIX_SCALE = 0.01

        # ========= Integrator & previous-error states =========
        controller_motor_forces.vel_int = torch.zeros(num_envs, 3, device=device)
        controller_motor_forces.att_int = torch.zeros(num_envs, 3, device=device)
        controller_motor_forces.prev_vel_err = torch.zeros(num_envs, 3, device=device)
        controller_motor_forces.prev_att_err = torch.zeros(num_envs, 3, device=device)
    
    Kp_pos = controller_motor_forces.Kp_pos
    Kd_pos = controller_motor_forces.Kd_pos
    Kp_vel = controller_motor_forces.Kp_vel
    Ki_vel = controller_motor_forces.Ki_vel
    Kd_vel = controller_motor_forces.Kd_vel
    Kp_att = controller_motor_forces.Kp_att
    Ki_att = controller_motor_forces.Ki_att
    Kd_att = controller_motor_forces.Kd_att
    MIX_SCALE = controller_motor_forces.MIX_SCALE

    vel_int = controller_motor_forces.vel_int
    att_int = controller_motor_forces.att_int
    prev_vel_err = controller_motor_forces.prev_vel_err
    prev_att_err = controller_motor_forces.prev_att_err

    #--- unpack state ---
    pos    = root_state[:, 0:3]
    quat   = root_state[:, 3:7]     # (x,y,z,w) in your comment, but your swap below assumes (w,x,y,z)
    w = quat[:, 0]
    x = quat[:, 1]
    y = quat[:, 2]
    z = quat[:, 3]
    quat_xyzw = torch.stack([x, y, z, w], dim=-1)
    linvel   = root_state[:, 7:10]

    # ---- discrete segment tracking (start -> goal) ----
    SEG_STEPS = 0          # more = finer discretization
    REACH_TOL = 0.03        # when we consider a waypoint reached (meters)

    # init once (or if num_envs changes)
    if (not hasattr(controller_motor_forces, "start_pos")) or (controller_motor_forces.start_pos.shape[0] != num_envs):
        controller_motor_forces.start_pos = pos.clone()
        controller_motor_forces.step_idx  = torch.zeros(num_envs, 1, device=pos.device)
    
    if SEG_STEPS <= 0:
        target_cmd = target_pos
    else:
        # compute current alpha and waypoint
        alpha = controller_motor_forces.step_idx / float(SEG_STEPS)     # (N,1)
        alpha = torch.clamp(alpha, 0.0, 1.0)

        target_cmd = (1.0 - alpha) * controller_motor_forces.start_pos + alpha * target_pos  # (N,3)

        # advance only when drone is close to current waypoint
        wp_dist = torch.linalg.norm(target_cmd - pos, dim=-1, keepdim=True)  # (N,1)
        advance = (wp_dist < REACH_TOL).float()

        controller_motor_forces.step_idx = torch.clamp(
            controller_motor_forces.step_idx + advance,
            0.0,
            float(SEG_STEPS),
        )

        # after the last step, just use final target
        target_cmd = torch.where(alpha >= 1.0, target_pos, target_cmd)  
    controller_motor_forces.last_target_cmd = target_cmd

    # ---------------------------------------------------
    # 1) Outer loop: Position → desired velocity
    # ---------------------------------------------------
    pos_err = target_cmd - pos
    vel_des = Kp_pos * pos_err - Kd_pos * linvel
    vel_des = torch.clamp(vel_des, -0.5, 0.5)

    # ---------------------------------------------------
    # 2) Inner loop: Velocity PID → desired acc / attitude
    # ---------------------------------------------------
    vel_err = vel_des - linvel
    vel_int.add_(vel_err * sim_dt)

    vel_der = (vel_err - prev_vel_err) / sim_dt
    prev_vel_err.copy_(vel_err)

    acc_cmd = Kp_vel * vel_err + Ki_vel * vel_int + Kd_vel * vel_der
    acc_cmd_xy = acc_cmd[:, :2]
    acc_cmd_z  = acc_cmd[:, 2]

    theta_des = acc_cmd_xy[:, 0] / gravity
    phi_des   = -acc_cmd_xy[:, 1] / gravity
    yaw_des   = torch.zeros_like(phi_des)

    # ---------------------------------------------------
    # 3) Altitude: use vertical acc_cmd_z to adjust thrust
    # ---------------------------------------------------
    T_des = robot_mass * (gravity + acc_cmd_z)
    max_T_des = 2.0 * robot_mass * gravity
    min_T_des = torch.zeros_like(T_des)
    T_des = torch.clamp(T_des, min=min_T_des, max=max_T_des)

    roll, pitch, yaw = quat_to_euler_xyzw(quat_xyzw)

    att_err_raw = torch.stack([phi_des - roll, theta_des - pitch, yaw_des - yaw], dim=-1)
    att_err = (att_err_raw + math.pi) % (2 * math.pi) - math.pi

    att_int.add_(att_err * sim_dt)

    att_der = (att_err - prev_att_err) / sim_dt
    prev_att_err.copy_(att_err)

    tau_cmd = Kp_att * att_err + Ki_att * att_int + Kd_att * att_der
    tau_roll  = tau_cmd[:, 0]
    tau_pitch = tau_cmd[:, 1]
    tau_yaw   = tau_cmd[:, 2]

    # ---------------------------------------------------
    # 4) Mixer: thrust + torques → 4 motor forces
    # ---------------------------------------------------
    base_thrust = T_des.view(-1, 1) / 4.0

    d_roll  = MIX_SCALE * tau_roll.view(-1, 1)
    d_pitch = MIX_SCALE * tau_pitch.view(-1, 1)
    d_yaw   = MIX_SCALE * tau_yaw.view(-1, 1)

    f1 = base_thrust - d_roll - d_pitch - d_yaw   # front-left
    f2 = base_thrust - d_roll + d_pitch + d_yaw   # rear-left
    f3 = base_thrust + d_roll + d_pitch - d_yaw   # rear-right
    f4 = base_thrust + d_roll - d_pitch + d_yaw   # front-right

    motor_forces_z = torch.cat([f1, f2, f3, f4], dim=-1)
    
    hover_thrust = robot_mass * gravity / 4.0    # per rotor hover thrust

    # max per motor per env
    max_motor_force = 2.0 * hover_thrust
    max_motor_force = max_motor_force.view(-1, 1).expand_as(motor_forces_z)
    min_motor_force = torch.zeros_like(motor_forces_z)

    motor_forces_z = torch.clamp(motor_forces_z, min=min_motor_force, max=max_motor_force)

    return motor_forces_z

@dataclass
class CrazyflieEnvCfg:
    num_envs: int = 1
    env_spacing: float = 2.0
    dt: float = 0.005
    device: str = "cuda:0"
    gate_x_min: float = 3.95
    gate_x_max: float =  4.0
    gate_y: float = 1.0
    gate_y_tol: float = 0.005   # how close to y=1 counts
    min_z: float = 0.02         # if z < 0.2 -> fail
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
    floating_spheres: bool = False     # enable planet-like floating sphere obstacles
    sphere_amplitude: float = 0.20     # per-axis sinusoid amplitude for spheres (m)
    sphere_frequency: float = 0.20     # base oscillation frequency for spheres (Hz)

    goal_y = 1.0
    goal_z = 1.0
    
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
        
        self._TableTopSceneCfg = CrazyflieSceneCfg
        self.num_envs = cfg.num_envs

        # render_interval=render_interval
    
        sim_cfg = sim_utils.SimulationCfg(dt=cfg.dt, device = cfg.device)
        self._sim = sim_utils.SimulationContext(sim_cfg)
        self._sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])

        scene_cfg = CrazyflieSceneCfg(num_envs=cfg.num_envs, env_spacing=cfg.env_spacing)
        self.scene = InteractiveScene(scene_cfg)
        self.robot = self.scene["crazyflie"]
        self.cam = self.scene["FPV_CAMERA_CFG"]
        self.chase_cam = self.scene["CHASE_CAMERA_CFG"]
        self.spectator_cam = self.scene["SPECTATOR_CAMERA_CFG"]

        self._sim.reset()
        self.robot.update(self._sim.get_physics_dt())

        # Save initial state
        self.initial_root_state = self.robot.data.root_state_w.clone()  # (N,13) world
        self.env_origins = self.initial_root_state[:, 0:3].clone().to(self.device)

        self.num_envs = self.robot.num_instances


        # prop_body_ids = robot.find_bodies("m.*_prop")[0]
        # robot_mass = robot.root_physx_view.get_masses().sum()
        # gravity = torch.tensor(sim.cfg.gravity, device=sim.device).norm()

        # propeller bodies
        self.prop_body_ids = self.robot.find_bodies("m.*_prop")[0]
        print("prop_body_ids:", self.prop_body_ids)
        print("prop names:", self.robot.body_names)

        # mass + hover thrust
        self.robot_mass = self.robot.root_physx_view.get_masses().sum() #remove[0] if you want per-env mass
        self.gravity = torch.tensor(self._sim.cfg.gravity, device=self.device).norm()
        self.hover_thrust = self.robot_mass * self.gravity / 4.0  # per rotor
        self.count = 100
        self._cyl_phys_radius = 0.06   # matches CylinderCfg radius in scene

        # episode bookkeeping
        self.success_count = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
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
            self._x_clamp = (0.3, CORRIDOR_LENGTH - 0.3)
            self._y_clamp = (-0.85, 0.85)
        self._obs_t = 0.0

        # ── floating sphere state (always track objects; park when inactive) ──
        self._floating_spheres = cfg.floating_spheres
        if len(SPHERES_XYZ) > 0:
            N = len(SPHERES_XYZ)
            self._sph_objs  = [self.scene[f"sph_{i:02d}"] for i in range(N)]
            self._sph_x0    = [float(p[0]) for p in SPHERES_XYZ]
            self._sph_y0    = [float(p[1]) for p in SPHERES_XYZ]
            self._sph_z0    = [float(p[2]) for p in SPHERES_XYZ]
            self._sph_ph_x  = [2.0 * math.pi * k / N                   for k in range(N)]
            self._sph_ph_y  = [2.0 * math.pi * k / N + math.pi / 3     for k in range(N)]
            self._sph_ph_z  = [2.0 * math.pi * k / N + 2*math.pi / 3   for k in range(N)]
            self._sph_amp   = cfg.sphere_amplitude
            self._sph_freq  = cfg.sphere_frequency
            self._sph_x_clamp = (0.3, CORRIDOR_LENGTH - 0.3)
            self._sph_y_clamp = (-0.85, 0.85)
            self._sph_z_clamp = (0.10, 1.80)
            # Park spheres immediately if not active (they are in the corridor by default)
            if not self._floating_spheres:
                self._park_spheres()
        else:
            self._sph_objs = []


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

    # ------------------------------------------------------------------
    # Floating sphere helpers
    # ------------------------------------------------------------------
    def _step_floating_spheres(self, dt: float) -> None:
        """Full 3D sinusoidal motion for floating sphere obstacles.
        Each axis oscillates at a different frequency multiple so the
        combined path never exactly repeats within a normal episode."""
        t = self._obs_t
        A = self._sph_amp
        f = self._sph_freq
        for i, sph_obj in enumerate(self._sph_objs):
            dx = A * math.sin(2.0 * math.pi * f * 0.90 * t + self._sph_ph_x[i])
            dy = A * math.sin(2.0 * math.pi * f * 1.30 * t + self._sph_ph_y[i])
            dz = A * math.sin(2.0 * math.pi * f * 0.70 * t + self._sph_ph_z[i])
            new_x = max(self._sph_x_clamp[0], min(self._sph_x_clamp[1], self._sph_x0[i] + dx))
            new_y = max(self._sph_y_clamp[0], min(self._sph_y_clamp[1], self._sph_y0[i] + dy))
            new_z = max(self._sph_z_clamp[0], min(self._sph_z_clamp[1], self._sph_z0[i] + dz))
            pose = torch.zeros(self.num_envs, 7, device=self.device)
            pose[:, 0] = new_x
            pose[:, 1] = new_y
            pose[:, 2] = new_z
            pose[:, 3] = 1.0
            sph_obj.write_root_pose_to_sim(pose)

    def _reset_floating_spheres(self) -> None:
        """Return all floating spheres to their rest positions."""
        for i, sph_obj in enumerate(self._sph_objs):
            pose = torch.zeros(self.num_envs, 7, device=self.device)
            pose[:, 0] = self._sph_x0[i]
            pose[:, 1] = self._sph_y0[i]
            pose[:, 2] = self._sph_z0[i]
            pose[:, 3] = 1.0
            sph_obj.write_root_pose_to_sim(pose)

    def _park_spheres(self) -> None:
        """Move all sphere objects to z=5 m (above play area) so they don't collide."""
        for sph_obj in self._sph_objs:
            pose = torch.zeros(self.num_envs, 7, device=self.device)
            pose[:, 0] = 0.0
            pose[:, 1] = 0.0
            pose[:, 2] = 5.0
            pose[:, 3] = 1.0
            sph_obj.write_root_pose_to_sim(pose)

    def get_sphere_positions(self) -> list:
        """Returns current (x, y, z) of every floating sphere.
        If floating_spheres is disabled returns rest positions."""
        if not self._floating_spheres or not SPHERES_XYZ:
            return [(float(p[0]), float(p[1]), float(p[2])) for p in SPHERES_XYZ]
        t = self._obs_t
        A = self._sph_amp
        f = self._sph_freq
        positions = []
        for i in range(len(SPHERES_XYZ)):
            dx = A * math.sin(2.0 * math.pi * f * 0.90 * t + self._sph_ph_x[i])
            dy = A * math.sin(2.0 * math.pi * f * 1.30 * t + self._sph_ph_y[i])
            dz = A * math.sin(2.0 * math.pi * f * 0.70 * t + self._sph_ph_z[i])
            x = max(self._sph_x_clamp[0], min(self._sph_x_clamp[1], self._sph_x0[i] + dx))
            y = max(self._sph_y_clamp[0], min(self._sph_y_clamp[1], self._sph_y0[i] + dy))
            z = max(self._sph_z_clamp[0], min(self._sph_z_clamp[1], self._sph_z0[i] + dz))
            positions.append((x, y, z))
        return positions

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
        
        self.done.zero_()
        self.success_count.zero_()
        # i = 0
        # for i in range(100):
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
        if self._floating_spheres:
            self._reset_floating_spheres()
        elif self._sph_objs:
            self._park_spheres()
        # ── Reset PID integrator state ──────────────────────────────
        if hasattr(controller_motor_forces, "vel_int"):
            controller_motor_forces.vel_int.zero_()
            controller_motor_forces.att_int.zero_()
            controller_motor_forces.prev_vel_err.zero_()
            controller_motor_forces.prev_att_err.zero_()
        if hasattr(controller_motor_forces, "start_pos"):
            controller_motor_forces.start_pos = \
                self.robot.data.root_state_w[:, :3].clone()
            controller_motor_forces.step_idx.zero_()
        # ────────────────────────────────────────────────────────────
        # one sim step to settle
        self._sim.step()
        self.robot.update(self._sim.get_physics_dt())
        self.scene.update(self._sim.get_physics_dt()) 

        print("Environment reset")
        return self.get_observation()
    
    def step(self, action):

        act = torch.as_tensor(action, device=self.device, dtype=torch.float32)
        forces = torch.zeros(self.num_envs, 4, 3, device=self.device)
        torques = torch.zeros_like(forces)
        if self._dynamic_obs:
            self._step_dynamic_obstacles(self._sim.get_physics_dt())
        if self._floating_spheres:
            self._step_floating_spheres(self._sim.get_physics_dt())

        # Pre-compute obstacle positions once per control step (cylinders don’t
        # move during the inner sub-step loop, so this is correct and avoids
        # rebuilding the tensor 100× per step).
        _cyl_list = self.get_cylinder_positions()   # current positions (dynamic or static)
        if _cyl_list:
            _cyl_xy = torch.tensor(
                [[p[0], p[1]] for p in _cyl_list], dtype=torch.float32, device=self.device
            )  # (N_cyl, 2)
            _cyl_collision_r = self._cyl_phys_radius + self.cfg.drone_radius
        else:
            _cyl_xy = None

        if self._floating_spheres and self._sph_objs:
            _sph_list = self.get_sphere_positions()
            _sph_xyz = torch.tensor(_sph_list, dtype=torch.float32, device=self.device)  # (N_sph, 3)
            _sph_collision_r = SPHERE_RADIUS + self.cfg.drone_radius
        else:
            _sph_xyz = None

        for _ in range(self.count):
            forces[..., 2] = controller_motor_forces(
                self.robot.data.root_state_w,
                act,
                self.env_origins,
                self._sim.get_physics_dt(),
                self.robot_mass,
                self.gravity,
                self.num_envs,
                self._sim,
            )
            self.robot.set_external_force_and_torque(forces, torques, body_ids=self.prop_body_ids)

            self.robot.write_data_to_sim()
            self._sim.step()
            self.robot.update(self._sim.get_physics_dt())
            self.scene.update(self._sim.get_physics_dt()) #camera won’t refresh.

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
            self.fell_acc |= (z > 1.0)                               # hit ceiling

            # cylinder collision: drone centre within (cyl_radius + drone_radius) of any cyl
            if _cyl_xy is not None:
                drone_xy = pos_world[:, :2].unsqueeze(1)             # (N_env, 1, 2)
                dists_cyl = torch.norm(drone_xy - _cyl_xy.unsqueeze(0), dim=-1)  # (N_env, N_cyl)
                self.fell_acc |= (dists_cyl < _cyl_collision_r).any(dim=-1)

            # floating sphere collision
            if _sph_xyz is not None:
                drone_xyz = pos_world.unsqueeze(1)                   # (N_env, 1, 3)
                dists_sph = torch.norm(drone_xyz - _sph_xyz.unsqueeze(0), dim=-1)  # (N_env, N_sph)
                self.fell_acc |= (dists_sph < _sph_collision_r).any(dim=-1)

            done = self.success_acc | self.fell_acc
            if done.all():
                break

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
        # Rays that miss everything within clipping_range come back as inf (occasionally
        # nan); replace with DEPTH_FAR so raw saved data never carries non-finite values.
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

    def get_chase_rgb(self):
        """Third-person, drone-mounted view for presentation video; not used as policy input."""
        rgb = self.chase_cam.data.output["rgb"]  # (N,H,W,3), float [0,1]
        rgb = rgb[0].detach().cpu().numpy().astype(np.uint8)
        return rgb

    def get_spectator_rgb(self):
        """Fixed, environment-mounted outside view; not used as policy input."""
        rgb = self.spectator_cam.data.output["rgb"]  # (N,H,W,3), float [0,1]
        rgb = rgb[0].detach().cpu().numpy().astype(np.uint8)
        return rgb
    
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
        self.simulation_app.close()

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

