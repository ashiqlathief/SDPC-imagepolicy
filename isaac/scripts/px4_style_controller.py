# Copyright (c) 2026 — thesis project (UPB DPCC)
#
# PX4-style low-level flight controller, ported from the real PX4-Autopilot
# firmware (BSD-3-Clause, Copyright (c) 2012-2025 PX4 Development Team) into
# a vectorized torch controller usable inside Isaac Lab.
#
# What's ported faithfully (same equations as the real firmware):
#   - Rate controller: P + I (with adaptive anti-windup) + D-on-measured-angular-
#     acceleration + rate feedforward, from src/lib/rate_control/rate_control.cpp.
#   - Attitude controller: quaternion-error P-control with reduced-yaw-priority
#     (roll/pitch always get full authority, yaw catches up separately), from
#     src/modules/mc_att_control/AttitudeControl/AttitudeControl.cpp — the
#     `update()` step only.
#   - Thrust-vector construction: proper `acc_cmd + gravity -> unit thrust
#     direction` (the same physical logic real quadrotor controllers, PX4
#     included, use), replacing the small-angle theta/phi approximation in
#     this repo's existing `controller_motor_forces` (crazyflie_env.py).
#   - Command interface: body-frame velocity + yaw-rate, rotated into world
#     frame via the drone's current yaw — same as
#     ROS2-PX4_Drone_Teleoperation_Using_Joystick/px4_offboard/velocity_offboard_control.py
#
# What's deliberately simplified (not ported):
#   - PX4's 2nd-order critically-damped attitude *reference model*
#     (`propagateReferenceModel`) is skipped — the attitude setpoint is used
#     directly, with no extra smoothing/feedforward shaping. That's the
#     "tractable subset" of AttitudeControl.cpp: the P-controller, not the
#     reference-model machinery around it.
#   - Gains are NOT PX4's raw firmware constants (those are tuned for PX4's
#     normalized actuator-output units and a specific real airframe's mass/
#     inertia — they don't transfer numerically to this sim's N and N*m
#     units). Only the *structure* (P+I+D+FF, anti-windup) is ported
#     faithfully; magnitudes are starting points sized to match this repo's
#     existing working controller (`controller_motor_forces` in
#     crazyflie_env.py / quadcopter.py) and will need empirical retuning.
#   - Mixer: same per-motor sign convention (f1..f4) as the existing
#     controller (matches the real Crazyflie cf2x rotor layout already
#     working in this repo), but with proportional desaturation instead of
#     an independent per-motor clamp — closer in spirit to PX4's
#     control_allocator, without porting its full pseudo-inverse allocation.

from __future__ import annotations

import math
import torch

from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_conjugate, quat_mul


# =============================================================================
# Quaternion helpers (all quaternions are (w, x, y, z), matching IsaacLab's
# root_state_w layout and isaaclab.utils.math's convention)
# =============================================================================
def quat_canonicalize(q: torch.Tensor) -> torch.Tensor:
    """Flip sign so w >= 0 (shortest-path convention), matching PX4's Quatf::canonical()."""
    sign = torch.where(q[..., 0:1] < 0.0, -torch.ones_like(q[..., 0:1]), torch.ones_like(q[..., 0:1]))
    return q * sign


def quat_from_two_vectors(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Shortest-arc quaternion rotating unit vector `a` onto unit vector `b`.

    Equivalent to PX4's matrix-library `Quatf(a, b)` two-vector constructor
    (used in AttitudeControl.cpp to build the tilt-only "reduced" setpoint).
    """
    dot = (a * b).sum(-1, keepdim=True).clamp(-1.0, 1.0)
    cross = torch.linalg.cross(a, b, dim=-1)
    w = 1.0 + dot

    # near-antipodal (a ~= -b): the cross product degenerates to ~0, so pick an
    # arbitrary axis perpendicular to `a` for the 180 deg rotation instead.
    degenerate = (w.squeeze(-1) < eps)
    if degenerate.any():
        world_x = torch.zeros_like(a)
        world_x[..., 0] = 1.0
        world_y = torch.zeros_like(a)
        world_y[..., 1] = 1.0
        use_y = a[..., 0:1].abs() > 0.9
        perp_axis = torch.where(use_y, world_y, world_x)
        perp = torch.linalg.cross(a, perp_axis, dim=-1)
        cross = torch.where(degenerate.unsqueeze(-1), perp, cross)
        w = torch.where(degenerate.unsqueeze(-1), torch.zeros_like(w), w)

    q = torch.cat([w, cross], dim=-1)
    return q / q.norm(dim=-1, keepdim=True).clamp_min(eps)


def yaw_quat_from_angle(yaw: torch.Tensor) -> torch.Tensor:
    """Pure-yaw quaternion (w, 0, 0, z) from a scalar yaw angle, shape (N,) -> (N, 4)."""
    half = yaw * 0.5
    q = torch.zeros(*yaw.shape, 4, device=yaw.device, dtype=yaw.dtype)
    q[..., 0] = torch.cos(half)
    q[..., 3] = torch.sin(half)
    return q


def yaw_from_quat(q: torch.Tensor) -> torch.Tensor:
    """Extract yaw (rad) from a (w, x, y, z) quaternion."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def body_vel_to_world(vel_cmd_b: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """Rotate a body-frame [vx, vy, vz] command into world frame using yaw only.

    Ported directly from velocity_offboard_control.py's `loop()`
    (ROS2-PX4_Drone_Teleoperation_Using_Joystick/px4_offboard/px4_offboard/
    velocity_offboard_control.py:179-183) — PX4 offboard velocity setpoints
    are inherently world-frame-with-yaw, so only yaw (not full 3D attitude)
    rotates the command; vz passes straight through.
    """
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    vx_b, vy_b, vz_b = vel_cmd_b[..., 0], vel_cmd_b[..., 1], vel_cmd_b[..., 2]
    wx = vx_b * cy - vy_b * sy
    wy = vx_b * sy + vy_b * cy
    return torch.stack([wx, wy, vz_b], dim=-1)


def position_error_to_body_vel_cmd(
    pos: torch.Tensor,
    target_pos: torch.Tensor,
    lin_vel_w: torch.Tensor,
    yaw: torch.Tensor,
    Kp_pos: torch.Tensor,
    Kd_pos: torch.Tensor,
    vel_limit: float,
) -> torch.Tensor:
    """Optional outer position loop, NOT part of the PX4 port above (PX4's own
    mc_pos_control position loop wasn't ported — this is a plain P-D
    substitute). Lets a position-target script (e.g. an autonomous
    fly-to-random-target demo) drive PX4StyleController.update(), which
    otherwise only accepts a velocity+yawrate command.

    Computes the desired velocity in world frame (P-D on position error,
    clamped to `vel_limit`), then rotates it into body frame by -yaw so
    that PX4StyleController.update()'s own body->world rotation (via
    body_vel_to_world) round-trips back to the intended world-frame vector.
    """
    pos_err = target_pos - pos
    vel_des_w = Kp_pos * pos_err - Kd_pos * lin_vel_w
    vel_des_w = torch.clamp(vel_des_w, -vel_limit, vel_limit)

    cy, sy = torch.cos(yaw), torch.sin(yaw)
    vx_b = vel_des_w[:, 0] * cy + vel_des_w[:, 1] * sy
    vy_b = -vel_des_w[:, 0] * sy + vel_des_w[:, 1] * cy
    vz_b = vel_des_w[:, 2]
    return torch.stack([vx_b, vy_b, vz_b], dim=-1)


# =============================================================================
# Controller
# =============================================================================
class PX4StyleController:
    """Vectorized (per-env) velocity+yawrate -> per-rotor-force controller.

    Cascade: body-vel cmd -> world-vel cmd -> velocity PID -> desired thrust
    vector -> reduced-yaw-priority attitude P-control -> rate PID+FF -> mixer.

    Call `update()` once per physics step with a body-frame
    [vx, vy, vz, yaw_rate] command (same shape as the Twist the ROS2-PX4
    teleop repo publishes on /offboard_velocity_cmd).
    """

    def __init__(self, num_envs: int, device: torch.device | str):
        self.num_envs = num_envs
        self.device = device

        # ---- outer velocity loop (reused from this repo's existing tuned
        # controller_motor_forces gains — crazyflie_env.py / quadcopter.py) ----
        self.Kp_vel = torch.tensor([1.0, 1.0, 2.0], device=device)
        self.Ki_vel = torch.tensor([0.1, 0.1, 0.3], device=device)
        self.Kd_vel = torch.tensor([0.2, 0.2, 0.5], device=device)
        self.vel_cmd_limit = torch.tensor([1.5, 1.5, 0.75], device=device)  # m/s, safety clamp

        # ---- attitude P gains (quaternion error -> body rate setpoint) ----
        # Structure matches AttitudeControl.cpp; roll/pitch share a gain,
        # yaw is weaker (PX4's yaw_weight < 1 pattern: yaw catches up slower
        # than tilt so translation authority is never sacrificed for yaw).
        self.Kp_att = torch.tensor([4.0, 4.0, 1.5], device=device)
        self.rate_sp_limit = torch.deg2rad(torch.tensor([220.0, 220.0, 200.0], device=device))

        # ---- rate PID+FF, structure ported from rate_control.cpp ----
        self.Kp_rate = torch.tensor([0.15, 0.15, 0.20], device=device)
        self.Ki_rate = torch.tensor([0.20, 0.20, 0.20], device=device)
        self.Kd_rate = torch.tensor([0.003, 0.003, 0.0], device=device)
        self.Kff_rate = torch.tensor([0.0, 0.0, 0.0], device=device)
        self.rate_int_limit = torch.deg2rad(torch.tensor([30.0, 30.0, 30.0], device=device))

        # mixer scaling: converts the rate-controller's torque command into
        # per-motor thrust deltas (same role as MIX_SCALE in the existing
        # controller_motor_forces).
        self.mix_scale = 0.5

        self._alloc_state(num_envs, device)

    def _alloc_state(self, num_envs: int, device) -> None:
        self.vel_int = torch.zeros(num_envs, 3, device=device)
        self.prev_vel_err = torch.zeros(num_envs, 3, device=device)
        self.rate_int = torch.zeros(num_envs, 3, device=device)
        self.prev_body_rate = torch.zeros(num_envs, 3, device=device)
        self.yaw_sp = torch.zeros(num_envs, device=device)
        self._yaw_sp_initialized = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        """Zero all integrator/derivative state (call this on env reset)."""
        if env_ids is None:
            env_ids = slice(None)
        self.vel_int[env_ids] = 0.0
        self.prev_vel_err[env_ids] = 0.0
        self.rate_int[env_ids] = 0.0
        self.prev_body_rate[env_ids] = 0.0
        self._yaw_sp_initialized[env_ids] = False

    def update(
        self,
        root_state_w: torch.Tensor,
        vel_cmd_b: torch.Tensor,
        yaw_rate_cmd: torch.Tensor,
        dt: float,
        robot_mass: torch.Tensor,
        gravity: torch.Tensor,
    ) -> torch.Tensor:
        """One control tick.

        Args:
            root_state_w: (N, 13) [pos(3), quat_wxyz(4), lin_vel_w(3), ang_vel_w(3)]
            vel_cmd_b: (N, 3) desired body-frame [vx, vy, vz] (m/s)
            yaw_rate_cmd: (N,) desired yaw rate (rad/s)
            dt: physics step (s)
            robot_mass: (N,) or scalar (kg)
            gravity: scalar (m/s^2, positive magnitude)

        Returns:
            motor_forces_z: (N, 4) upward force (N) to apply at each of the
            4 propeller bodies via set_external_force_and_torque.
        """
        pos = root_state_w[:, 0:3]
        q = root_state_w[:, 3:7]
        lin_vel_w = root_state_w[:, 7:10]
        ang_vel_w = root_state_w[:, 10:13]

        vel_cmd_b = torch.clamp(vel_cmd_b, -self.vel_cmd_limit, self.vel_cmd_limit)
        yaw = yaw_from_quat(q)

        if not self._yaw_sp_initialized.all():
            uninit = ~self._yaw_sp_initialized
            self.yaw_sp[uninit] = yaw[uninit]
            self._yaw_sp_initialized[uninit] = True

        # ---------------------------------------------------------------
        # 1) body -> world velocity command, integrate yaw setpoint
        # ---------------------------------------------------------------
        vel_cmd_w = body_vel_to_world(vel_cmd_b, yaw)
        self.yaw_sp = wrap_to_pi(self.yaw_sp + yaw_rate_cmd * dt)

        # ---------------------------------------------------------------
        # 2) outer velocity PID -> desired world-frame acceleration
        # ---------------------------------------------------------------
        vel_err = vel_cmd_w - lin_vel_w
        self.vel_int = self.vel_int + vel_err * dt
        vel_der = (vel_err - self.prev_vel_err) / dt
        self.prev_vel_err = vel_err
        acc_cmd = self.Kp_vel * vel_err + self.Ki_vel * self.vel_int + self.Kd_vel * vel_der

        # ---------------------------------------------------------------
        # 3) desired thrust vector: f = acc_cmd + gravity (world frame).
        # This is the proper differential-flatness thrust direction (what
        # PX4's mc_pos_control effectively hands to the attitude controller)
        # instead of the small-angle theta=acc_x/g approximation used in
        # this repo's existing controller_motor_forces.
        # ---------------------------------------------------------------
        g_vec = torch.zeros_like(acc_cmd)
        g_vec[:, 2] = gravity
        f_des = acc_cmd + g_vec
        thrust_mag = f_des.norm(dim=-1).clamp_min(1e-6)
        z_des_world = f_des / thrust_mag.unsqueeze(-1)

        mass = robot_mass if robot_mass.dim() > 0 else robot_mass.expand(pos.shape[0])
        T_des = torch.clamp(mass * thrust_mag, min=torch.zeros_like(mass), max=2.0 * mass * gravity)

        # ---------------------------------------------------------------
        # 4) reduced-yaw-priority attitude setpoint (AttitudeControl.cpp
        # `update()`, minus the reference-model smoothing): tilt-only
        # rotation from current to desired thrust axis, then yaw applied
        # on top from the integrated yaw setpoint.
        # ---------------------------------------------------------------
        body_z = torch.zeros_like(pos)
        body_z[:, 2] = 1.0
        e_z = quat_apply(q, body_z)
        qd_red = quat_from_two_vectors(e_z, z_des_world)
        qd_red = quat_mul(qd_red, q)  # world-frame reduced setpoint (PX4: qd_red *= q)
        qd_yaw = yaw_quat_from_angle(self.yaw_sp)
        qd = quat_mul(qd_red, qd_yaw)

        # quaternion attitude error -> body rate setpoint (P control)
        qe = quat_mul(quat_conjugate(q), qd)
        qe = quat_canonicalize(qe)
        eq = 2.0 * qe[:, 1:4]
        rate_sp = self.Kp_att * eq
        rate_sp[:, 2] = rate_sp[:, 2] + yaw_rate_cmd  # feedforward the commanded yaw rate
        rate_sp = torch.clamp(rate_sp, -self.rate_sp_limit, self.rate_sp_limit)

        # ---------------------------------------------------------------
        # 5) rate controller: P + I(anti-windup) + D(measured accel) + FF,
        # ported from rate_control.cpp RateControl::update().
        # ---------------------------------------------------------------
        body_rate = quat_apply_inverse(q, ang_vel_w)
        rate_err = rate_sp - body_rate
        angular_accel = (body_rate - self.prev_body_rate) / dt
        self.prev_body_rate = body_rate

        # PX4's adaptive I-factor: shrink the integral gain as the error
        # grows, so a large setpoint step doesn't cause integrator windup
        # / bounce-back once it's caught up.
        i_factor = (1.0 - (rate_err / math.radians(400.0)) ** 2).clamp_min(0.0)
        self.rate_int = torch.clamp(
            self.rate_int + i_factor * self.Ki_rate * rate_err * dt,
            -self.rate_int_limit,
            self.rate_int_limit,
        )

        tau_cmd = (
            self.Kp_rate * rate_err
            + self.rate_int
            - self.Kd_rate * angular_accel
            + self.Kff_rate * rate_sp
        )
        tau_roll, tau_pitch, tau_yaw = tau_cmd[:, 0], tau_cmd[:, 1], tau_cmd[:, 2]

        # ---------------------------------------------------------------
        # 6) mixer: thrust + torques -> 4 motor forces, same per-motor sign
        # convention as this repo's existing controller_motor_forces
        # (matches the real Crazyflie cf2x rotor layout), but with
        # proportional desaturation instead of an independent per-motor
        # clamp — closer in spirit to PX4's control_allocator, which
        # scales thrust+torque together to preserve the commanded ratio
        # rather than silently distorting torque balance when one motor
        # would clip.
        # ---------------------------------------------------------------
        base_thrust = (T_des / 4.0).unsqueeze(-1)
        d_roll = (self.mix_scale * tau_roll).unsqueeze(-1)
        d_pitch = (self.mix_scale * tau_pitch).unsqueeze(-1)
        d_yaw = (self.mix_scale * tau_yaw).unsqueeze(-1)

        f1 = -d_roll - d_pitch - d_yaw  # front-left
        f2 = -d_roll + d_pitch + d_yaw  # rear-left
        f3 = d_roll + d_pitch - d_yaw   # rear-right
        f4 = d_roll - d_pitch + d_yaw   # front-right
        dev = torch.cat([f1, f2, f3, f4], dim=-1)  # pure torque-mix contribution per motor

        hover_thrust = mass * gravity / 4.0
        max_motor_force = (2.0 * hover_thrust).unsqueeze(-1).expand(-1, 4)
        min_motor_force = torch.zeros_like(max_motor_force)

        pos_dev = dev > 1e-8
        neg_dev = dev < -1e-8
        s_hi = torch.where(pos_dev, (max_motor_force - base_thrust) / dev.clamp_min(1e-8), torch.ones_like(dev))
        s_lo = torch.where(neg_dev, (base_thrust - min_motor_force) / (-dev).clamp_min(1e-8), torch.ones_like(dev))
        scale = torch.minimum(s_hi, s_lo).clamp(0.0, 1.0)
        scale = scale.min(dim=-1, keepdim=True).values  # most-restrictive motor sets the scale for all

        motor_forces_z = base_thrust + scale * dev
        motor_forces_z = torch.clamp(motor_forces_z, min_motor_force, max_motor_force)

        return motor_forces_z
