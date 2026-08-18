# Copyright (c) 2026 — thesis project (UPB DPCC)
#
# Keyboard teleoperation of the Crazyflie in Isaac Lab, driven by the
# PX4-style controller in px4_style_controller.py.
#
# This is a NEW file — crazyflie_env.py, crazyflie_env_cfg.py and
# quadcopter_env.py are all left untouched. It:
#   - reuses CrazyflieSceneCfg from crazyflie_env_cfg.py (same robot/scene
#     the RL env and quadcopter.py already fly),
#   - mirrors the sim-setup / physics-step / keyboard-callback / debug-print
#     structure already working in quadcopter.py and quadcopter_env.py,
#   - reuses the prop_body_ids / robot_mass / gravity / set_external_force_and_torque
#     pattern already working in crazyflie_env.py,
#   - replaces the hand-tuned position-target controller_motor_forces() with
#     PX4StyleController, fed by a body-frame [vx, vy, vz, yaw_rate] command
#     — the same command shape
#     ROS2-PX4_Drone_Teleoperation_Using_Joystick/px4_offboard publishes on
#     /offboard_velocity_cmd — read here from the keyboard instead of a
#     joystick/ROS2 topic (this repo has no ROS2/joystick wiring), the same
#     substitution that repo itself offers via its own teleop_keyboard.py
#     alongside teleop_joystick.py.
#
# How this maps onto ROS2-PX4_Drone_Teleoperation_Using_Joystick, piece by
# piece (operationally analogous — NOT a byte-for-byte or process-for-process
# match, see "Fidelity" below):
#
#   ROS2-PX4 teleop repo                          | This file
#   -----------------------------------------------|----------------------------------------
#   teleop_joystick.py / teleop_keyboard.py reads   | _command_from_keys() reads held
#   the joystick/keys, publishes geometry_msgs/     | carb.input.KeyboardInput state, returns
#   Twist on /offboard_velocity_cmd                | (vx, vy, vz, yaw_rate) directly, no topic
#   -----------------------------------------------|----------------------------------------
#   velocity_offboard_control.py subscribes to that| PX4StyleController.update() takes that
#   Twist, rotates body-frame vx/vy into world     | tuple as vel_cmd_b/yaw_rate_cmd and does
#   frame using the vehicle's current yaw, and     | the identical body->world yaw rotation
#   publishes PX4 TrajectorySetpoint (velocity     | internally (body_vel_to_world(), ported
#   mode) over uXRCE-DDS                            | line-for-line from that file's loop())
#   -----------------------------------------------|----------------------------------------
#   PX4 SITL's real flight-stack (mc_pos_control -> | PX4StyleController's own cascade:
#   mc_att_control -> mc_rate_control ->            | velocity PID -> quaternion attitude
#   control_allocator) turns that setpoint into     | P-control (reduced-yaw-priority) -> rate
#   per-motor commands                              | PID+FF -> per-motor mixer (see
#                                                    | px4_style_controller.py's own header for
#                                                    | exactly what's ported vs. simplified)
#   -----------------------------------------------|----------------------------------------
#   Gazebo simulates the airframe's rigid-body      | Isaac Lab/PhysX simulates the Crazyflie
#   physics from those per-motor commands           | rigid body from set_external_force_and_torque
#   -----------------------------------------------|----------------------------------------
#   ARM (button) / DISARM (button)                  | SPACE = reset to the initial pose
#
# Fidelity — read this before treating "maps onto" as "is": this file does
# NOT run ROS2, MAVLink, uXRCE-DDS, or the real PX4 SITL binary. There is no
# separate flight-stack process and no network bridge — everything above the
# "physics" row runs as plain Python/torch inside this one Isaac Lab process.
# PX4StyleController is a hand-ported reimplementation of PX4's control LAW
# (same equations, see px4_style_controller.py's header for exactly what's
# faithful vs. simplified, e.g. no attitude reference-model, gains are
# starting points not PX4's tuned firmware constants) — it is PX4-*style*,
# not PX4 itself. If you need the literal PX4 firmware in the loop, that's
# the separate MAVLink-HIL-bridge approach we discussed and decided against
# here specifically because it doesn't scale to this repo's num_envs-
# vectorized simulation the way this in-process controller does.
#
# Controls (held-key = continuous command, like a joystick, not one-shot):
#   W / S        forward / back          (body vx)
#   A / D        strafe left / right     (body vy)
#   UP / DOWN    ascend / descend        (body vz)
#   LEFT / RIGHT yaw left / right        (yaw rate)
#   SPACE        reset to the initial pose
#
# Run (from the dpcc-thesis root, matching quadcopter.py's import style):
#   ./isaaclab.sh -p -m isaac.scripts.crazyflie_px4_teleop --num_envs 1

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Keyboard-teleoperate the Crazyflie via a PX4-style controller.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import carb
import omni.appwindow
import torch

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene
from isaaclab.sim import SimulationContext

from isaac.scripts.crazyflie_env_cfg import CrazyflieSceneCfg
from isaac.scripts.px4_style_controller import PX4StyleController

# Same magnitudes as teleop_joystick.py's XY_VELOCITY_MAX / YAW_RATE_MAX
# (ROS2-PX4_Drone_Teleoperation_Using_Joystick/px4_offboard/px4_offboard/teleop_joystick.py).
XY_VELOCITY_MAX = 0.8   # m/s
Z_VELOCITY_MAX = 0.5    # m/s
YAW_RATE_MAX = 1.5      # rad/s

_held_keys: set[str] = set()
_reset_requested = False


def _on_keyboard_event(event, *args, **kwargs) -> bool:
    global _reset_requested
    key = event.input
    if event.type == carb.input.KeyboardEventType.KEY_PRESS:
        if key == carb.input.KeyboardInput.SPACE:
            _reset_requested = True
        else:
            _held_keys.add(key)
    elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
        _held_keys.discard(key)
    return True


def _command_from_keys() -> tuple[float, float, float, float]:
    """Held WASD/arrow keys -> body-frame (vx, vy, vz, yaw_rate)."""
    K = carb.input.KeyboardInput
    vx = XY_VELOCITY_MAX * ((K.W in _held_keys) - (K.S in _held_keys))
    vy = XY_VELOCITY_MAX * ((K.A in _held_keys) - (K.D in _held_keys))
    vz = Z_VELOCITY_MAX * ((K.UP in _held_keys) - (K.DOWN in _held_keys))
    yaw_rate = YAW_RATE_MAX * ((K.LEFT in _held_keys) - (K.RIGHT in _held_keys))
    return vx, vy, vz, yaw_rate


def main():
    global _reset_requested

    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[2.0, 2.0, 2.0], target=[0.0, 0.0, 0.5])

    scene_cfg = CrazyflieSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    robot = scene["crazyflie"]

    appwindow = omni.appwindow.get_default_app_window()
    input_iface = carb.input.acquire_input_interface()
    input_iface.subscribe_to_keyboard_events(appwindow.get_keyboard(), _on_keyboard_event)

    sim.reset()
    robot.update(sim.get_physics_dt())

    initial_root_state = robot.data.root_state_w.clone()
    num_envs = robot.num_instances
    device = sim.device

    prop_body_ids = robot.find_bodies("m.*_prop")[0]
    robot_mass = robot.root_physx_view.get_masses().sum(dim=1).to(device)
    gravity = torch.tensor(sim.cfg.gravity, device=device).norm()

    controller = PX4StyleController(num_envs, device)

    print("[INFO]: Setup complete. W/A/S/D = move, UP/DOWN = climb/descend, "
          "LEFT/RIGHT = yaw, SPACE = reset.")

    sim_dt = sim.get_physics_dt()
    count = 0

    while simulation_app.is_running():
        if _reset_requested:
            _reset_requested = False
            joint_pos, joint_vel = robot.data.default_joint_pos, robot.data.default_joint_vel
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            robot.write_root_pose_to_sim(initial_root_state[:, :7])
            robot.write_root_velocity_to_sim(torch.zeros_like(initial_root_state[:, 7:]))
            robot.reset()
            controller.reset()
            print(">>>>>>>> Reset!")

        vx, vy, vz, yaw_rate = _command_from_keys()
        vel_cmd_b = torch.tensor([[vx, vy, vz]], device=device).expand(num_envs, 3)
        yaw_rate_cmd = torch.full((num_envs,), yaw_rate, device=device)

        motor_forces_z = controller.update(
            robot.data.root_state_w,
            vel_cmd_b,
            yaw_rate_cmd,
            sim_dt,
            robot_mass,
            gravity,
        )

        forces = torch.zeros(num_envs, 4, 3, device=device)
        torques = torch.zeros_like(forces)
        forces[..., 2] = motor_forces_z
        robot.set_external_force_and_torque(forces, torques, body_ids=prop_body_ids)
        robot.write_data_to_sim()

        sim.step()
        robot.update(sim_dt)
        scene.update(sim_dt)
        count += 1

        if count % 50 == 0:
            pos = robot.data.root_state_w[0, 0:3].detach().cpu().numpy()
            print(
                f"[step {count}] cmd(vx={vx:+.2f}, vy={vy:+.2f}, vz={vz:+.2f}, "
                f"yaw_rate={yaw_rate:+.2f}) pos=({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})"
            )


if __name__ == "__main__":
    main()
    simulation_app.close()
