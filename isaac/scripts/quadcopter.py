import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(description="This script demonstrates how to simulate a quadcopter.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--use_depth", action="store_true",
                     help="Also record depth (distance_to_camera) alongside RGB for RGBD datasets.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import math
import torch
import carb
import omni
import numpy as np
import pickle
import os
import zarr
from numcodecs import Blosc
from zarr.codecs import BloscCodec
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.markers import VisualizationMarkers
from isaaclab.scene import InteractiveScene

from diffuser.utils.path import project_path
from isaac.scripts.env_cfg import CrazyflieSceneCfg, CYLINDERS, DEPTH_FAR
from isaac.scripts.zarr_episode_writer import ZarrEpisodeWriter

scene_cfg = CrazyflieSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
if args_cli.use_depth:
    scene_cfg.FPV_CAMERA_CFG = scene_cfg.FPV_CAMERA_CFG.replace(
        data_types=["rgb", "distance_to_camera"]
    )

reset = False
save_now = False
clear_now = False
recording = True 

data_dir = project_path("isaac", "dataset", "avoiding_crazyflie", "data")
os.makedirs(data_dir, exist_ok=True)
print("Resolved data path:", data_dir)

# Auto-save settings
SUCCESS_RADIUS = 0.05      # meters
SUCCESS_HOLD_STEPS = 30    # require staying in radius for 2s before save+reset
                           # (60 steps @ sim_dt=1/30s in main()) -- gives the recorded
                           # trajectory a genuine "hold at goal" tail instead of ending
                           # the instant it first enters the radius, so the model has
                           # something to imitate besides just approaching the target.

def _sub_keyboard_event(event, *args, **kwargs):
    global reset, save_now,recording,clear_now
    if event.type in (carb.input.KeyboardEventType.KEY_PRESS, carb.input.KeyboardEventType.KEY_REPEAT):
        if event.input == carb.input.KeyboardInput.SPACE:
            reset = True
            clear_now = True
            print(f"[INFO] Reset = {reset}")
        elif event.input == carb.input.KeyboardInput.ENTER:
            save_now = True
            print(f"[INFO] Saved = {save_now}")
        elif event.input == carb.input.KeyboardInput.R:
            recording = not recording
            print(f"[INFO] Recording = {recording}")
        elif event.input == carb.input.KeyboardInput.C:
            clear_now = True
            print(f"[INFO] Cleared = {clear_now}")

# subscribe to keyboard
appwindow = omni.appwindow.get_default_app_window()
input = carb.input.acquire_input_interface()
input.subscribe_to_keyboard_events(appwindow.get_keyboard(), _sub_keyboard_event)

def quat_to_euler_xyzw(q):
    # q is (N,4) in (x,y,z,w)
    x, y, z, w = q.unbind(-1)

    t0 = 2*(w*x + y*z)
    t1 = 1 - 2*(x*x + y*y)
    roll = torch.atan2(t0, t1)

    t2 = 2*(w*y - z*x)
    t2 = torch.clamp(t2, -1.0, 1.0)
    pitch = torch.asin(t2)

    t3 = 2*(w*z + x*y)
    t4 = 1 - 2*(y*y + z*z)
    yaw = torch.atan2(t3, t4)

    return roll, pitch, yaw

def sample_targets(num_envs: int, device, env_origins,
                   x_min=3.8, x_max=4.5,
                   y_min=-2.0, y_max=-1.7, #y_min=-0.94, y_max=-0.6, y_min=0.94, y_max=0.6,
                   z_min=1.5, z_max=1.0):
    """
    Sample one random target position per environment.
    Returns a tensor of shape (num_envs, 3).
    """
    target_pos = torch.empty((num_envs, 3), device=device)

    target_pos[:, 0] = torch.rand(num_envs, device=device) * (x_max - x_min) + x_min  # x
    target_pos[:, 1] = torch.rand(num_envs, device=device) * (y_max - y_min) + y_min  # y
    target_pos[:, 2] = torch.rand(num_envs, device=device) * (z_max - z_min) + z_min  # z

    return target_pos

def save_dataset_for_all_envs(episodes_states,
                              episodes_images,
                              curr_states,
                              curr_images,
                              num_envs,
                              data_dir,zarr_writer=None,
                              episodes_depths=None,
                              curr_depths=None,
                              episodes_targets=None,
                              curr_targets=None,
                              frame_dt_sec=10 * (1.0 / 30.0)):
    """frame_dt_sec: seconds of sim time between consecutive recorded frames
    (sim_dt * IMG_STRIDE at the call site) -- used only to print each episode's
    duration; default matches main()'s sim_dt=1/30, IMG_STRIDE=10."""
    use_depth = curr_depths is not None
    use_targets = curr_targets is not None

    # 1) Flush the current (possibly partial) episode into episodes_*
    if len(curr_states) > 0:
        ep_states  = np.stack(curr_states, axis=0)   # (T, N, state_dim)
        ep_images = np.stack(curr_images, axis=0)  # (T, N, H, W, 3)

        episodes_states.append(ep_states)
        episodes_images.append(ep_images)

        curr_states.clear()
        curr_images.clear()

        if use_depth:
            ep_depths = np.stack(curr_depths, axis=0)  # (T, N, H, W)
            episodes_depths.append(ep_depths)
            curr_depths.clear()

        if use_targets:
            ep_targets = np.stack(curr_targets, axis=0)  # (T, N, target_dim)
            episodes_targets.append(ep_targets)
            curr_targets.clear()

    if len(episodes_states) == 0:
        print("[INFO] No data recorded, nothing to save.")
        return

    os.makedirs(data_dir, exist_ok=True)

    # 3) Save one file per env
    for env_id in range(num_envs):
        env_states_list  = []
        env_images_list = []
        env_depths_list = []
        env_targets_list = []

        prefix = f"env_{env_id:03d}_"
        existing = [f for f in os.listdir(data_dir)
                    if f.startswith(prefix) and f.endswith(".pkl")]
        next_idx = len(existing)

        filename = f"{prefix}{next_idx:05d}.pkl"
        images_filename = f"{prefix}{next_idx:05d}_images.npz"

        save_path = os.path.join(data_dir, filename)
        images_path = os.path.join(data_dir, images_filename)

        for ep_s, ep_img in zip(episodes_states, episodes_images):
            # ep_s: (T_i, N, state_dim)
            env_states_list.append(ep_s[:, env_id, :])   # (T_i, state_dim)
            env_images_list.append(ep_img[:, env_id, ...])  # (T_i, H, W, 3)
        if use_depth:
            for ep_d in episodes_depths:
                env_depths_list.append(ep_d[:, env_id, ...])  # (T_i, H, W)
        if use_targets:
            for ep_t in episodes_targets:
                env_targets_list.append(ep_t[:, env_id, :])  # (T_i, target_dim)

        dataset_env = {
            "states": env_states_list,
            "images_file": images_filename,
        }
        if use_targets:
            dataset_env["targets"] = env_targets_list

        with open(save_path, "wb") as f:
            pickle.dump(dataset_env, f)

        images_arr = np.concatenate(env_images_list, axis=0)  # or keep list if variable lengths
        total_steps = int(sum(s.shape[0] for s in env_states_list))
        if use_depth:
            depths_arr = np.concatenate(env_depths_list, axis=0)
            np.savez_compressed(images_path, rgb=images_arr, depth=depths_arr)
        else:
            np.savez_compressed(images_path, rgb=images_arr)

        duration_sec = total_steps * frame_dt_sec
        print(f"[INFO] Env {env_id}: total_steps={total_steps}, duration={duration_sec:.1f}s, images={images_arr.shape}")
        print(f"[INFO] Saved env {env_id} data to: {save_path}")
        print(f"[INFO] Saved env {env_id} images to: {images_path}")

    # append to zarr
    assert zarr_writer is not None
    for i in range(len(episodes_states)):
        kwargs = {}
        if use_depth:
            kwargs["ep_depths"] = episodes_depths[i]
        if use_targets:
            kwargs["ep_targets"] = episodes_targets[i]
        zarr_writer.append_episode(episodes_states[i], episodes_images[i], **kwargs)
    if use_depth:
        episodes_depths.clear()
    if use_targets:
        episodes_targets.clear()

    episodes_images.clear()
    episodes_states.clear()
    print(f"[INFO] Appended episode to Zarr.")

def quat_to_yaw(quat: torch.Tensor) -> torch.Tensor:
    """
    quat: (..., 4) in (x, y, z, w)
    returns yaw ψ in radians, shape (...)
    """
    x, y, z, w = quat.unbind(-1)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return torch.atan2(siny_cosp, cosy_cosp)

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
        controller_motor_forces.Kp_pos = torch.tensor([2.8, 2.8, 3.5], device=device)
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
    # Set to 0 to disable waypoint interpolation and track target directly.
    SEG_STEPS = 0
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
    MAX_SPEED = 1.5  # m/s, cruise speed cap for the generated dataset

    pos_err = target_cmd - pos
    vel_des = Kp_pos * pos_err - Kd_pos * linvel
    # clamp the *magnitude* (not per-axis) so diagonal motion still caps at
    # MAX_SPEED instead of allowing up to MAX_SPEED per axis (sqrt(2)*MAX_SPEED
    # combined in xy)
    speed = torch.linalg.norm(vel_des, dim=-1, keepdim=True)
    scale = torch.clamp(MAX_SPEED / speed.clamp_min(1e-6), max=1.0)
    vel_des = vel_des * scale

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



def main():
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 30.0, device=args_cli.device)  # 30Hz physics/control step
    sim = SimulationContext(sim_cfg)

    sim.set_camera_view(eye=[2.0, 2.0, 2.0], target=[0.0, 0.0, 0.5]) # Set main camera
    scene = InteractiveScene(scene_cfg)

    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    # goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))
    
    robot = scene["crazyflie"]

    sim.reset()
    # print(robot.body_names)

    robot.update(sim.get_physics_dt())
    # print("Initial root state:", robot.data.root_state_w[0])
    # print("All root positions:\n", robot.data.root_state_w[:, :3])

    initial_root_state = robot.data.root_state_w.clone()# Save per-env initial root state in world frame
    env_origins = initial_root_state[:, 0:3].clone()
    env_origins = env_origins.to(sim.device)
    num_envs = robot.num_instances

    # Randomize the very first episode's spawn too (not just resets after episode 1) --
    # same box as the reset block below, keeps behaviour consistent across all episodes.
    spawn_pos = sample_targets(num_envs, sim.device, env_origins,
                                x_min=-5.0, x_max=-5.0,
                                y_min=1.7, y_max=2.0,
                                z_min=0.8, z_max=1.5)
    spawn_pose = torch.cat([spawn_pos, initial_root_state[:, 3:7]], dim=-1)  # keep initial orientation
    robot.write_root_pose_to_sim(spawn_pose)
    robot.write_root_velocity_to_sim(torch.zeros_like(initial_root_state[:, 7:]))
    robot.update(sim.get_physics_dt())
    success_count = torch.zeros(num_envs, dtype=torch.int32, device=sim.device)
    # Fetch relevant parameters to make the quadcopter hover in place
    prop_body_ids = robot.find_bodies("m.*_prop")[0]
    # robot_mass = robot.root_physx_view.get_masses().sum()
    robot_mass = robot.root_physx_view.get_masses().sum(dim=1).to(sim.device)
    gravity = torch.tensor(sim.cfg.gravity, device=sim.device).norm()

    target_pos = sample_targets(num_envs, sim.device, env_origins)
    # goal_marker.visualize(target_pos, None)

    # Now we are ready!
    print("[INFO]: Setup complete...")

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0
    warmup_steps = 0
    WARMUP_FRAMES = 5
    IMG_STRIDE = 10  # record every Nth physics step (1 = every step, no skipping).
                     # e.g. IMG_STRIDE=10 at a 200Hz physics dt -> ~20Hz recording.

    # --------- Trajectory logging buffers ---------
    episodes_states = []   # list of np arrays, one per episode
    episodes_images = []
    episodes_depths = [] if args_cli.use_depth else None
    episodes_targets = []  # list of (T, num_envs, target_dim), one per episode

    curr_images = []
    curr_states = []       # list of (num_envs, state_dim)
    curr_depths = [] if args_cli.use_depth else None
    curr_targets = []      # list of (num_envs, target_dim)

    # Simulate physics
    while simulation_app.is_running():
        global reset, save_now, recording,clear_now

        if save_now:
            save_now = False
            if "zarr_writer" not in locals():
                print("[WARN] zarr_writer not initialized yet (no camera frame yet). Skipping save.")
            else:
                print("[INFO] Saving dataset due to ENTER key...")
                save_dataset_for_all_envs(
                    episodes_states,
                    episodes_images,
                    curr_states,
                    curr_images,
                    num_envs,
                    data_dir,zarr_writer=zarr_writer,
                    episodes_depths=episodes_depths,
                    curr_depths=curr_depths,
                    episodes_targets=episodes_targets,
                    curr_targets=curr_targets,
                    frame_dt_sec=sim_dt * IMG_STRIDE,
                )

        if reset:
        
            reset = False
            clear_now = True
            # reset counters
            sim_time = 0.0
            count = 0
            # reset dof state
            joint_pos, joint_vel = robot.data.default_joint_pos, robot.data.default_joint_vel
            robot.write_joint_state_to_sim(joint_pos, joint_vel)

            spawn_pos = sample_targets(num_envs, sim.device, env_origins,
                                        x_min=-5.0, x_max=-5.0,
                                        y_min=1.7, y_max=2.0,
                                        z_min=0.8, z_max=1.5)
            spawn_pose = torch.cat([spawn_pos, initial_root_state[:, 3:7]], dim=-1)  # keep initial orientation
            robot.write_root_pose_to_sim(spawn_pose)
            zero_root_vel = torch.zeros_like(initial_root_state[:, 7:])
            robot.write_root_velocity_to_sim(zero_root_vel)

            target_pos = sample_targets(robot.num_instances, sim.device, env_origins)
            # goal_marker.visualize(target_pos, None)
            robot.reset()

            if hasattr(controller_motor_forces, "vel_int"):
                controller_motor_forces.vel_int.zero_()
            if hasattr(controller_motor_forces, "att_int"):
                controller_motor_forces.att_int.zero_()
            if hasattr(controller_motor_forces, "prev_vel_err"):
                controller_motor_forces.prev_vel_err.zero_()
            if hasattr(controller_motor_forces, "prev_att_err"):
                controller_motor_forces.prev_att_err.zero_()
            target_cmd = None

            # restart the discretized segment for the new goal
            if hasattr(controller_motor_forces, "start_pos"):
                controller_motor_forces.start_pos = robot.data.root_state_w[:, 0:3].clone()
            if hasattr(controller_motor_forces, "step_idx"):
                controller_motor_forces.step_idx.zero_()
            # reset command
            warmup_steps = WARMUP_FRAMES
            print(">>>>>>>> Reset!")
        
        if clear_now:
            clear_now = False
            episodes_states.clear()
            episodes_images.clear()
            curr_images.clear()
            curr_states.clear()
            episodes_targets.clear()
            curr_targets.clear()
            if args_cli.use_depth:
                episodes_depths.clear()
                curr_depths.clear()
            print("[INFO] Cleared all recorded data (no files deleted).")
        
        root_state = robot.data.root_state_w
        pos    = root_state[:, 0:3]
        quat   = root_state[:, 3:7]
        w = quat[:, 0]
        x = quat[:, 1]
        y = quat[:, 2]
        z = quat[:, 3]    
        quat_xyzw = torch.stack([x, y, z, w], dim=-1)

        linvel   = root_state[:, 7:10]
        angvel_w = root_state[:, 10:13] 

        # ---------------------------------------------------                        
        motor_forces_z = controller_motor_forces(
            robot.data.root_state_w,
            target_pos,
            env_origins,
            sim_dt,
            robot_mass,
            gravity,
            num_envs,
            sim,
        )
        # target_cmd = controller_motor_forces.last_target_cmd
        # pos_local    = pos - env_origins
        # print(f"env_origins Position: x={env_origins[0,0]:.3f}, y={env_origins[0,1]:.3f}, z={env_origins[0,2]:.3f}")
        # print(f"pos_local Position: x={pos_local[0,0]:.3f}, y={pos_local[0,1]:.3f}, z={pos_local[0,2]:.3f}")
        # ---------------------------------------------------
        # 6) Build force & torque tensors for PhysX
        # ---------------------------------------------------
        forces = torch.zeros(robot.num_instances, 4, 3, device=sim.device)
        torques = torch.zeros_like(forces)
        forces[..., 2] = motor_forces_z  # z component per motor
        
        robot.set_external_force_and_torque(forces, torques, body_ids=prop_body_ids)
        robot.write_data_to_sim()
        count += 1

        # --------- Build observation & log trajectory ---------
        state = torch.cat([pos, quat_xyzw, linvel, angvel_w], dim=-1)   
        
        state_np = state.detach().cpu().numpy()            # (N, 13)

        # Update sensors (important)
        scene.update(sim_dt)

        # Read RGB
        cam = scene["FPV_CAMERA_CFG"]
        rgb = cam.data.output["rgb"]
        rgb_np = rgb.detach().cpu().numpy().astype(np.uint8)

        depth_np = None
        if args_cli.use_depth:
            if "distance_to_camera" not in cam.data.output:
                raise RuntimeError(
                    "--use_depth was set but the FPV camera was not configured with "
                    "distance_to_camera output. Check the scene_cfg.FPV_CAMERA_CFG override above."
                )
            depth = cam.data.output["distance_to_camera"]
            depth_np = depth.detach().cpu().numpy().astype(np.float32)[..., 0]  # (N,H,W,1) -> (N,H,W)
            # Rays that miss everything within clipping_range come back as inf (occasionally
            # nan); replace with DEPTH_FAR so the saved zarr/npz data never carries non-finite
            # values (np.clip alone won't fix nan, so this has to happen at the source).
            non_finite = ~np.isfinite(depth_np)
            if non_finite.any():
                print(f"[WARN] {non_finite.sum()} non-finite depth pixels this step "
                      f"(no hit within clipping_range) -> clamped to DEPTH_FAR={DEPTH_FAR}")
                depth_np[non_finite] = DEPTH_FAR

        if "zarr_writer" not in locals():
            img_h, img_w = rgb_np.shape[1], rgb_np.shape[2]   # rgb_np is (N, H, W, 3)
            state_dim = state_np.shape[1]
            zarr_writer = ZarrEpisodeWriter(
                root_dir=os.path.join(data_dir, "zarr"),
                num_envs=num_envs,
                img_h=img_h,
                img_w=img_w,
                state_dim=state_dim,
                use_depth=args_cli.use_depth,
                target_dim=target_pos.shape[-1],
            )

        if warmup_steps > 0:
            warmup_steps -= 1
        elif count % IMG_STRIDE == 0:
            curr_states.append(state_np)
            curr_images.append(rgb_np)
            curr_targets.append(target_pos.detach().cpu().numpy())
            if args_cli.use_depth:
                curr_depths.append(depth_np)

        # --- Auto-save + reset when target reached ---
        target_local1 = target_pos - env_origins 
        # dist = torch.linalg.norm(pos_local - target_local1, dim=-1)  # (num_envs,)
        dist = torch.linalg.norm(pos - target_pos, dim=-1)  # (num_envs,)

        # count consecutive steps inside the success radius
        inside = dist < SUCCESS_RADIUS
        success_count = torch.where(inside, success_count + 1, torch.zeros_like(success_count))

        # if any env has been inside long enough, treat as success
        if torch.any(success_count >= SUCCESS_HOLD_STEPS):
            print(f"[INFO] Target reached (<= {SUCCESS_RADIUS}m for {SUCCESS_HOLD_STEPS} steps). Saving & resetting...")

            # Save everything recorded so far
            save_dataset_for_all_envs(
                episodes_states,
                episodes_images,
                curr_states,
                curr_images,
                num_envs,
                data_dir,zarr_writer=zarr_writer,
                episodes_depths=episodes_depths,
                curr_depths=curr_depths,
                episodes_targets=episodes_targets,
                curr_targets=curr_targets,
                frame_dt_sec=sim_dt * IMG_STRIDE,
            )

            # Reset sim to initial pose (your reset block already does this) 
            reset = True
            success_count.zero_() # reset success counter
            
            # Sample a new target + visualize it 
            target_pos = sample_targets(num_envs, sim.device, env_origins)
            # goal_marker.visualize(target_pos, None)
            # restart the discretized segment for the new goal
            if hasattr(controller_motor_forces, "start_pos"):
                controller_motor_forces.start_pos = robot.data.root_state_w[:, 0:3].clone()
            if hasattr(controller_motor_forces, "step_idx"):
                controller_motor_forces.step_idx.zero_()

        sim.step()
        sim_time += sim_dt
        robot.update(sim_dt)

        # ==========================
        # Debug
        # ==========================
        if count % 20 == 0:
            pos_np = pos[0].detach().cpu().numpy()
            tgt_np = target_pos[0].detach().cpu().numpy()
            roll, pitch, yaw = quat_to_euler_xyzw(quat_xyzw)
            print("\n==== DEBUG INFO ====")
            print(f"Step: {count}")
            print(f"Current Position: x={pos_np[0]:.3f}, y={pos_np[1]:.3f}, z={pos_np[2]:.3f}")
            print(f"Target Position : x={tgt_np[0]:.3f}, y={tgt_np[1]:.3f}, z={tgt_np[2]:.3f}")
            print(f"Roll={float(roll[0]):.3f}, Pitch={float(pitch[0]):.3f}, Yaw={float(yaw[0]):.3f}")
            mf = forces[0].detach().cpu().numpy()
            print(f"Motor Forces(N): {mf}")
    
if __name__ == "__main__":
    main()
    simulation_app.close()