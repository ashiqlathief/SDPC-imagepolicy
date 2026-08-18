# Copyright (c) 2026 — thesis project (UPB DPCC)
#
# Manually-flown data collection: same recording pipeline as quadcopter.py
# (random target sampling, goal marker, keyboard reset/save/clear, zarr +
# pickle dataset recording, --use_depth) — quadcopter.py itself is left
# untouched — but the drone is flown by hand instead of an autonomous
# position controller.
#
# Flight control: same keyboard-teleop scheme as crazyflie_px4_teleop.py
# (_command_from_keys() below, ported from there), driving PX4StyleController
# directly with a body-frame [vx, vy, vz, yaw_rate] command. target_pos /
# goal_marker are kept purely as a target for the pilot to fly toward and to
# drive the existing success-radius auto-save/reset loop — they no longer
# autopilot the drone (the old position->velocity outer loop is gone).
#
# Controls (held-key = continuous command, like crazyflie_px4_teleop.py):
#   W / S        forward / back          (body vx)
#   A / D        strafe left / right     (body vy)
#   UP / DOWN    ascend / descend        (body vz)
#   LEFT / RIGHT yaw left / right        (yaw rate)
#   SPACE        reset + clear recorded data
#   ENTER        save recorded episodes
#   R            toggle recording
#   C            clear recorded data (no files deleted)
#
# Run exactly like quadcopter.py:
#   ./isaaclab.sh -p -m isaac.scripts.quadcopter_px4 --num_envs 1

import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(description="Quadcopter simulation flown by the PX4-style controller.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--use_depth", action="store_true",
                     help="Also record depth (distance_to_camera) alongside RGB for RGBD datasets.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import carb
import omni
import numpy as np
import pickle
import os
import zarr
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.markers import VisualizationMarkers
from isaaclab.scene import InteractiveScene

from diffuser.utils.path import project_path
from isaac.scripts.crazyflie_env_cfg import CrazyflieSceneCfg, BOXES, CYLINDERS, DEPTH_FAR
from isaac.scripts.px4_style_controller import PX4StyleController

scene_cfg = CrazyflieSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
if args_cli.use_depth:
    scene_cfg.FPV_CAMERA_CFG = scene_cfg.FPV_CAMERA_CFG.replace(
        data_types=["rgb", "distance_to_camera"]
    )

reset = False
save_now = False
clear_now = False
recording = True

data_dir = project_path("isaac", "dataset", "avoiding_crazyflie", "data_px4")
os.makedirs(data_dir, exist_ok=True)
print("Resolved data path:", data_dir)

# Auto-save settings
SUCCESS_RADIUS = 0.05      # meters
SUCCESS_HOLD_STEPS = 1    # require staying in radius for 20 sim steps

# Keyboard teleop velocity caps, same magnitudes as
# crazyflie_px4_teleop.py's XY_VELOCITY_MAX / Z_VELOCITY_MAX / YAW_RATE_MAX.
XY_VELOCITY_MAX = 0.8   # m/s
Z_VELOCITY_MAX = 0.5    # m/s
YAW_RATE_MAX = 1.5      # rad/s

# Movement keys are held (continuous command, like a joystick) instead of
# one-shot, same as crazyflie_px4_teleop.py.
_MOVE_KEYS = {
    carb.input.KeyboardInput.W, carb.input.KeyboardInput.S,
    carb.input.KeyboardInput.A, carb.input.KeyboardInput.D,
    carb.input.KeyboardInput.UP, carb.input.KeyboardInput.DOWN,
    carb.input.KeyboardInput.LEFT, carb.input.KeyboardInput.RIGHT,
}
_held_keys: set = set()

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
        elif event.input in _MOVE_KEYS:
            _held_keys.add(event.input)
    elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
        _held_keys.discard(event.input)

def _command_from_keys() -> tuple[float, float, float, float]:
    """Held WASD/arrow keys -> body-frame (vx, vy, vz, yaw_rate).
    Ported from crazyflie_px4_teleop.py."""
    K = carb.input.KeyboardInput
    vx = XY_VELOCITY_MAX * ((K.W in _held_keys) - (K.S in _held_keys))
    vy = XY_VELOCITY_MAX * ((K.A in _held_keys) - (K.D in _held_keys))
    vz = Z_VELOCITY_MAX * ((K.UP in _held_keys) - (K.DOWN in _held_keys))
    yaw_rate = YAW_RATE_MAX * ((K.LEFT in _held_keys) - (K.RIGHT in _held_keys))
    return vx, vy, vz, yaw_rate

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
                   x_min=4.0, x_max=4.0,
                   y_min=-0.9, y_max=0.9,
                   z_min=0.65, z_max=0.65):
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
                              curr_depths=None):
    use_depth = curr_depths is not None

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

    if len(episodes_states) == 0:
        print("[INFO] No data recorded, nothing to save.")
        return

    os.makedirs(data_dir, exist_ok=True)

    # 3) Save one file per env
    for env_id in range(num_envs):
        env_states_list  = []
        env_images_list = []
        env_depths_list = []

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

        dataset_env = {
            "states": env_states_list,
            "images_file": images_filename,
        }

        with open(save_path, "wb") as f:
            pickle.dump(dataset_env, f)

        images_arr = np.concatenate(env_images_list, axis=0)  # or keep list if variable lengths
        total_steps = int(sum(s.shape[0] for s in env_states_list))
        if use_depth:
            depths_arr = np.concatenate(env_depths_list, axis=0)
            np.savez_compressed(images_path, rgb=images_arr, depth=depths_arr)
        else:
            np.savez_compressed(images_path, rgb=images_arr)

        print(f"[INFO] Env {env_id}: total_steps={total_steps}, images={images_arr.shape}")
        print(f"[INFO] Saved env {env_id} data to: {save_path}")
        print(f"[INFO] Saved env {env_id} images to: {images_path}")

    # append to zarr
    assert zarr_writer is not None
    if use_depth:
        for ep_s, ep_img, ep_d in zip(episodes_states, episodes_images, episodes_depths):
            zarr_writer.append_episode(ep_s, ep_img, ep_depths=ep_d)
        episodes_depths.clear()
    else:
        for ep_s, ep_img in zip(episodes_states, episodes_images):
            zarr_writer.append_episode(ep_s, ep_img)

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

class ZarrEpisodeWriter:
    def __init__(self, root_dir: str, num_envs: int, img_h: int, img_w: int,
                 state_dim: int, chunk_t: int = 256, use_depth: bool = False):
        self.root_dir = root_dir
        self.num_envs = num_envs
        self.img_h = img_h
        self.img_w = img_w
        self.state_dim = state_dim
        self.use_depth = use_depth

        os.makedirs(root_dir, exist_ok=True)

        self.groups = []
        for env_id in range(num_envs):
            path = os.path.join(root_dir, f"env_{env_id:03d}.zarr")
            g = zarr.open_group(path, mode="a")

            # Create datasets if they don't exist (appendable along time axis)
            if "rgb" not in g:
                g.create_array(
                    "rgb",
                    shape=(0, img_h, img_w, 3),
                    chunks=(min(chunk_t, 64), img_h, img_w, 3),
                    dtype="uint8",
                )
            if self.use_depth and "depth" not in g:
                g.create_array(
                    "depth",
                    shape=(0, img_h, img_w),
                    chunks=(min(chunk_t, 64), img_h, img_w),
                    dtype="float32",
                )
            g.attrs["use_depth"] = self.use_depth
            if "states" not in g:
                g.create_array(
                    "states",
                    shape=(0, state_dim),
                    chunks=(chunk_t, state_dim),
                    dtype="float32",
                )
            if "terminals" not in g:
                g.create_array(
                    "terminals",
                    shape=(0,),
                    chunks=(chunk_t,),
                    dtype="uint8",
                )
            if "episode_id" not in g:
                g.create_array(
                    "episode_id",
                    shape=(0,),
                    chunks=(chunk_t,),
                    dtype="int32",
                )

            self.groups.append(g)

        self._episode_counter = [0 for _ in range(num_envs)]

    def append_episode(self, ep_states, ep_images, ep_depths=None):
        """
        ep_states:  (T, N, state_dim)
        ep_images:  (T, N, H, W, 3) uint8
        ep_depths:  (T, N, H, W) float32, required if self.use_depth
        """
        T, N, _ = ep_states.shape
        assert N == self.num_envs
        if self.use_depth:
            assert ep_depths is not None, "use_depth=True but no depth data was provided"

        terminals = np.zeros((T,), dtype=np.uint8)
        terminals[-1] = 1

        for env_id in range(self.num_envs):
            g = self.groups[env_id]

            rgb = ep_images[:, env_id, ...]          # (T, H, W, 3)
            st  = ep_states[:, env_id, :]            # (T, state_dim)

            eid = self._episode_counter[env_id]
            ep_ids = np.full((T,), eid, dtype=np.int32)

            g["rgb"].append(rgb)
            if self.use_depth:
                depth = ep_depths[:, env_id, ...]    # (T, H, W)
                g["depth"].append(depth.astype(np.float32))
            g["states"].append(st.astype(np.float32))
            g["terminals"].append(terminals)
            g["episode_id"].append(ep_ids)

            self._episode_counter[env_id] += 1


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    sim = SimulationContext(sim_cfg)

    sim.set_camera_view(eye=[2.0, 2.0, 2.0], target=[0.0, 0.0, 0.5]) # Set main camera
    scene = InteractiveScene(scene_cfg)

    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))

    robot = scene["crazyflie"]

    sim.reset()

    robot.update(sim.get_physics_dt())

    initial_root_state = robot.data.root_state_w.clone()# Save per-env initial root state in world frame
    env_origins = initial_root_state[:, 0:3].clone()
    env_origins = env_origins.to(sim.device)
    num_envs = robot.num_instances
    success_count = torch.zeros(num_envs, dtype=torch.int32, device=sim.device)
    # Fetch relevant parameters to make the quadcopter hover in place
    prop_body_ids = robot.find_bodies("m.*_prop")[0]
    robot_mass = robot.root_physx_view.get_masses().sum(dim=1).to(sim.device)
    gravity = torch.tensor(sim.cfg.gravity, device=sim.device).norm()

    # ---- PX4-style controller, flown manually via keyboard ----
    controller = PX4StyleController(num_envs, sim.device)
    # Start each episode facing +X, same spawn heading quadcopter.py's
    # original controller always held; LEFT/RIGHT then yaw freely from there.
    controller.yaw_sp.zero_()
    controller._yaw_sp_initialized[:] = True

    target_pos = sample_targets(num_envs, sim.device, env_origins)
    goal_marker.visualize(target_pos, None)

    # Now we are ready!
    print("[INFO]: Setup complete...")

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0
    warmup_steps = 0
    WARMUP_FRAMES = 5

    # --------- Trajectory logging buffers ---------
    episodes_states = []   # list of np arrays, one per episode
    episodes_images = []
    episodes_depths = [] if args_cli.use_depth else None

    curr_images = []
    curr_states = []       # list of (num_envs, state_dim)
    curr_depths = [] if args_cli.use_depth else None

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

            robot.write_root_pose_to_sim(initial_root_state[:, :7])

            zero_root_vel = torch.zeros_like(initial_root_state[:, 7:])
            robot.write_root_velocity_to_sim(zero_root_vel)

            target_pos = sample_targets(robot.num_instances, sim.device, env_origins)
            goal_marker.visualize(target_pos, None)
            robot.reset()

            controller.reset()
            controller.yaw_sp.zero_()
            controller._yaw_sp_initialized[:] = True

            # reset command
            warmup_steps = WARMUP_FRAMES
            print(">>>>>>>> Reset!")

        if clear_now:
            clear_now = False
            episodes_states.clear()
            episodes_images.clear()
            curr_images.clear()
            curr_states.clear()
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
        # Manual flight: held WASD/arrow keys -> body-frame velocity+yawrate
        # command straight into PX4StyleController (see crazyflie_px4_teleop.py).
        # ---------------------------------------------------
        vx, vy, vz, yaw_rate = _command_from_keys()
        vel_cmd_b = torch.tensor([[vx, vy, vz]], device=sim.device).expand(num_envs, 3)
        yaw_rate_cmd = torch.full((num_envs,), yaw_rate, device=sim.device)
        motor_forces_z = controller.update(
            robot.data.root_state_w,
            vel_cmd_b,
            yaw_rate_cmd,
            sim_dt,
            robot_mass,
            gravity,
        )
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
            )

        if warmup_steps > 0:
            warmup_steps -= 1
        else:
            curr_states.append(state_np)
            curr_images.append(rgb_np)
            if args_cli.use_depth:
                curr_depths.append(depth_np)

        # --- Auto-save + reset when target reached ---
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
            )

            # Reset sim to initial pose (your reset block already does this)
            reset = True
            success_count.zero_() # reset success counter

            # Sample a new target + visualize it
            target_pos = sample_targets(num_envs, sim.device, env_origins)
            goal_marker.visualize(target_pos, None)

        sim.step()
        sim_time += sim_dt
        robot.update(sim_dt)

        # ==========================
        # Debug
        # ==========================
        if count % 20 == 0:
            pos_np = pos[0].detach().cpu().numpy()
            tgt_np = target_pos[0].detach().cpu().numpy()
            roll, pitch, yaw_dbg = quat_to_euler_xyzw(quat_xyzw)
            print("\n==== DEBUG INFO ====")
            print(f"Step: {count}")
            print(f"Current Position: x={pos_np[0]:.3f}, y={pos_np[1]:.3f}, z={pos_np[2]:.3f}")
            print(f"Target Position : x={tgt_np[0]:.3f}, y={tgt_np[1]:.3f}, z={tgt_np[2]:.3f}")
            print(f"Roll={float(roll[0]):.3f}, Pitch={float(pitch[0]):.3f}, Yaw={float(yaw_dbg[0]):.3f}")
            mf = forces[0].detach().cpu().numpy()
            print(f"Motor Forces(N): {mf}")

if __name__ == "__main__":
    main()
    simulation_app.close()
