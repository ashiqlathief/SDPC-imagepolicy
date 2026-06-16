import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Fly Crazyflie on a Lissajous figure-8 path (above obstacles).")
parser.add_argument("--num_envs",    type=int,   default=1)
parser.add_argument("--n_waypoints", type=int,   default=40,
                    help="Number of waypoints sampled on the figure-8 (per lap)")
parser.add_argument("--laps",        type=int,   default=2,
                    help="Number of complete figure-8 laps to fly")
parser.add_argument("--wp_radius",   type=float, default=0.12,
                    help="Acceptance radius to consider a waypoint reached (m)")
parser.add_argument("--save_dir",    type=str,   default="trajectories",
                    help="Directory where the trajectory .npz is written")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True          # required: CrazyflieSceneCfg spawns an FPV camera
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── imports that require the sim app to be running ───────────────────────────
import math
import os
import carb
import omni
import time
import torch
import numpy as np
import zarr

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene
from isaaclab.sim import SimulationContext
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.markers import VisualizationMarkers
from diffuser.utils.path import project_path
from isaac.scripts.crazyflie_env_cfg import CrazyflieSceneCfg

# ── global flags (keyboard-driven) ───────────────────────────────────────────
reset     = False
save_now  = False
clear_now = False
recording = True

data_dir = project_path("isaac", "dataset", "figure8_lissajous", "data0")
os.makedirs(data_dir, exist_ok=True)
print("Resolved data path:", data_dir)

WARMUP_FRAMES = 5


def _sub_keyboard_event(event, *args, **kwargs):
    global reset, save_now, recording, clear_now
    if event.type in (carb.input.KeyboardEventType.KEY_PRESS,
                      carb.input.KeyboardEventType.KEY_REPEAT):
        if event.input == carb.input.KeyboardInput.SPACE:
            reset     = True
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


appwindow = omni.appwindow.get_default_app_window()
_input    = carb.input.acquire_input_interface()
_input.subscribe_to_keyboard_events(appwindow.get_keyboard(), _sub_keyboard_event)


# ─────────────────────────────────────────────────────────────────────────────
# Zarr dataset writer  (self-contained — avoids importing quadcopter.py which
# creates its own AppLauncher at module level)
# ─────────────────────────────────────────────────────────────────────────────

class ZarrEpisodeWriter:
    def __init__(self, root_dir, num_envs, img_h, img_w, state_dim, chunk_t=256):
        self.num_envs = num_envs
        os.makedirs(root_dir, exist_ok=True)
        self.groups = []
        for env_id in range(num_envs):
            path = os.path.join(root_dir, f"env_{env_id:03d}.zarr")
            g    = zarr.open_group(path, mode="a")
            if "rgb" not in g:
                g.create_array("rgb",        shape=(0, img_h, img_w, 3),
                               chunks=(min(chunk_t, 64), img_h, img_w, 3), dtype="uint8")
            if "states" not in g:
                g.create_array("states",     shape=(0, state_dim),
                               chunks=(chunk_t, state_dim), dtype="float32")
            if "terminals" not in g:
                g.create_array("terminals",  shape=(0,),
                               chunks=(chunk_t,), dtype="uint8")
            if "episode_id" not in g:
                g.create_array("episode_id", shape=(0,),
                               chunks=(chunk_t,), dtype="int32")
            self.groups.append(g)
        self._episode_counter = [0] * num_envs

    def append_episode(self, ep_states, ep_images):
        """ep_states: (T, N, state_dim)  ep_images: (T, N, H, W, 3) uint8"""
        T, N, _ = ep_states.shape
        terminals = np.zeros((T,), dtype=np.uint8)
        terminals[-1] = 1
        for env_id in range(self.num_envs):
            g   = self.groups[env_id]
            eid = self._episode_counter[env_id]
            g["rgb"].append(ep_images[:, env_id, ...])
            g["states"].append(ep_states[:, env_id, :].astype(np.float32))
            g["terminals"].append(terminals)
            g["episode_id"].append(np.full((T,), eid, dtype=np.int32))
            self._episode_counter[env_id] += 1


def save_dataset(episodes_states, episodes_images, curr_states, curr_images,
                 num_envs, data_dir, zarr_writer):
    """Flush current episode buffers and append everything to zarr."""
    if len(curr_states) > 0:
        episodes_states.append(np.stack(curr_states, axis=0))
        episodes_images.append(np.stack(curr_images, axis=0))
        curr_states.clear()
        curr_images.clear()
    if not episodes_states:
        print("[INFO] No data to save.")
        return
    for ep_s, ep_img in zip(episodes_states, episodes_images):
        zarr_writer.append_episode(ep_s, ep_img)
    episodes_images.clear()
    episodes_states.clear()
    print("[INFO] Episode appended to zarr.")


# ─────────────────────────────────────────────────────────────────────────────
# Lissajous figure-8 path  (flies ABOVE the obstacle zone)
#
# Uses the parametric equations:
#   x(t) = cx + rx · sin(t)       ← single frequency
#   y(t) = cy + ry · sin(2t)      ← double frequency creates the crossover
#
# At z = 1.2 m the drone clears all obstacles (WALL_HEIGHT = 1.0 m) by 0.2 m.
# The path is a smooth analytic curve — no sharp corners, no arc-length
# re-sampling needed.  Waypoints are uniformly spaced in parameter t.
#
# Shape viewed from above (x = right, y = up):
#
#           upper half (y > 0)
#   (1.1,0) ─────────────────── (3.9,0)
#              (2.5, ±0.55)
#           lower half (y < 0)
#
# Path extents:  x ∈ [1.1, 3.9]   y ∈ [-0.55, +0.55]   z = 1.2 m
# ─────────────────────────────────────────────────────────────────────────────

def make_figure8_waypoints(n=40, cx=2.5, cy=0.0, z=1.2, rx=1.4, ry=0.55):
    """
    Return (n, 3) waypoints for one full Lissajous figure-8 lap.

    Parameters
    ----------
    n   : number of waypoints per lap (uniform in t ∈ [0, 2π))
    cx  : x centre of the loop (middle of corridor)
    cy  : y centre (0 = corridor midline)
    z   : flight altitude — keep above WALL_HEIGHT (1.0 m) to clear obstacles
    rx  : half-width in x  → x ∈ [cx-rx, cx+rx]
    ry  : half-width in y  → y ∈ [cy-ry, cy+ry]
    """
    t  = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False, dtype=np.float32)
    xs = cx + rx * np.sin(t)
    ys = cy + ry * np.sin(2.0 * t)
    zs = np.full(n, z, dtype=np.float32)
    return np.stack([xs, ys, zs], axis=1)   # (n, 3)


# ─────────────────────────────────────────────────────────────────────────────
# Cascaded PID controller  (position setpoint → 4 motor forces)
# ─────────────────────────────────────────────────────────────────────────────

def quat_wxyz_to_euler(q_wxyz):
    """(N,4) quaternion (w,x,y,z) → roll, pitch, yaw."""
    w, x, y, z = q_wxyz[:, 0], q_wxyz[:, 1], q_wxyz[:, 2], q_wxyz[:, 3]
    roll  = torch.atan2(2.0*(w*x + y*z), 1.0 - 2.0*(x*x + y*y))
    pitch = torch.asin(torch.clamp(2.0*(w*y - z*x), -1.0, 1.0))
    yaw   = torch.atan2(2.0*(w*z + x*y), 1.0 - 2.0*(y*y + z*z))
    return roll, pitch, yaw


def controller_motor_forces(root_state, target_pos, sim_dt, robot_mass, gravity,
                             num_envs, device, yaw_setpoint=None):
    s = controller_motor_forces
    if not hasattr(s, "vel_int") or s.vel_int.shape[0] != num_envs:
        s.Kp_pos = torch.tensor([2.8, 2.8, 3.5], device=device)
        s.Kd_pos = torch.tensor([0.8, 0.8, 0.8], device=device)
        s.Kp_vel = torch.tensor([1.0, 1.0, 2.0], device=device)
        s.Ki_vel = torch.tensor([0.1, 0.1, 0.3], device=device)
        s.Kd_vel = torch.tensor([0.2, 0.2, 0.5], device=device)
        s.Kp_att = torch.tensor([4.0, 4.0, 3.0], device=device)   # yaw Kp raised
        s.Kd_att = torch.tensor([0.5, 0.5, 0.5], device=device)   # yaw Kd raised
        s.Ki_att = torch.tensor([0.1, 0.1, 0.0], device=device)
        s.MIX_SCALE    = 0.01
        s.vel_int      = torch.zeros(num_envs, 3, device=device)
        s.att_int      = torch.zeros(num_envs, 3, device=device)
        s.prev_vel_err = torch.zeros(num_envs, 3, device=device)
        s.prev_att_err = torch.zeros(num_envs, 3, device=device)

    pos    = root_state[:, 0:3]
    quat   = root_state[:, 3:7]   # Isaac Lab: (w, x, y, z)
    linvel = root_state[:, 7:10]

    pos_err = target_pos - pos
    vel_des = s.Kp_pos * pos_err - s.Kd_pos * linvel
    vel_des = torch.clamp(vel_des, -0.5, 0.5)

    vel_err = vel_des - linvel
    s.vel_int.add_(vel_err * sim_dt)
    vel_der = (vel_err - s.prev_vel_err) / sim_dt
    s.prev_vel_err.copy_(vel_err)

    acc_cmd   = s.Kp_vel * vel_err + s.Ki_vel * s.vel_int + s.Kd_vel * vel_der
    theta_des = acc_cmd[:, 0] / gravity
    phi_des   = -acc_cmd[:, 1] / gravity

    T_des = robot_mass * (gravity + acc_cmd[:, 2])
    T_des = torch.clamp(T_des, torch.zeros_like(T_des), 2.0 * robot_mass * gravity)

    roll, pitch, yaw = quat_wxyz_to_euler(quat)

    # Use the precomputed path-tangent heading supplied by main().
    # During hover (yaw_setpoint=None) hold current heading so the drone
    # doesn't spin while waiting to enter the path.
    if yaw_setpoint is not None:
        yaw_des = yaw_setpoint
    else:
        yaw_des = yaw

    att_err = torch.stack([phi_des - roll, theta_des - pitch, yaw_des - yaw], dim=-1)
    att_err = (att_err + math.pi) % (2.0 * math.pi) - math.pi

    s.att_int.add_(att_err * sim_dt)
    att_der = (att_err - s.prev_att_err) / sim_dt
    s.prev_att_err.copy_(att_err)

    tau  = s.Kp_att * att_err + s.Ki_att * s.att_int + s.Kd_att * att_der
    base = T_des.view(-1, 1) / 4.0
    dr   = s.MIX_SCALE * tau[:, 0].view(-1, 1)
    dp   = s.MIX_SCALE * tau[:, 1].view(-1, 1)
    dy   = s.MIX_SCALE * tau[:, 2].view(-1, 1)

    forces_z = torch.cat([base - dr - dp - dy,
                           base - dr + dp + dy,
                           base + dr + dp - dy,
                           base + dr - dp + dy], dim=-1)

    max_f    = (2.0 * robot_mass * gravity / 4.0).view(-1, 1).expand_as(forces_z)
    forces_z = torch.clamp(forces_z, torch.zeros_like(forces_z), max_f)
    return forces_z


def reset_pid():
    s = controller_motor_forces
    for attr in ("vel_int", "att_int", "prev_vel_err", "prev_att_err"):
        if hasattr(s, attr):
            getattr(s, attr).zero_()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    sim     = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[2.5, -4.0, 3.5], target=[2.5, 0.0, 1.2])

    scene = InteractiveScene(CrazyflieSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0))
    robot = scene["crazyflie"]

    m_cfg = FRAME_MARKER_CFG.copy()
    m_cfg.markers["frame"].scale = (0.12, 0.12, 0.12)
    wp_marker = VisualizationMarkers(m_cfg.replace(prim_path="/Visuals/wp_target"))

    sim.reset()
    robot.update(sim.get_physics_dt())

    initial_root_state = robot.data.root_state_w.clone()
    device        = sim.device
    num_envs      = robot.num_instances
    prop_body_ids = robot.find_bodies("m.*_prop")[0]
    robot_mass    = robot.root_physx_view.get_masses().sum(dim=1).to(device)
    gravity       = torch.tensor(sim.cfg.gravity, device=device).norm()
    sim_dt        = sim.get_physics_dt()

    # ── dataset buffers ───────────────────────────────────────────────────────
    episodes_states = []
    episodes_images = []
    curr_states     = []
    curr_images     = []

    # ── Lissajous figure-8 waypoints ─────────────────────────────────────────
    one_lap   = make_figure8_waypoints(args_cli.n_waypoints)
    all_wps   = np.tile(one_lap, (args_cli.laps, 1))
    total_wps = len(all_wps)

    # Precompute the path tangent heading at every waypoint.
    # tangent[i] = unit vector from waypoint i toward waypoint i+1.
    # This is the exact direction the drone (and camera) should face at step i.
    nxt_xy      = np.roll(all_wps[:, :2], -1, axis=0)          # shift by 1 (wraps)
    delta_xy    = (nxt_xy - all_wps[:, :2]).astype(np.float32)
    norms       = np.maximum(np.linalg.norm(delta_xy, axis=1, keepdims=True), 1e-6)
    tang_yaw    = np.arctan2(delta_xy[:, 1] / norms[:, 0],
                              delta_xy[:, 0] / norms[:, 0]).astype(np.float32)
    tang_yaw_t  = torch.from_numpy(tang_yaw).to(device)        # (total_wps,)

    print("[INFO] Lissajous figure-8 navigation  (above obstacles)")
    print(f"       Waypoints per lap : {args_cli.n_waypoints}")
    print(f"       Laps              : {args_cli.laps}  →  {total_wps} targets total")
    print(f"       Flight altitude   : z=1.2 m  (0.2 m above obstacle tops at z=1.0)")
    print(f"       Path extents      : x∈[1.1,3.9]  y∈[-0.55,+0.55]")
    print(f"       Acceptance radius : {args_cli.wp_radius} m")
    print()

    # ── flight state ──────────────────────────────────────────────────────────
    HOVER_TARGET = torch.tensor([[0.0, 0.0, 1.2]], device=device).expand(num_envs, -1)
    HOVER_STEPS  = 400        # 2 s at dt=0.005 to climb and stabilise

    hover_count  = 0
    wp_idx       = -1         # -1 = hover phase, 0+ = figure-8 phase
    warmup_steps = 0
    count        = 0
    sim_time     = 0.0

    # ── trajectory log (npz) ─────────────────────────────────────────────────
    traj_pos = []
    traj_wp  = []

    while simulation_app.is_running():
        global reset, save_now, recording, clear_now

        # ── ENTER: manual save ────────────────────────────────────────────────
        if save_now:
            save_now = False
            if "zarr_writer" not in locals():
                print("[WARN] zarr_writer not ready yet. Skipping.")
            else:
                save_dataset(episodes_states, episodes_images,
                             curr_states, curr_images, num_envs, data_dir, zarr_writer)

        # ── SPACE: reset to start of figure-8 ────────────────────────────────
        if reset:
            reset     = False
            clear_now = True
            count     = 0
            sim_time  = 0.0

            joint_pos, joint_vel = robot.data.default_joint_pos, robot.data.default_joint_vel
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            robot.write_root_pose_to_sim(initial_root_state[:, :7])
            robot.write_root_velocity_to_sim(torch.zeros_like(initial_root_state[:, 7:]))
            robot.reset()

            wp_idx      = -1
            hover_count = 0
            reset_pid()
            warmup_steps = WARMUP_FRAMES
            print(">>>>>>>> Reset!")

        # ── C: clear buffers ──────────────────────────────────────────────────
        if clear_now:
            clear_now = False
            episodes_states.clear()
            episodes_images.clear()
            curr_states.clear()
            curr_images.clear()
            traj_pos.clear()
            traj_wp.clear()
            print("[INFO] Cleared all recorded data (no files deleted).")

        # ── current robot state ───────────────────────────────────────────────
        root_state = robot.data.root_state_w
        pos_world  = root_state[:, 0:3]
        quat       = root_state[:, 3:7]
        linvel     = root_state[:, 7:10]
        angvel_w   = root_state[:, 10:13]

        state    = torch.cat([pos_world, quat, linvel, angvel_w], dim=-1)  # (N, 13)
        state_np = state.detach().cpu().numpy()

        # ── phase 1: climb and hover at z=1.2 ────────────────────────────────
        if wp_idx < 0:
            target_pos  = HOVER_TARGET
            hover_count += 1
            if hover_count >= HOVER_STEPS:
                wp_idx = 0
                reset_pid()
                print("[INFO] Hover stable. Entering figure-8 path at waypoint 0 ...")

        # ── phase 2: follow Lissajous figure-8 waypoints ─────────────────────
        else:
            wp_xyz     = all_wps[wp_idx]
            target_pos = torch.tensor(wp_xyz[None, :], dtype=torch.float32,
                                      device=device).expand(num_envs, -1)
            wp_marker.visualize(target_pos[:1], None)

            dist = torch.linalg.norm(pos_world - target_pos, dim=-1).mean().item()
            if dist < args_cli.wp_radius:
                wp_idx += 1
                if wp_idx >= total_wps:
                    print(f"[INFO] {args_cli.laps} laps complete — saving and restarting ...")
                    if "zarr_writer" in locals():
                        save_dataset(episodes_states, episodes_images,
                                     curr_states, curr_images, num_envs, data_dir, zarr_writer)
                    # reset drone and restart from waypoint 0
                    joint_pos, joint_vel = robot.data.default_joint_pos, robot.data.default_joint_vel
                    robot.write_joint_state_to_sim(joint_pos, joint_vel)
                    robot.write_root_pose_to_sim(initial_root_state[:, :7])
                    robot.write_root_velocity_to_sim(torch.zeros_like(initial_root_state[:, 7:]))
                    robot.reset()
                    wp_idx       = -1
                    hover_count  = 0
                    count        = 0
                    sim_time     = 0.0
                    warmup_steps = WARMUP_FRAMES
                    reset_pid()
                    traj_pos.clear()
                    traj_wp.clear()

        # ── compute and apply motor forces ────────────────────────────────────
        # During path phase supply the tangent heading so the camera always
        # faces the direction of travel; during hover keep current heading.
        yaw_sp = tang_yaw_t[wp_idx].unsqueeze(0).expand(num_envs) if wp_idx >= 0 else None
        forces_z = controller_motor_forces(
            root_state, target_pos, sim_dt, robot_mass, gravity, num_envs, device,
            yaw_setpoint=yaw_sp,
        )
        forces  = torch.zeros(num_envs, 4, 3, device=device)
        torques = torch.zeros_like(forces)
        forces[..., 2] = forces_z
        robot.set_external_force_and_torque(forces, torques, body_ids=prop_body_ids)
        robot.write_data_to_sim()

        # ── step simulation ───────────────────────────────────────────────────
        sim.step()
        robot.update(sim_dt)
        scene.update(sim_dt)
        count    += 1
        sim_time += sim_dt

        # ── camera ────────────────────────────────────────────────────────────
        cam    = scene["FPV_CAMERA_CFG"]
        rgb    = cam.data.output["rgb"]
        rgb_np = rgb.detach().cpu().numpy().astype(np.uint8)

        if "zarr_writer" not in locals():
            img_h, img_w = rgb_np.shape[1], rgb_np.shape[2]
            zarr_writer  = ZarrEpisodeWriter(
                root_dir  = os.path.join(data_dir, "zarr"),
                num_envs  = num_envs,
                img_h     = img_h,
                img_w     = img_w,
                state_dim = state_np.shape[1],
            )

        # ── record (figure-8 phase only, after warmup) ────────────────────────
        if warmup_steps > 0:
            warmup_steps -= 1
        elif recording and wp_idx >= 0:
            curr_states.append(state_np)
            curr_images.append(rgb_np)

        # ── lightweight npz trajectory log ────────────────────────────────────
        if wp_idx >= 0:
            traj_pos.append(pos_world.detach().cpu().numpy().copy())
            traj_wp.append(int(wp_idx))

        # ── console progress ──────────────────────────────────────────────────
        if count % 200 == 0:
            p     = pos_world[0].detach().cpu().numpy()
            phase = (f"hover ({hover_count}/{HOVER_STEPS})" if wp_idx < 0
                     else f"wp {wp_idx:3d}/{total_wps}")
            print(f"[Step {count:6d}]  pos=({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f})  "
                  f"phase={phase}")

    # ── save lightweight trajectory npz ──────────────────────────────────────
    if len(traj_pos) == 0:
        print("[WARN] No trajectory data recorded.")
        return

    os.makedirs(args_cli.save_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_path  = os.path.join(args_cli.save_dir, f"lissajous_{timestamp}.npz")
    np.savez(
        out_path,
        positions           = np.stack(traj_pos, axis=0),   # (T, N, 3)
        wp_indices          = np.array(traj_wp, dtype=np.int32),
        waypoints           = all_wps,                        # (total_wps, 3)
        n_waypoints_per_lap = args_cli.n_waypoints,
        laps                = args_cli.laps,
        wp_radius           = args_cli.wp_radius,
        sim_dt              = sim_dt,
    )
    print(f"[INFO] Trajectory saved → {out_path}  ({len(traj_pos)} steps)")
    print(f"       Load: data = np.load('{out_path}'); xyz = data['positions'][:, 0, :]")


if __name__ == "__main__":
    main()
    simulation_app.close()
