import os, yaml, pickle
import numpy as np
import matplotlib.pyplot as plt
import diffuser.utils as utils
from diffuser.utils.path import project_path
import numpy as np


def plot_plane_3d(ax3d, normal, d, xlim, ylim, zlim, alpha=0.12, n=20):
    """
    Plot plane n·p = d as a surface patch within the given limits.
    Works for vertical planes too.
    """
    nvec = np.asarray(normal, dtype=float)
    nx, ny, nz = nvec

    # Choose which variable to solve for: pick the largest normal component
    absn = np.abs(nvec)
    solve_idx = int(np.argmax(absn))  # 0->x, 1->y, 2->z

    if solve_idx == 2 and abs(nz) > 1e-9:
        # Solve for Z on (X,Y) grid
        xs = np.linspace(xlim[0], xlim[1], n)
        ys = np.linspace(ylim[0], ylim[1], n)
        X, Y = np.meshgrid(xs, ys)
        Z = (d - nx * X - ny * Y) / nz
        ax3d.plot_surface(X, Y, Z, alpha=alpha)
        return

    if solve_idx == 0 and abs(nx) > 1e-9:
        # Solve for X on (Y,Z) grid  (good for x-walls)
        ys = np.linspace(ylim[0], ylim[1], n)
        zs = np.linspace(zlim[0], zlim[1], n)
        Y, Z = np.meshgrid(ys, zs)
        X = (d - ny * Y - nz * Z) / nx
        ax3d.plot_surface(X, Y, Z, alpha=alpha)
        return

    if solve_idx == 1 and abs(ny) > 1e-9:
        # Solve for Y on (X,Z) grid  (good for y-walls)
        xs = np.linspace(xlim[0], xlim[1], n)
        zs = np.linspace(zlim[0], zlim[1], n)
        X, Z = np.meshgrid(xs, zs)
        Y = (d - nx * X - nz * Z) / ny
        ax3d.plot_surface(X, Y, Z, alpha=alpha)
        return

def plot_vertical_halfspace_from_2d_line(ax3d, p1, p2, zlim, alpha=0.12):
    """
    Extrude a 2D line segment (p1->p2 in xy) into a vertical plane patch across zlim.
    p1, p2: [x,y]
    """
    x1, y1 = p1
    x2, y2 = p2
    z0, z1 = zlim

    # Make a 2x2 grid (line segment × two z values)
    X = np.array([[x1, x2],
                  [x1, x2]], dtype=float)
    Y = np.array([[y1, y2],
                  [y1, y2]], dtype=float)
    Z = np.array([[z0, z0],
                  [z1, z1]], dtype=float)

    ax3d.plot_surface(X, Y, Z, alpha=alpha)

def draw_cylinder(ax3d, cx, cy, r, z0, z1, n_theta=40):
    theta = np.linspace(0, 2*np.pi, n_theta)
    z = np.linspace(z0, z1, 2)
    Theta, Z = np.meshgrid(theta, z)
    X = cx + r * np.cos(Theta)
    Y = cy + r * np.sin(Theta)
    ax3d.plot_surface(X, Y, Z, alpha=0.5)

def draw_sphere(ax3d, cx, cy, cz, r, n_u=30, n_v=30):
    u = np.linspace(0, 2*np.pi, n_u)
    v = np.linspace(0, np.pi, n_v)
    U, V = np.meshgrid(u, v)
    X = cx + r * np.cos(U) * np.sin(V)
    Y = cy + r * np.sin(U) * np.sin(V)
    Z = cz + r * np.cos(V)
    ax3d.plot_surface(X, Y, Z, alpha=0.5)


fig3d = plt.figure(figsize=(10, 8))
ax3d = fig3d.add_subplot(111, projection="3d")

# Load configuration
with open('config/projection_eval.yaml', 'r') as f:
    config = yaml.safe_load(f)

exp = 'avoiding-crazyflie'
obstacles = config.get("obstacle_constraints", {}).get(exp, [])
halfspaces = config.get("halfspace_constraints", {}).get(exp, [])
ax_limits = config.get("ax_limits", {}).get(exp, None)

# Your obs layout (from your script)
obs_indices = {
    "x": 0, "y": 1, "z": 2,
    "qx": 3, "qy": 4, "qz": 5, "qw": 6,
    "vx": 7, "vy": 8, "vz": 9,
    "wx": 10, "wy": 11, "wz": 12,
    "ex": 13, "ey": 14, "ez": 15,
    "tx": 16, "ty": 17, "tz": 18
}

data_dir = project_path("isaac", "dataset", "avoiding_crazyflie", "data")
files = sorted([f for f in os.listdir(data_dir) if f.endswith(".pkl")])

traj_obs = []
traj_act = []
path_lengths = []

for fname in files:
    with open(os.path.join(data_dir, fname), "rb") as f:
        env_state = pickle.load(f)

    states  = env_state["states"][0]                 # (T, 13)
    motor_forces = env_state["actions_motor_forces"][0]  # (T, 4)
    targets = env_state["targets"][0]                # (T, 3)

    pos     = states[:, 0:3]
    quat    = states[:, 3:7]
    linvel  = states[:, 7:10]
    angvel  = states[:, 10:13]

    pos_err = targets - pos
    goal = targets[0:1, :]
    goal = np.repeat(goal, len(pos), axis=0)

    obs = np.concatenate([pos, quat, linvel, angvel, pos_err, goal], axis=-1)  # (T, 19)

    # Keep obs aligned with actions: typically obs[t] -> action[t] -> obs[t+1]
    valid_len = min(len(motor_forces), len(obs) - 1)
    obs = obs[:valid_len]
    act = motor_forces[:valid_len]

    traj_obs.append(obs)
    traj_act.append(act)
    path_lengths.append(valid_len)

print(f"Loaded {len(traj_obs)} trajectories from {data_dir}")

# # -------------------- 1) Plot XY trajectories --------------------
# fig_xy, ax_xy = plt.subplots(figsize=(9, 10))
# for i, obs in enumerate(traj_obs):
#     ax_xy.plot(obs[:, obs_indices["x"]], obs[:, obs_indices["y"]], alpha=0.7, linewidth=1.5)
#     # start marker
#     ax_xy.plot(obs[0, obs_indices["x"]], obs[0, obs_indices["y"]], "o", markersize=4)

# utils.plot_environment_constraints(exp, ax_xy)
# ax_xy.set_facecolor([1, 1, 0.9])
# ax_xy.set_xlim(ax_limits[0])
# ax_xy.set_ylim(ax_limits[1])
# ax_xy.set_title("Dataset trajectories (X-Y)")
# os.makedirs("logs/figures", exist_ok=True)
# fig_xy.savefig("logs/figures/avoiding_data_xy.png", bbox_inches="tight")

# # -------------------- 2) Plot x/y/z vs time for a few trajectories --------------------
n_show = min(10, len(traj_obs))
# fig_xyz, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
# for i in range(n_show):
#     obs = traj_obs[i]
#     axs[0].plot(obs[:, obs_indices["x"]], alpha=0.8)
#     axs[1].plot(obs[:, obs_indices["y"]], alpha=0.8)
#     axs[2].plot(obs[:, obs_indices["z"]], alpha=0.8)

# axs[0].set_ylabel("x")
# axs[1].set_ylabel("y")
# axs[2].set_ylabel("z")
# axs[2].set_xlabel("timestep")
# axs[0].set_title(f"x/y/z vs time (first {n_show} trajectories)")
# fig_xyz.savefig("logs/figures/avoiding_xyz_time.png", bbox_inches="tight")

# # -------------------- 3) Plot motor forces (actions) --------------------
# fig_act, axs_act = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
# for i in range(n_show):
#     act = traj_act[i]
#     for k in range(4):
#         axs_act[k].plot(act[:, k], alpha=0.8)

# for k in range(4):
#     axs_act[k].set_ylabel(f"u{k}")
# axs_act[-1].set_xlabel("timestep")
# axs_act[0].set_title(f"Motor forces vs time (first {n_show} trajectories)")
# fig_act.savefig("logs/figures/avoiding_actions_time.png", bbox_inches="tight")

# # -------------------- 4) Episode length distribution --------------------
# fig_len, ax_len = plt.subplots(figsize=(10, 5))
# ax_len.hist(path_lengths, bins=30)
# ax_len.set_xlabel("episode length")
# ax_len.set_ylabel("count")
# ax_len.set_title("Episode length distribution")
# fig_len.savefig("logs/figures/avoiding_episode_lengths.png", bbox_inches="tight")

#-------------------- 5) plot some trajectories --------------------

for i in range(n_show):
    obs = traj_obs[i]
    ax3d.plot(obs[:, obs_indices["x"]],
              obs[:, obs_indices["y"]],
              obs[:, obs_indices["z"]], alpha=0.8)


xlim = ax3d.get_xlim3d()
ylim = ax3d.get_ylim3d()
zlim = ax3d.get_zlim3d()

for hs in halfspaces:
    # Case A: NEW 3D plane format
    for hs in halfspaces:
        if isinstance(hs, dict) and hs.get("type") == "plane":
            plot_plane_3d(ax3d, hs["normal"], hs["d"], xlim, ylim, zlim, alpha=0.10)
        # Case B: OLD 2D line format -> vertical plane
        elif isinstance(hs, (list, tuple)) and len(hs) == 3:
            p1, p2, above_below = hs  # [[x1,y1],[x2,y2],'below']
            plot_vertical_halfspace_from_2d_line(ax3d, p1, p2, zlim, alpha=0.10)

# draw obstacles from your YAML
for obs in obstacles:
    dims = obs.get("dimensions", [])
    if obs["type"] == "cylinder_outside" and dims == ["x", "y", "z"]:
        cx, cy = obs["center"]
        r = obs["radius"]
        z0, z1 = obs.get("z_range", [0.0, 1.0])
        draw_cylinder(ax3d, cx, cy, r, z0, z1)

    elif obs["type"] == "sphere_outside" and dims == ["x", "y", "z"]:
        cx, cy, cz = obs["center"]
        r = obs["radius"]
        draw_sphere(ax3d, cx, cy, cz, r)

# Ensure x and y ranges are equal
x0, x1 = ax3d.get_xlim3d()
y0, y1 = ax3d.get_ylim3d()
xr = x1 - x0
yr = y1 - y0
r = max(xr, yr)
xc = 0.5 * (x0 + x1)
yc = 0.5 * (y0 + y1)
ax3d.set_xlim3d(xc - r/2, xc + r/2)
ax3d.set_ylim3d(yc - r/2, yc + r/2)
z0, z1 = ax3d.get_zlim3d()
ax3d.set_box_aspect((r, r, z1 - z0))

ax3d.set_xlabel("x")
ax3d.set_ylabel("y")
ax3d.set_zlabel("z")
ax3d.set_title("3D trajectories + 3D obstacles")

plt.show()
