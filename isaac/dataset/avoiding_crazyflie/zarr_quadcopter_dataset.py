import zarr, numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
from isaac.dataset.sim_path import sim_framework_path

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 12

# ── corridor / scene constants (kept in sync with plotall.py) ─────────────
CORRIDOR_END  = 4.1      # x position of the end wall / gate
WALL_HEIGHT   = 1.0      # floor=0, ceiling=1 m
X_LIM_XY     = (-0.2, 4.7)
Y_LIM_XY     = (-1.2,  1.2)
X_LIM_XZ     = (-0.2, 4.7)
Z_LIM_XZ     = (-0.05, 1.35)

WALL_KW = dict(linewidth=1.5, edgecolor="#555555", facecolor="#cccccc",
               alpha=0.50, zorder=1)

BOXES = [
    # (x, y)  — 2-D centre, activate when obstacles are present at collection time
    # (3.5,  0.2), (2.2, -0.2), (1.6,  0.2),
    # (3.7, -0.3), (0.6,  0.1), (1.0, -0.3),
]
CYLINDERS = [
    # (x, y)
    # (1.4, -0.2), (2.8,  0.3), (1.2,  0.4),
    # (3.5,  0.4), (2.0,  0.4), (2.5,  0.3),
]


# ── scene geometry helpers (mirrors plotall.py) ───────────────────────────
def add_corridor_xy(ax):
    """Draw top wall, bottom wall and end wall on an XY axes."""
    ax.add_patch(Rectangle((0.0,  1.00), CORRIDOR_END, 0.10, **WALL_KW))
    ax.add_patch(Rectangle((0.0, -1.10), CORRIDOR_END, 0.10, **WALL_KW))
    ax.add_patch(Rectangle((CORRIDOR_END, -1.10), 0.10, 2.20, **WALL_KW))


def add_corridor_xz(ax):
    """Draw floor and ceiling dashed lines on an XZ axes."""
    ax.axhline(0.0, color="#555555", linewidth=1.2, linestyle="--", alpha=0.55)
    ax.axhline(WALL_HEIGHT, color="#555555", linewidth=1.2,
               linestyle="--", alpha=0.55)


def add_goal_line(ax, plane="xy"):
    """Draw the gate / goal line at x = 4.0 m."""
    if plane == "xy":
        ax.plot([4.0, 4.0], [-1.0, 1.0], color="#00aa00", linewidth=2.2,
                linestyle="-", alpha=0.85, zorder=5)
    else:
        ax.plot([4.0, 4.0], [0.0, WALL_HEIGHT], color="#00aa00",
                linewidth=2.2, linestyle="-", alpha=0.85, zorder=5)


def add_obstacles(ax, boxes, cylinders, tighten=0.0,
                  box_size=0.20, cyl_radius=0.06):
    """Overlay box (red) and cylinder (orange) obstacles with optional tighten ring."""
    for x, y in boxes:
        ax.add_patch(Rectangle(
            (x - box_size/2, y - box_size/2), box_size, box_size,
            linewidth=0.8, edgecolor="black", facecolor="#cc3030",
            alpha=0.35, zorder=2,
        ))
        if tighten > 0:
            ax.add_patch(Rectangle(
                (x - box_size/2 - tighten, y - box_size/2 - tighten),
                box_size + 2*tighten, box_size + 2*tighten,
                linewidth=0.8, edgecolor="#cc3030", facecolor="none",
                linestyle=":", alpha=0.55, zorder=2,
            ))
    for x, y in cylinders:
        ax.add_patch(Circle(
            (x, y), cyl_radius,
            linewidth=0.8, edgecolor="black", facecolor="#e87020",
            alpha=0.40, zorder=2,
        ))
        if tighten > 0:
            ax.add_patch(Circle(
                (x, y), cyl_radius + tighten,
                linewidth=0.8, edgecolor="#e87020", facecolor="none",
                linestyle=":", alpha=0.55, zorder=2,
            ))


# ── load zarr dataset ─────────────────────────────────────────────────────
path = sim_framework_path("isaac", "dataset", "avoiding_crazyflie", "data", "zarr", "env_000.zarr")
g = zarr.open_group(path, mode="r")

print("arrays:", list(g.array_keys()))
for k in g.array_keys():
    print(k, g[k].shape, g[k].dtype)

rgb    = g["rgb"]                      # zarr array (T, H, W, 3)
states = g["states"][:]                # (T, 13)
pos    = states[:, :3]                 # (T, 3)  x y z

term   = g["terminals"][:].astype(np.uint8)
ends   = np.where(term == 1)[0]
starts = np.r_[0, ends[:-1] + 1]

lengths = ends - starts + 1
print("episode lengths (first 10):", lengths[:10])
print("min/mean/max:", lengths.min(), lengths.mean(), lengths.max())
print("num episodes:", int((term == 1).sum()))
print("first terminal indices:", np.where(term == 1)[0][:10])



def plot_episode_frames(zarr_path, episode_idx=0, num_frames=12):
    import zarr
    g = zarr.open_group(zarr_path, mode="r")
    
    terminals = g["terminals"][:].astype("uint8")
    ends = np.where(terminals == 1)[0]
    starts = np.concatenate([[0], ends[:-1] + 1])
    
    ep_start = starts[episode_idx]
    ep_end   = ends[episode_idx]
    ep_len   = ep_end - ep_start + 1
    
    # pick evenly spaced frames across the episode
    indices = np.linspace(ep_start, ep_end, num_frames, dtype=int)
    
    fig, axes = plt.subplots(2, num_frames // 2, figsize=(18, 5))
    axes = axes.flatten()
    
    for i, idx in enumerate(indices):
        frame = g["rgb"][idx]   # (H, W, 3) uint8
        axes[i].imshow(frame)
        axes[i].set_title(f"t={idx - ep_start}", fontsize=9)
        axes[i].axis("off")
    
    fig.suptitle(f"Episode {episode_idx} — {ep_len} steps", fontsize=11)
    plt.tight_layout()
    plt.show()
    
    # also print the mean brightness per frame for the first 10 frames
    print(f"\nFirst 10 frame brightness (episode {episode_idx}):")
    for i in range(min(10, ep_len)):
        frame = g["rgb"][ep_start + i].astype(np.float32) / 255.0
        print(f"  t={i:03d}: mean={frame.mean():.4f}  min={frame.min():.4f}  max={frame.max():.4f}")

# run it
plot_episode_frames(sim_framework_path("isaac", "dataset", "avoiding_crazyflie", "data", "zarr", "env_000.zarr"), episode_idx=0, num_frames=30)


fig_xz, ax_xz_solo = plt.subplots(figsize=(10, 4))
cmap = plt.cm.Blues
n_ep = len(starts)
for i, (s, e) in enumerate(zip(starts, ends)):
    xz = pos[s:e+1, ::2]   # columns 0 (x) and 2 (z)
    color = cmap(0.35 + 0.55 * i / max(n_ep - 1, 1))
    ax_xz_solo.plot(xz[:, 0], xz[:, 1], color=color, linewidth=0.7, alpha=0.65)
add_corridor_xz(ax_xz_solo)
add_goal_line(ax_xz_solo, plane="xz")
ax_xz_solo.set_xlabel("x (m)")
ax_xz_solo.set_ylabel("z (m)")
ax_xz_solo.set_xlim(*X_LIM_XZ)
ax_xz_solo.set_ylim(*Z_LIM_XZ)
ax_xz_solo.grid(True, alpha=0.25)
fig_xz.tight_layout()
z_path = "plots/datasetxz.pdf"
fig_xz.savefig(z_path, dpi=150, bbox_inches="tight")
plt.show()

fig, (ax_xy, ax_xz) = plt.subplots(1, 2, figsize=(14, 5))
cmap = plt.cm.Blues
n_ep = len(starts)

for i, (s, e) in enumerate(zip(starts, ends)):
    xy = pos[s:e+1, :2]
    xz = pos[s:e+1, ::2]   # columns 0 (x) and 2 (z)
    color = cmap(0.35 + 0.55 * i / max(n_ep - 1, 1))

    ax_xy.plot(xy[:, 0], xy[:, 1], color=color, linewidth=0.7, alpha=0.65)
    ax_xz.plot(xz[:, 0], xz[:, 1], color=color, linewidth=0.7, alpha=0.65)

# start / goal scatter (one per episode)
ax_xy.scatter([pos[s, 0] for s in starts], [pos[s, 1] for s in starts],
              s=8, color="#2ecc71", zorder=5, label="Start")
ax_xy.scatter([pos[e, 0] for e in ends],   [pos[e, 1] for e in ends],
              s=8, color="#e74c3c", zorder=5, label="Goal (gate)")

# --- corridor geometry ---
add_corridor_xy(ax_xy)
add_corridor_xz(ax_xz)
add_goal_line(ax_xy, plane="xy")
add_goal_line(ax_xz, plane="xz")

# --- obstacles (uncomment when obstacles are active at collection time) ---
add_obstacles(ax_xy, BOXES, CYLINDERS)

# --- labels & limits ---
ax_xy.set_xlabel("x (m)"); ax_xy.set_ylabel("y (m)")
ax_xy.set_xlim(*X_LIM_XY);  ax_xy.set_ylim(*Y_LIM_XY)
ax_xy.set_aspect("equal", adjustable="box")
ax_xy.legend(fontsize=8, loc="upper left")
ax_xy.grid(True, alpha=0.25)

ax_xz.set_xlabel("x (m)"); ax_xz.set_ylabel("z (m)")
ax_xz.set_xlim(*X_LIM_XZ);  ax_xz.set_ylim(*Z_LIM_XZ)
ax_xz.set_aspect("equal", adjustable="box")
ax_xz.grid(True, alpha=0.25)

fig.tight_layout()
fig.savefig("plots/dataset.pdf", dpi=150, bbox_inches="tight")
plt.show()
# 3D trajectory view (x, y, z)
fig3d = plt.figure(figsize=(12, 8))
ax3d  = fig3d.add_subplot(111, projection="3d")
cmap  = plt.cm.Blues
n_ep  = len(starts)

for i, (s, e) in enumerate(zip(starts, ends)):
    xyz   = pos[s:e+1, :3]
    color = cmap(0.35 + 0.55 * i / max(n_ep - 1, 1))
    ax3d.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2],
              color=color, linewidth=0.8, alpha=0.75)

# corridor floor outline (z=0 rectangle)
cx = [0, CORRIDOR_END, CORRIDOR_END, 0, 0]
cy = [-1.0, -1.0, 1.0, 1.0, -1.0]
cz = [0, 0, 0, 0, 0]
ax3d.plot(cx, cy, cz, color="#555555", linewidth=1.0, linestyle="--", alpha=0.4)

# goal plane at x=4.0
gy = np.linspace(-1.0, 1.0, 2)
gz = np.linspace(0.0, WALL_HEIGHT, 2)
GY, GZ = np.meshgrid(gy, gz)
GX = np.full_like(GY, 4.0)
ax3d.plot_surface(GX, GY, GZ, color="#00aa00", alpha=0.15)

# obstacle centres in 3D (uncomment when active)
if BOXES:
    bx = np.array([[x, y, WALL_HEIGHT/2] for x, y in BOXES])
    ax3d.scatter(bx[:, 0], bx[:, 1], bx[:, 2],
                 c="tab:red", marker="s", s=40, label="Box obstacles")
if CYLINDERS:
    cx_ = np.array([[x, y, WALL_HEIGHT/2] for x, y in CYLINDERS])
    ax3d.scatter(cx_[:, 0], cx_[:, 1], cx_[:, 2],
                 c="tab:orange", marker="o", s=40, label="Cylinder obstacles")

ax3d.set_xlabel("x (m)"); ax3d.set_ylabel("y (m)"); ax3d.set_zlabel("z (m)")
ax3d.set_xlim(0, CORRIDOR_END)
ax3d.set_ylim(-1.1, 1.1)
ax3d.set_zlim(0, WALL_HEIGHT)
if BOXES or CYLINDERS:
    ax3d.legend(loc="best")
fig3d.tight_layout()
fig3d.savefig("plots/dataset3d.pdf", dpi=150, bbox_inches="tight")
plt.show()
