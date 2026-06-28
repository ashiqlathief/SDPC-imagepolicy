import zarr, numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from isaac.dataset.sim_path import sim_framework_path

# Set up the plot style for better aesthetics
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 10)
plt.rcParams['font.size'] = 12

path = sim_framework_path("isaac", "dataset", "avoiding_crazyflie", "data", "zarr", "env_000.zarr")
g = zarr.open_group(path, mode="r")

WALL_HEIGHT = 1.0
BOXES = [
    # (3.5,  0.2, WALL_HEIGHT / 2.0),
    # (2.2, -0.2, WALL_HEIGHT / 2.0),
    # (1.6,  0.2, WALL_HEIGHT / 2.0),
    # (3.7, -0.3, WALL_HEIGHT / 2.0),
    # (0.6,  0.1, WALL_HEIGHT / 2.0),
    # (1.0, -0.3, WALL_HEIGHT / 2.0),
]
CYLINDERS = [
    # (1.4, -0.2, WALL_HEIGHT / 2.0),
    # (2.8,  0.3, WALL_HEIGHT / 2.0),
    # (1.2, 0.4, WALL_HEIGHT / 2.0),
    # (3.5,  0.4, WALL_HEIGHT / 2.0),
    # (2.0,  0.4, WALL_HEIGHT / 2.0),
    # (2.5,  0.3, WALL_HEIGHT / 2.0),
]

print("arrays:", list(g.array_keys()))
for k in g.array_keys():
    print(k, g[k].shape, g[k].dtype)

rgb = g["rgb"]  # zarr array
states = g["states"][:]          # (T, 13)
pos = states[:, :3]

term = g["terminals"][:].astype(np.uint8)
ends = np.where(term == 1)[0]
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


plt.figure()
ax = plt.gca()
colors = plt.cm.Set1(np.linspace(0, 1, len(starts)))
for i, (s, e) in enumerate(zip(starts, ends)):
    xz = pos[s:e+1, ::2]
    plt.scatter(xz[:,0], xz[:,1], color=colors[i], s=5, alpha=0.7)
    plt.scatter(xz[0,0], xz[0,1], marker="o")
    plt.scatter(xz[-1,0], xz[-1,1], marker="x")
plt.xlabel("x")
plt.ylabel("z")
plt.title("XZ trajectories (start=o, end=x) per episode")
plt.axis("equal")
plt.ylim(bottom=0)  # remove negative z
z_path = "plots/datasetxz.png"
plt.savefig(z_path, dpi=150)
plt.show()

fig, (ax_xy, ax_xz) = plt.subplots(1, 2, figsize=(14, 6))
colors = plt.cm.Set1(np.linspace(0, 1, len(starts)))

for i, (s, e) in enumerate(zip(starts, ends)):
    xy = pos[s:e+1, :2]
    xz = pos[s:e+1, ::2]  # columns 0 (x) and 2 (z)

    # --- XY plot ---
    ax_xy.scatter(xy[:, 0], xy[:, 1], color=colors[i], s=5, alpha=0.7)
    ax_xy.scatter(xy[0, 0], xy[0, 1], marker="o", color=colors[i])
    ax_xy.scatter(xy[-1, 0], xy[-1, 1], marker="x", color=colors[i])

    # --- XZ plot ---
    ax_xz.scatter(xz[:, 0], xz[:, 1], color=colors[i], s=5, alpha=0.7)
    ax_xz.scatter(xz[0, 0], xz[0, 1], marker="o", color=colors[i])
    ax_xz.scatter(xz[-1, 0], xz[-1, 1], marker="x", color=colors[i])

# --- Obstacle overlay XY ---
box_size_xy = 0.20
cyl_radius = 0.06
for x, y, _ in BOXES:
    ax_xy.add_patch(Rectangle(
        (x - box_size_xy / 2.0, y - box_size_xy / 2.0),
        box_size_xy, box_size_xy,
        facecolor="tab:red", edgecolor="black", alpha=0.35, linewidth=1.0,
    ))
for x, y, _ in CYLINDERS:
    ax_xy.add_patch(Circle(
        (x, y), cyl_radius,
        facecolor="tab:orange", edgecolor="black", alpha=0.45, linewidth=1.0,
    ))

# --- Obstacle overlay XZ ---
for x, _, z in BOXES:
    ax_xz.add_patch(Rectangle(
        (x - box_size_xy / 2.0, z - box_size_xy / 2.0),
        box_size_xy, box_size_xy,
        facecolor="tab:red", edgecolor="black", alpha=0.35, linewidth=1.0,
    ))
for x, _, z in CYLINDERS:
    ax_xz.add_patch(Circle(
        (x, z), cyl_radius,
        facecolor="tab:orange", edgecolor="black", alpha=0.45, linewidth=1.0,
    ))

# --- Labels ---
ax_xy.set_xlabel("x"); ax_xy.set_ylabel("y")
ax_xy.set_title("XY trajectories (start=o, end=x) per episode")
ax_xy.set_aspect("equal")

ax_xz.set_xlabel("x"); ax_xz.set_ylabel("z")
ax_xz.set_title("XZ trajectories (start=o, end=x) per episode")
ax_xz.set_aspect("equal")

plt.tight_layout()
plt.savefig("plots/dataset.png", dpi=150)
plt.show()
# 3D trajectory view (X-Y-Z) with obstacle centers
fig = plt.figure(figsize=(12, 9))
ax3d = fig.add_subplot(111, projection="3d")

colors = plt.cm.Set1(np.linspace(0, 1, len(starts)))
for i, (s, e) in enumerate(zip(starts, ends)):
    xyz = pos[s:e+1, :3]
    ax3d.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], color=colors[i], linewidth=1.0, alpha=0.9)
    ax3d.scatter(xyz[0, 0], xyz[0, 1], xyz[0, 2], marker="o", color=colors[i], s=20)
    ax3d.scatter(xyz[-1, 0], xyz[-1, 1], xyz[-1, 2], marker="x", color=colors[i], s=25)

box_centers = np.array(BOXES)
cyl_centers = np.array(CYLINDERS)
# ax3d.scatter(box_centers[:, 0], box_centers[:, 1], box_centers[:, 2], c="tab:red", marker="s", s=40, label="boxes")
# ax3d.scatter(cyl_centers[:, 0], cyl_centers[:, 1], cyl_centers[:, 2], c="tab:orange", marker="o", s=40, label="cylinders")

ax3d.set_xlabel("x")
ax3d.set_ylabel("y")
ax3d.set_zlabel("z")
ax3d.set_title("3D trajectories (x, y, z) with obstacle centers")
ax3d.legend(loc="best")
plt.tight_layout()
z_path = f"plots/dataset3d.png"
plt.savefig(z_path, dpi=150)
plt.show()
