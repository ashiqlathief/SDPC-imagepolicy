import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

from diffuser.utils.path import project_path

def load_all_files(data_dir: str):
    files = sorted(f for f in os.listdir(data_dir) if f.endswith(".pkl"))
    if not files:
        raise FileNotFoundError(f"No .pkl files found in {data_dir}")

    datasets = []
    for f in files:
        path = os.path.join(data_dir, f)
        with open(path, "rb") as fh:
            datasets.append(pickle.load(fh))
        # print(f"[INFO] Loaded {path}")
    return datasets

def collect_trajectories_from_all_envs(datasets):
    all_states = []
    all_targets = []
    all_motor_forces = []

    for data in datasets:
        states_list = data["states"]
        targets_list = data["targets"]
        motor_list = data.get("actions_motor_forces", None)

        for idx, (s, tg) in enumerate(zip(states_list, targets_list)):
            all_states.append(np.asarray(s))
            all_targets.append(np.asarray(tg))

            if motor_list is not None:
                all_motor_forces.append(np.asarray(motor_list[idx]))  # (T,4)
            else:
                all_motor_forces.append(None)

    return all_states, all_targets, all_motor_forces

def plot_motor_forces_over_time(all_motor_forces, max_eps=20):
    fig = plt.figure(figsize=(8, 4))

    shown = 0
    for i, mf in enumerate(all_motor_forces):
        if mf is None:
            continue
        mf = np.asarray(mf)
        if mf.ndim != 2 or mf.shape[1] != 4:
            continue

        t = np.arange(mf.shape[0])
        # plot only rotor-0 for readability across many episodes
        plt.plot(t, mf[:, 0], alpha=0.5, label=f"ep {i} m1" if shown < 10 else None)
        shown += 1
        if shown >= max_eps:
            break

    plt.xlabel("timestep")
    plt.ylabel("motor force")
    plt.title("Motor force (m1) over time (subset of episodes)")
    plt.grid(True)
    plt.tight_layout()
    return fig

def plot_xyz_trajectories(all_states, all_targets=None):
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")

    for states in all_states:
        x, y, z = states[:, 0], states[:, 1], states[:, 2]
        ax.plot3D(x, y, z, alpha=0.6)
        ax.scatter3D(x[0], y[0], z[0], c="green", s=10)
        ax.scatter3D(x[-1], y[-1], z[-1], c="red", s=10)

    if all_targets:
        for tg in all_targets:
            tx, ty, tz = tg[-1]
            tx, ty, tz = tg[:, 0], tg[:, 1], tg[:, 2]
            ax.scatter3D(tx, ty, tz, c="orange", marker="x", s=40)

    # Equal scale
    xs = np.concatenate([s[:, 0] for s in all_states])
    ys = np.concatenate([s[:, 1] for s in all_states])
    zs = np.concatenate([s[:, 2] for s in all_states])
    max_range = np.max([xs.ptp(), ys.ptp(), zs.ptp()]) / 2
    mid_x, mid_y, mid_z = xs.mean(), ys.mean(), zs.mean()
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title("3D Quadcopter Trajectories")

    plt.tight_layout()
    return fig

def plot_motor_forces_single_episode(mf, ep_idx=0):
    mf = np.asarray(mf)
    fig = plt.figure(figsize=(8, 4))
    t = np.arange(mf.shape[0])
    plt.plot(t, mf[:, 0], label="m1")
    plt.plot(t, mf[:, 1], label="m2")
    plt.plot(t, mf[:, 2], label="m3")
    plt.plot(t, mf[:, 3], label="m4")
    plt.xlabel("timestep")
    plt.ylabel("motor force")
    plt.title(f"Motor forces (episode {ep_idx})")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    return fig


def plot_xy_trajectories(all_states, all_targets=None):
    fig = plt.figure(figsize=(6, 6))

    for states in all_states:
        x, y = states[:, 0], states[:, 1]
        plt.plot(x, y, marker='.', linestyle='None', markersize=2,alpha=0.5)
        plt.scatter(x[0], y[0], c="green", s=10)
        plt.scatter(x[-1], y[-1], c="red", s=10)

    if all_targets:
        for tg in all_targets:
            tx, ty = tg[-1, 0], tg[-1, 1]
            plt.scatter(tx, ty, c="orange", marker="x", s=40)

    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("XY Trajectories – All PKL Files")
    plt.axis("equal")
    plt.grid(True)
    plt.tight_layout()
    return fig

def plot_motor_forces_one_episode_separate(all_motor_forces, ep_to_plot=0):
    mf = all_motor_forces[ep_to_plot]
    if mf is None:
        raise ValueError(f"Episode {ep_to_plot} has no motor force data.")

    mf = np.asarray(mf)
    if mf.ndim != 2 or mf.shape[1] != 4:
        raise ValueError(f"Episode {ep_to_plot} motor forces shape is {mf.shape}, expected (T,4).")

    figs = []
    t = np.arange(mf.shape[0])
    for j, name in enumerate(["m1", "m2", "m3", "m4"]):
        fig = plt.figure(figsize=(8, 3))
        plt.plot(t, mf[:, j])
        plt.xlabel("timestep")
        plt.ylabel("motor force")
        plt.title(f"{name} motor force (episode {ep_to_plot})")
        plt.grid(True)
        plt.tight_layout()
        figs.append((name, fig))
    return figs

def plot_z_over_time(all_states, max_eps=20):
    fig = plt.figure(figsize=(7, 4))

    for i, states in enumerate(all_states[:max_eps]):
        z = states[:, 2]
        t = np.arange(len(z))
        plt.plot(t, z, label=f"ep {i}")

    plt.xlabel("timestep")
    plt.ylabel("z [m]")
    plt.title("Altitude Over Time")
    plt.legend(fontsize=7)
    plt.grid(True)
    plt.tight_layout()
    return fig

def plot_pos_error_over_time(all_states, all_targets, max_eps=20, plot='norm'):
    """
    plot='norm'  -> ||target - pos|| over time
    plot='xyz'   -> x/y/z error components over time
    """
    fig = plt.figure(figsize=(8, 4))

    shown = 0
    for states, tg in zip(all_states, all_targets):
        if shown >= max_eps:
            break

        states = np.asarray(states)
        tg = np.asarray(tg)

        pos = states[:, :3]  # x,y,z (your plots use columns 0,1,2)

        # handle target shapes:
        # - constant target: (1,3) or (3,)
        # - time-varying target: (T,3)
        if tg.ndim == 2 and tg.shape[0] == 1:
            target = np.repeat(tg[0:1, :3], repeats=pos.shape[0], axis=0)
        elif tg.ndim == 1:
            target = np.repeat(tg[None, :3], repeats=pos.shape[0], axis=0)
        else:
            T = min(pos.shape[0], tg.shape[0])
            pos = pos[:T]
            target = tg[:T, :3]

        err = target - pos
        t = np.arange(err.shape[0])

        if plot == 'xyz':
            plt.plot(t, err[:, 0], alpha=0.6, label='x err' if shown == 0 else None)
            plt.plot(t, err[:, 1], alpha=0.6, label='y err' if shown == 0 else None)
            plt.plot(t, err[:, 2], alpha=0.6, label='z err' if shown == 0 else None)
        elif plot == 'x':
            plt.plot(t, err[:, 0], alpha=0.6, label='x err' if shown == 0 else None)
        elif plot == 'y':
            plt.plot(t, err[:, 1], alpha=0.6, label='y err' if shown == 0 else None)
        elif plot == 'z':
            plt.plot(t, err[:, 2], alpha=0.6, label='z err' if shown == 0 else None)
        else:
            err_norm = np.linalg.norm(err, axis=1)
            plt.plot(t, err_norm, alpha=0.6)

        shown += 1

    plt.xlabel("timestep")
    plt.ylabel("pos error" if plot == 'xyz' else "||pos error||")
    plt.title(f"Position error over time ({plot})")
    plt.grid(True)
    if plot == 'xyz':
        plt.legend()
    plt.tight_layout()
    return fig


def main():
    # Locations
    data_dir = project_path("isaac", "dataset", "avoiding_crazyflie", "data")
    # data_dir = project_path("isaac", "scripts")
    print("Resolved data path:", data_dir)

    figures_dir = project_path("isaac", "dataset", "avoiding_crazyflie", "figures")
    os.makedirs(figures_dir, exist_ok=True)

    # Load & collect
    datasets = load_all_files(data_dir)
    all_states, all_targets, all_motor_forces = collect_trajectories_from_all_envs(datasets)
    print(f"[INFO] Total episodes loaded: {len(all_states)}")

    # --- Plot ---
    # fig_xyz = plot_xyz_trajectories(all_states, all_targets)
    fig_xy  = plot_xy_trajectories(all_states, all_targets)
    fig_z   = plot_z_over_time(all_states, max_eps=70)
    # fig_m1 = plot_motor_forces_over_time(all_motor_forces, max_eps=20)
    # fig_err = plot_pos_error_over_time(all_states, all_targets, max_eps=70, plot='norm')
    # fig_err_xyz = plot_pos_error_over_time(all_states, all_targets, max_eps=20, plot='xyz')
    # fig_err_x = plot_pos_error_over_time(all_states, all_targets, max_eps=20, plot='x')
    # fig_err_y = plot_pos_error_over_time(all_states, all_targets, max_eps=20, plot='y')
    # fig_err_z = plot_pos_error_over_time(all_states, all_targets, max_eps=20, plot='z')

    # for ep_idx, mf in enumerate(all_motor_forces):
    #     if mf is not None:
    #         fig_m_all = plot_motor_forces_single_episode(mf, ep_idx=ep_idx)
    #         fig_m_all.savefig(os.path.join(figures_dir, f"motor_forces_ep{ep_idx:03d}.png"))
    #         break
    
    # ep_to_plot = 0  # choose episode index

    # figs = plot_motor_forces_one_episode_separate(all_motor_forces, ep_to_plot=ep_to_plot)
    # for name, fig in figs:
    #     fig.savefig(os.path.join(figures_dir, f"motor_{name}_ep{ep_to_plot:03d}.png"))
    #     plt.close(fig)

    # print(f"[INFO] Saved motor plots (m1..m4) for episode {ep_to_plot} to {figures_dir}")

    # --- Save ---
    # fig_xyz.savefig(os.path.join(figures_dir, "xyz_plot.png"))
    fig_xy.savefig(os.path.join(figures_dir, "xy_plot.png"))
    fig_z.savefig(os.path.join(figures_dir, "z_plot.png"))
    # fig_m1.savefig(os.path.join(figures_dir, "motor_m1_over_time.png"))
    # fig_err.savefig(os.path.join(figures_dir, "pos_error_norm.png"))
    # fig_err_xyz.savefig(os.path.join(figures_dir, "pos_error_xyz.png"))
    # fig_err_x.savefig(os.path.join(figures_dir, "pos_error_x.png"))
    # fig_err_y.savefig(os.path.join(figures_dir, "pos_error_y.png"))
    # fig_err_z.savefig(os.path.join(figures_dir, "pos_error_z.png"))

    print(f"[INFO] Saved plots to {figures_dir}")

    plt.show()


if __name__ == "__main__":
    main()
