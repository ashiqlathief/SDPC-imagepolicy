"""Zarr dataset writer shared by isaac/scripts/quadcopter.py's own collection
loop and any other expert/policy-driven data-collection script (e.g. the
exp2vla RL-expert rollout) that needs to write episodes in the same on-disk
layout consumed by config.avoiding-crazyflie's dataset loader:

  env_XXX.zarr/
    rgb          (T, H, W, 3)      uint8
    depth        (T, H, W)         float32   -- only if use_depth
    targets      (T, target_dim)   float32   -- only if target_dim is set
    states       (T, state_dim)    float32   -- [pos(3), quat_xyzw(4), linvel(3), angvel_w(3)]
    terminals    (T,)              uint8     -- 1 on the last step of each episode
    episode_id   (T,)              int32

No IsaacLab imports here on purpose -- this module must stay importable from
a script that has already launched its own AppLauncher/SimulationApp without
triggering a second one.
"""
import os
import numpy as np
import zarr


class ZarrEpisodeWriter:
    def __init__(self, root_dir: str, num_envs: int, img_h: int, img_w: int,
                 state_dim: int, chunk_t: int = 256, use_depth: bool = False,
                 target_dim: int | None = None):
        self.root_dir = root_dir
        self.num_envs = num_envs
        self.img_h = img_h
        self.img_w = img_w
        self.state_dim = state_dim
        self.use_depth = use_depth
        self.target_dim = target_dim

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
            if self.target_dim is not None and "targets" not in g:
                g.create_array(
                    "targets",
                    shape=(0, self.target_dim),
                    chunks=(chunk_t, self.target_dim),
                    dtype="float32",
                )
            g.attrs["use_depth"] = self.use_depth
            g.attrs["has_targets"] = self.target_dim is not None
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

    def append_episode(self, ep_states, ep_images, ep_depths=None, ep_targets=None):
        """
        ep_states:  (T, N, state_dim)
        ep_images:  (T, N, H, W, 3) uint8
        ep_depths:  (T, N, H, W) float32, required if self.use_depth
        ep_targets: (T, N, target_dim) float32, required if self.target_dim is set
        """
        T, N, _ = ep_states.shape
        assert N == self.num_envs
        if self.use_depth:
            assert ep_depths is not None, "use_depth=True but no depth data was provided"
        if self.target_dim is not None:
            assert ep_targets is not None, "target_dim is set but no target data was provided"

        terminals = np.zeros((T,), dtype=np.uint8)
        terminals[-1] = 1

        for env_id in range(self.num_envs):
            g = self.groups[env_id]

            rgb = ep_images[:, env_id, ...]          # (T, H, W, 3)
            st  = ep_states[:, env_id, :]             # (T, state_dim)

            eid = self._episode_counter[env_id]
            ep_ids = np.full((T,), eid, dtype=np.int32)

            g["rgb"].append(rgb)
            if self.use_depth:
                depth = ep_depths[:, env_id, ...]    # (T, H, W)
                g["depth"].append(depth.astype(np.float32))
            if self.target_dim is not None:
                tgt = ep_targets[:, env_id, :]       # (T, target_dim)
                g["targets"].append(tgt.astype(np.float32))
            g["states"].append(st.astype(np.float32))
            g["terminals"].append(terminals)
            g["episode_id"].append(ep_ids)

            self._episode_counter[env_id] += 1
