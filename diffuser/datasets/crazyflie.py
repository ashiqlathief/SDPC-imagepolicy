from collections import namedtuple
import importlib
import os
import glob
import cv2
import numpy as np
import torch
import zarr

from diffuser.utils.path import project_path
from .normalization import LimitsNormalizer
_scene_cfg_module = importlib.import_module("config.avoiding-crazyflie")

# Keep the same “shape” of outputs as other datasets
Batch = namedtuple("Batch", "trajectories conditions")
RewardBatch = namedtuple("RewardBatch", "trajectories conditions returns")


class CrazyflieImageDataset(torch.utils.data.Dataset):
    """
    Loads Crazyflie image+action data saved as Zarr stores:
        isaac/dataset/avoiding_crazyflie/data/zarr/env_XXX.zarr

    Returns samples:
        trajectories: (H, action_dim)  float32  (normalized to [-1,1] via LimitsNormalizer)
        conditions: {
            "obs_rgb": (To, 3, H_img, W_img) float32 in [0,1]
        }
    """

    def __init__(
        self,
        env="avoiding-crazyflie",
        horizon=16,
        n_obs_steps=2,              # To
        normalizer="LimitsNormalizer",
        preprocess_fns=None,        # unused, kept for signature compatibility
        use_padding=True,           # unused here
        max_path_length=250,        # used only to optionally truncate episodes
        discount=0.99,              # used only if include_returns=True (we return zeros by default)
        returns_scale=100.0,
        include_returns=False,
        zarr_subdir="zarr",
        data_subdir="data",         # isaac/dataset/avoiding_crazyflie/<data_subdir>/<zarr_subdir>
        stats_path=None,
        use_pose_cond=False,
        **_legacy_kwargs,
    ):
        super().__init__()
        if _legacy_kwargs:
            print(f"[CrazyflieImageDataset] Ignoring legacy kwargs from an older "
                  f"dataset_config.pkl: {sorted(_legacy_kwargs)}")
        self.env = env
        self.horizon = int(horizon)         # H
        self.n_obs_steps = int(n_obs_steps) # To
        self.include_returns = include_returns
        self.discount = float(discount)
        self.returns_scale = float(returns_scale)
        self.max_path_length = int(max_path_length)
        self.use_pose_cond = bool(use_pose_cond)
        self.pose_normalizer = None

        if preprocess_fns is None:
            preprocess_fns = []

        # --- Find Zarr stores ---
        data_dir = project_path("isaac", "dataset", "avoiding_crazyflie", data_subdir, zarr_subdir)
        if not os.path.isdir(data_dir):
            mins = maxs = None
            source = None
            if stats_path is not None and os.path.isfile(stats_path):
                ckpt = torch.load(stats_path, map_location="cpu")
                if "action_min" in ckpt and "action_max" in ckpt:
                    mins, maxs = np.asarray(ckpt["action_min"]), np.asarray(ckpt["action_max"])
                    source = stats_path
            if mins is None and stats_path is not None:
                legacy_path = os.path.join(os.path.dirname(stats_path), "normalizer_stats.npz")
                if os.path.isfile(legacy_path):
                    legacy = np.load(legacy_path)
                    mins, maxs = legacy["mins"], legacy["maxs"]
                    source = legacy_path

            if mins is not None:
                X = np.stack([mins, maxs], axis=0).astype(np.float32)
                self.action_normalizer = LimitsNormalizer(X)
                self.action_dim = int(mins.shape[0])
                self.observation_dim = 0
                self.goal_dim = 0
                self.groups = []
                self.episodes = []
                self.indices = []
                print(f"[CrazyflieImageDataset] Zarr folder not found; loaded normalizer stats from: {source}")
                return
            raise FileNotFoundError(f"Zarr folder not found: {data_dir}")

        zarr_paths = sorted(glob.glob(os.path.join(data_dir, "env_*.zarr")))
        if len(zarr_paths) == 0:
            raise FileNotFoundError(f"No env_*.zarr stores found in: {data_dir}")

        self.groups = [zarr.open_group(p, mode="r") for p in zarr_paths]

        g0 = self.groups[0]
        raw_h, raw_w = int(g0["rgb"].shape[1]), int(g0["rgb"].shape[2])
        self.img_size = int(getattr(_scene_cfg_module, "VIT_IMG_SIZE", 96)) #from cfg file 96 or 128
        self.img_h = self.img_w = self.img_size
        self._raw_img_h, self._raw_img_w = raw_h, raw_w

        self.pos_slice = slice(0, 3)   # CHANGE if needed
        self.action_dim = 3            # (dx, dy, dz)
        self.observation_dim = 0
        self.goal_dim = 0
        self.episodes = []
        for gi, g in enumerate(self.groups):
            terminals = g["terminals"][:].astype(np.uint8)
            ends = np.where(terminals == 1)[0].tolist()
            start = 0
            for end in ends:
                ep_len = end - start + 1

                # still need at least To+H steps to form one training sample
                if ep_len >= (self.n_obs_steps + self.horizon):
                    self.episodes.append((gi, start, end))
                else:
                    print(f"[WARN] Episode gi={gi} start={start} end={end} "
                          f"too short ({ep_len} < {self.n_obs_steps + self.horizon}), skipping")
                start = end + 1
            # handle any trailing data after the last terminal
            if start < len(terminals):
                ep_len = len(terminals) - start
                if ep_len >= (self.n_obs_steps + self.horizon):
                    self.episodes.append((gi, start, len(terminals) - 1))
                    print(f"[WARN] Trailing data after last terminal: gi={gi} "
                          f"start={start} end={len(terminals)-1}, included")
        if len(self.episodes) == 0:
            raise RuntimeError(
                f"No episodes long enough for To+H = {self.n_obs_steps}+{self.horizon}. "
                "Collect longer episodes or lower To/H."
            )

        self.indices = []
        for (gi, ep_start, ep_end) in self.episodes:
            t_min = ep_start + (self.n_obs_steps - 1)
            t_max = ep_end - self.horizon  # t_start + H <= ep_end
            for t_start in range(t_min, t_max + 1):
                self.indices.append((gi, t_start))

        if len(self.indices) == 0:
            raise RuntimeError("Indices empty after episode parsing. Check To/H and terminals.")

        # --- Build action normalizer (LimitsNormalizer) over all actions in all stores ---
        # Compute global min/max without loading everything at once
        a_mins = None
        a_maxs = None
        for gi, g in enumerate (self.groups):
            s = g["states"]  # (T, state_dim): [pos(3), quat_xyzw(4), linvel_w(3), angvel_w(3)]
            pos_all = s[:, self.pos_slice]

            for (ep_gi, ep_start, ep_end) in self.episodes:
                if ep_gi != gi:
                    continue

                p = pos_all[ep_start : ep_end + 1]                    # fully within real episode
                act = p[1:] - p[:-1]                                  # (>=0, action_dim)

                if act.shape[0] == 0:
                    continue

                cmin = act.min(axis=0)
                cmax = act.max(axis=0)

                a_mins = cmin if a_mins is None else np.minimum(a_mins, cmin)
                a_maxs = cmax if a_maxs is None else np.maximum(a_maxs, cmax)

        X = np.stack([a_mins, a_maxs], axis=0).astype(np.float32)
        self.action_normalizer = LimitsNormalizer(X)

    def __len__(self):
        return len(self.indices)

    def _center_crop_resize(self, frames, interpolation):
        T, h, w = frames.shape[:3]
        if h == self.img_size and w == self.img_size:
            return frames
        side = min(h, w)
        y0, x0 = (h - side) // 2, (w - side) // 2
        cropped = frames[:, y0:y0 + side, x0:x0 + side, ...]
        out = np.empty((T, self.img_size, self.img_size) + frames.shape[3:], dtype=frames.dtype)
        for t in range(T):
            out[t] = cv2.resize(cropped[t], (self.img_size, self.img_size), interpolation=interpolation)
        return out

    def __getitem__(self, idx):
        gi, t_start = self.indices[idx]
        g = self.groups[gi]

        To = self.n_obs_steps
        H = self.horizon

        # --- condition: image window ending at t_start --- rgb in zarr: (T, H, W, 3) uint8
        rgb = g["rgb"][t_start - To + 1 : t_start + 1]  # (To, H, W, 3) uint8
        rgb = self._center_crop_resize(rgb, cv2.INTER_AREA)  # (To, img_size, img_size, 3) uint8
        rgb = (rgb.astype(np.float32) / 255.0).transpose(0, 3, 1, 2)  # (To, 3, img_size, img_size)

        states = g["states"][t_start : t_start + H + 1].astype(np.float32)  # (H+1, state_dim)
        pos = states[:, self.pos_slice]                                 # (H+1, 3)
        actions = pos[1:] - pos[:-1]                                    # (H, action_dim), each spanning one frame

        actions = self.action_normalizer.normalize(actions.astype(np.float32))
        conditions = {"obs_rgb": rgb}

        if self.use_pose_cond:
            pose_now = states[0, self.pos_slice]                  # (3,) position at t_start
            pose_target = g["targets"][t_start].astype(np.float32)  # (3,) this episode's goal
            conditions["pose_now"] = pose_now
            conditions["pose_target"] = pose_target

        if self.include_returns:
            returns = np.array([0.0 / self.returns_scale], dtype=np.float32)
            return RewardBatch(actions, conditions, returns)

        return Batch(actions, conditions)
