from collections import namedtuple
import os
import glob
import numpy as np
import torch
import zarr

from diffuser.utils.path import project_path
from .normalization import LimitsNormalizer

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
            "rgb": (To, 3, H_img, W_img) float32 in [0,1]
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
        stride=1,                   # frames-per-action-step at collection rate (see `dt`)
        dt=0.005,                   # sim dt the raw zarr frames were collected at
    ):
        super().__init__()
        self.env = env
        self.horizon = int(horizon)         # H
        self.n_obs_steps = int(n_obs_steps) # To
        self.include_returns = include_returns
        self.discount = float(discount)
        self.returns_scale = float(returns_scale)
        self.max_path_length = int(max_path_length)
        self.stride = int(stride)
        self.dt = float(dt)
        # The wall-clock time one predicted action step spans. eval's sim dt
        # must equal this for a0_real to mean what the model thinks it means.
        self.control_dt = self.stride * self.dt

        if preprocess_fns is None:
            preprocess_fns = []

        # --- Find Zarr stores ---
        data_dir = project_path("isaac", "dataset", "avoiding_crazyflie", "data", zarr_subdir)
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(f"Zarr folder not found: {data_dir}")

        zarr_paths = sorted(glob.glob(os.path.join(data_dir, "env_*.zarr")))
        if len(zarr_paths) == 0:
            raise FileNotFoundError(f"No env_*.zarr stores found in: {data_dir}")

        self.groups = [zarr.open_group(p, mode="r") for p in zarr_paths]
        # --- Sanity / dims from first store ---
        g0 = self.groups[0]
        self.img_h, self.img_w = int(g0["rgb"].shape[1]), int(g0["rgb"].shape[2])
        # self.action_dim = int(g0["actions"].shape[1])

        self.pos_slice = slice(0, 3)   # CHANGE if needed
        self.action_dim = 3            # velocity from x,y,z diffs
        # For DPCC compatibility prints
        self.observation_dim = 0
        self.goal_dim = 0

        # --- Build episodes list across all stores: (store_idx, ep_start, ep_end) ---
        # terminals == 1 marks episode end (inclusive index)
        self.episodes = []
        for gi, g in enumerate(self.groups):
            terminals = g["terminals"][:].astype(np.uint8)
            ends = np.where(terminals == 1)[0].tolist()
            start = 0
            for end in ends:
                ep_len = end - start + 1

                # still need at least To+H steps to form one training sample
                if ep_len >= (self.n_obs_steps + self.horizon * self.stride):
                    self.episodes.append((gi, start, end))
                else:
                    print(f"[WARN] Episode gi={gi} start={start} end={end} "
                          f"too short ({ep_len} < {self.n_obs_steps + self.horizon * self.stride}), skipping")
                # # optional truncate to max_path_length
                # if end - start + 1 > self.max_path_length:
                #     end = start + self.max_path_length - 1

                # ep_len = end - start + 1
                # # Need at least To images and H actions
                # if ep_len >= (self.n_obs_steps + self.horizon):
                #     self.episodes.append((gi, start, end))
                start = end + 1
            # handle any trailing data after the last terminal
            if start < len(terminals):
                ep_len = len(terminals) - start
                if ep_len >= (self.n_obs_steps + self.horizon * self.stride):
                    self.episodes.append((gi, start, len(terminals) - 1))
                    print(f"[WARN] Trailing data after last terminal: gi={gi} "
                          f"start={start} end={len(terminals)-1}, included")
        if len(self.episodes) == 0:
            raise RuntimeError(
                f"No episodes long enough for To+H*stride = {self.n_obs_steps}+{self.horizon}*{self.stride}. "
                "Collect longer episodes or lower To/H/stride."
            )

        # --- Build indices (like SequenceDataset.make_indices) ---
        # Each index points to (store_idx, t_start) where:
        #   obs window: [t_start-To+1 : t_start+1]
        #   action window: raw frames [t_start : t_start+H*stride+1 : stride]
        self.indices = []
        for (gi, ep_start, ep_end) in self.episodes:
            t_min = ep_start + (self.n_obs_steps - 1)
            t_max = ep_end - self.horizon * self.stride  # t_start + H*stride <= ep_end
            for t_start in range(t_min, t_max + 1):
                self.indices.append((gi, t_start))

        if len(self.indices) == 0:
            raise RuntimeError("Indices empty after episode parsing. Check To/H and terminals.")

        # --- Build action normalizer (LimitsNormalizer) over all actions in all stores ---
        # Compute global min/max without loading everything at once
        a_mins = None
        a_maxs = None
        for gi, g in enumerate (self.groups):
            s = g["states"]  # (T, state_dim)
            pos_all = s[:, self.pos_slice]
            T = s.shape[0]
            print(f"\n[DIAG] group gi={gi}, states shape={s.shape}")
            
            for (ep_gi, ep_start, ep_end) in self.episodes:
                if ep_gi != gi:
                    continue
                
                p   = pos_all[ep_start : ep_end + 1]   # fully within real episode
                vel = p[self.stride:] - p[:-self.stride] if self.stride > 1 else p[1:] - p[:-1]

                if vel.shape[0] == 0:
                    continue

                cmin = vel.min(axis=0)
                cmax = vel.max(axis=0)

                a_mins = cmin if a_mins is None else np.minimum(a_mins, cmin)
                a_maxs = cmax if a_maxs is None else np.maximum(a_maxs, cmax)

        X = np.stack([a_mins, a_maxs], axis=0).astype(np.float32)
        self.action_normalizer = LimitsNormalizer(X)

        print(f"[ZarrCrazyflieImageDataset] Loaded {len(self.groups)} zarr stores from: {data_dir}")
        print(f"[ZarrCrazyflieImageDataset] Episodes: {len(self.episodes)} | Indices: {len(self.indices)}")
        print(f"[ZarrCrazyflieImageDataset] Image: {self.img_h}x{self.img_w} | action_dim={self.action_dim}")
        print(f"[ZarrCrazyflieImageDataset] To={self.n_obs_steps} | H={self.horizon} | stride={self.stride} "
              f"| collection_dt={self.dt} | control_dt={self.control_dt} (eval sim dt must match this)")
        print(f"[ZarrCrazyflieImageDataset] Action min (should be small negatives): {a_mins}")
        print(f"[ZarrCrazyflieImageDataset] Action max (should be small positives): {a_maxs}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        gi, t_start = self.indices[idx]
        g = self.groups[gi]

        To = self.n_obs_steps
        H = self.horizon

        # --- condition: image window ending at t_start ---
        # rgb in zarr: (T, H, W, 3) uint8
        rgb = g["rgb"][t_start - To + 1 : t_start + 1]  # (To, H, W, 3) uint8
        # convert to float32 in [0,1], and channel-first for PyTorch
        rgb = (rgb.astype(np.float32) / 255.0).transpose(0, 3, 1, 2)  # (To, 3, H, W)

        # --- target: action chunk starting at t_start ---
        # actions = g["actions"][t_start : t_start + H].astype(np.float32)  # (H, action_dim)
        # actions = self.action_normalizer.normalize(actions)              # -> [-1, 1]
        
        # Need H+1 strided states (stride raw frames apart) to compute H velocity steps
        states = g["states"][t_start : t_start + H * self.stride + 1 : self.stride].astype(np.float32)  # (H+1, state_dim)
        pos = states[:, self.pos_slice]                                     # (H+1, 3)
        actions = pos[1:] - pos[:-1]                                        # (H, 3), each spanning control_dt

        actions = self.action_normalizer.normalize(actions.astype(np.float32))
        conditions = {"obs_rgb": rgb}

        if self.include_returns:
            returns = np.array([0.0 / self.returns_scale], dtype=np.float32)
            return RewardBatch(actions, conditions, returns)

        return Batch(actions, conditions)
