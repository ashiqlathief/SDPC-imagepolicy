from collections import namedtuple
import importlib
import os
import glob
import numpy as np
import torch
import zarr

from diffuser.utils.path import project_path
from .normalization import LimitsNormalizer
_scene_cfg_module = importlib.import_module("config.avoiding-crazyflie")
_DEFAULT_DEPTH_NEAR = _scene_cfg_module.DEPTH_NEAR
_DEFAULT_DEPTH_FAR = _scene_cfg_module.DEPTH_FAR

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
            "obs_rgb": (To, 3, H_img, W_img) float32 in [0,1]   if use_depth=False
            "obs_rgb": (To, 4, H_img, W_img) float32 in [0,1]   if use_depth=True (4th chan = normalized depth)
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
        use_depth=False,            # load and concat the depth channel collected via quadcopter.py --use_depth
        depth_near=None,            # metres; defaults to config/avoiding-crazyflie.py DEPTH_NEAR
        depth_far=None,             # metres; defaults to config/avoiding-crazyflie.py DEPTH_FAR
        stats_path=None,            # path to the state_*.pt checkpoint, which embeds
                                     # action_min/action_max (see Trainer.save/save_best).
                                     # Used as a fallback ONLY when the Zarr data_dir isn't
                                     # found, so eval can run on a machine without the raw
                                     # dataset. Also accepts a legacy normalizer_stats.npz
                                     # sidecar for runs trained before stats were embedded.
        **_legacy_kwargs,           # swallow stride/dt/action_mode etc. from a
                                     # dataset_config.pkl pickled before those params were
                                     # removed -- Config.__call__ forwards every key it has
                                     # saved, so old checkpoints must still be accepted here.
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
        self.use_depth = bool(use_depth)
        self.in_chans = 4 if self.use_depth else 3
        self.depth_near = float(depth_near) if depth_near is not None else _DEFAULT_DEPTH_NEAR
        self.depth_far = float(depth_far) if depth_far is not None else _DEFAULT_DEPTH_FAR

        if preprocess_fns is None:
            preprocess_fns = []

        # --- Find Zarr stores ---
        data_dir = project_path("isaac", "dataset", "avoiding_crazyflie", data_subdir, zarr_subdir)
        if not os.path.isdir(data_dir):
            # No raw dataset on this machine (e.g. eval-only box): rebuild just the action
            # normalizer, skip everything else. Prefer stats embedded in the checkpoint
            # itself; fall back to a legacy normalizer_stats.npz sidecar for runs trained
            # before stats were embedded (see Trainer.save/save_best).
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

        if self.use_depth:
            for p, g in zip(zarr_paths, self.groups):
                if not bool(g.attrs.get("use_depth", False)) or "depth" not in g:
                    raise ValueError(
                        f"use_depth=True but zarr store has no depth data: {p}. "
                        "Re-collect with `quadcopter.py --use_depth`, or set use_depth=False."
                    )

        # --- Sanity / dims from first store ---
        g0 = self.groups[0]
        self.img_h, self.img_w = int(g0["rgb"].shape[1]), int(g0["rgb"].shape[2])

        self.pos_slice = slice(0, 3)   # CHANGE if needed
        self.action_dim = 3            # (dx,dy,dz)
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

        # --- Build indices (like SequenceDataset.make_indices) ---
        # Each index points to (store_idx, t_start) where:
        #   obs window: [t_start-To+1 : t_start+1]
        #   action window: raw frames [t_start : t_start+H+1]
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
            s = g["states"]  # (T, state_dim)
            pos_all = s[:, self.pos_slice]

            for (ep_gi, ep_start, ep_end) in self.episodes:
                if ep_gi != gi:
                    continue

                p = pos_all[ep_start : ep_end + 1]                        # fully within real episode
                vel = p[1:] - p[:-1]                                      # (>=0, action_dim)

                if vel.shape[0] == 0:
                    continue

                cmin = vel.min(axis=0)
                cmax = vel.max(axis=0)

                a_mins = cmin if a_mins is None else np.minimum(a_mins, cmin)
                a_maxs = cmax if a_maxs is None else np.maximum(a_maxs, cmax)

        X = np.stack([a_mins, a_maxs], axis=0).astype(np.float32)
        self.action_normalizer = LimitsNormalizer(X)

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

        if self.use_depth:
            depth = g["depth"][t_start - To + 1 : t_start + 1]  # (To, H, W) float32 metres
            # quadcopter.py clamps inf/nan to DEPTH_FAR at collection time, but np.clip alone
            # would NOT fix nan (np.clip(nan, lo, hi) == nan), so guard here too in case older
            # data was collected before that fix.
            non_finite = ~np.isfinite(depth)
            if non_finite.any():
                depth = depth.copy()
                depth[non_finite] = self.depth_far
            depth = np.clip(depth, self.depth_near, self.depth_far)
            depth = (depth - self.depth_near) / (self.depth_far - self.depth_near)  # -> [0,1]
            depth = depth[:, None, :, :].astype(np.float32)  # (To, 1, H, W)
            assert np.isfinite(depth).all(), "non-finite values in normalized depth after clipping"
            rgb = np.concatenate([rgb, depth], axis=1)  # (To, 4, H, W)

        # --- target: action chunk starting at t_start ---
        # Need H+1 consecutive states to compute H velocity steps
        states = g["states"][t_start : t_start + H + 1].astype(np.float32)  # (H+1, state_dim)
        pos = states[:, self.pos_slice]                                     # (H+1, 3)
        actions = pos[1:] - pos[:-1]                                        # (H, action_dim), each spanning one frame

        actions = self.action_normalizer.normalize(actions.astype(np.float32))
        conditions = {"obs_rgb": rgb}

        if self.include_returns:
            returns = np.array([0.0 / self.returns_scale], dtype=np.float32)
            return RewardBatch(actions, conditions, returns)

        return Batch(actions, conditions)
