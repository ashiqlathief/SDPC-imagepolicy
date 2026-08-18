from collections import namedtuple
import importlib
import os
import glob
import numpy as np
import torch
import zarr

from diffuser.utils.path import project_path
from .normalization import LimitsNormalizer

# DEPTH_NEAR/DEPTH_FAR default from the single source of truth in
# config/avoiding-crazyflie.py (importlib since that filename has a hyphen),
# so the clip range used here matches what quadcopter.py clamps non-finite
# depth pixels to at collection time.
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
        stride=1,                   # frames-per-action-step at collection rate (see `dt`)
        dt=0.005,                   # sim dt the raw zarr frames were collected at
        use_depth=False,            # load and concat the depth channel collected via quadcopter.py --use_depth
        depth_near=None,            # metres; defaults to config/avoiding-crazyflie.py DEPTH_NEAR
        depth_far=None,             # metres; defaults to config/avoiding-crazyflie.py DEPTH_FAR
        action_mode="xyz",          # "xyz" (default: world-frame dx,dy,dz) or
                                     # "xz_yaw" (dx=body-frame forward distance, dz=world
                                     # altitude delta, dyaw=wrapped heading change; needs
                                     # quaternion in states, e.g. data collected via
                                     # quadcopter_px4.py -- see data_subdir="data_px4")
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
        self.use_depth = bool(use_depth)
        self.in_chans = 4 if self.use_depth else 3
        self.depth_near = float(depth_near) if depth_near is not None else _DEFAULT_DEPTH_NEAR
        self.depth_far = float(depth_far) if depth_far is not None else _DEFAULT_DEPTH_FAR
        if action_mode not in ("xyz", "xz_yaw"):
            raise ValueError(f"action_mode must be 'xyz' or 'xz_yaw', got {action_mode!r}")
        self.action_mode = action_mode
        # The wall-clock time one predicted action step spans. eval's sim dt
        # must equal this for a0_real to mean what the model thinks it means.
        self.control_dt = self.stride * self.dt

        if preprocess_fns is None:
            preprocess_fns = []

        # --- Find Zarr stores ---
        data_dir = project_path("isaac", "dataset", "avoiding_crazyflie", data_subdir, zarr_subdir)
        if not os.path.isdir(data_dir):
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
        # self.action_dim = int(g0["actions"].shape[1])

        self.pos_slice = slice(0, 3)   # CHANGE if needed
        self.action_dim = 3            # (dx,dy,dz) or (dx_body,dz,dyaw), see action_mode
        if self.action_mode == "xz_yaw":
            state_dim = g0["states"].shape[1]
            if state_dim < 7:
                raise ValueError(
                    f"action_mode='xz_yaw' needs quaternion in states (dims 3:7), but "
                    f"states has only {state_dim} dims. Collect with quadcopter_px4.py "
                    "(13-dim state: pos, quat_xyzw, linvel, angvel) via data_subdir='data_px4'."
                )
            self.quat_slice = slice(3, 7)   # (x,y,z,w), matches quadcopter_px4.py's state layout
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
            quat_all = s[:, self.quat_slice] if self.action_mode == "xz_yaw" else None
            T = s.shape[0]
            print(f"\n[DIAG] group gi={gi}, states shape={s.shape}")

            for (ep_gi, ep_start, ep_end) in self.episodes:
                if ep_gi != gi:
                    continue

                p = pos_all[ep_start : ep_end + 1]                        # fully within real episode
                q = quat_all[ep_start : ep_end + 1] if quat_all is not None else None
                vel = self._actions_from_deltas(p, q, self.stride)        # (>=0, action_dim)

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
        print(f"[ZarrCrazyflieImageDataset] Image: {self.img_h}x{self.img_w} | action_mode={self.action_mode} "
              f"| action_dim={self.action_dim}")
        print(f"[ZarrCrazyflieImageDataset] To={self.n_obs_steps} | H={self.horizon} | stride={self.stride} "
              f"| collection_dt={self.dt} | control_dt={self.control_dt} (eval sim dt must match this)")
        print(f"[ZarrCrazyflieImageDataset] Action min (should be small negatives): {a_mins}")
        print(f"[ZarrCrazyflieImageDataset] Action max (should be small positives): {a_maxs}")

    @staticmethod
    def _yaw_from_quat_xyzw(quat_xyzw):
        """quat_xyzw: (..., 4) in (x,y,z,w) -- matches quadcopter_px4.py's stored state
        layout. Returns yaw (...,) in radians."""
        x, y, z, w = quat_xyzw[..., 0], quat_xyzw[..., 1], quat_xyzw[..., 2], quat_xyzw[..., 3]
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return np.arctan2(siny_cosp, cosy_cosp)

    @staticmethod
    def _wrap_to_pi(angle):
        return (angle + np.pi) % (2.0 * np.pi) - np.pi

    def _actions_from_deltas(self, pos, quat_xyzw, gap):
        """pos: (L,3) world xyz. quat_xyzw: (L,4) or None (only needed for 'xz_yaw').
        gap: frame separation each action spans -- `self.stride` when `pos`/`quat_xyzw`
        are dense per-episode arrays (normalizer fitting), or 1 when they're already
        pre-strided (__getitem__). Returns (L-gap, action_dim), each row spanning
        control_dt seconds of sim time.
        """
        if self.action_mode == "xyz":
            return pos[gap:] - pos[:-gap]

        # "xz_yaw": dx = body-frame forward distance (world xy delta projected onto
        # the heading at the START of the step), dz = world altitude delta, dyaw =
        # wrapped heading change. See feedback thread: only meaningful once the
        # demonstration policy actually turns to steer; with a fixed heading (as in
        # today's collectors) dyaw~=0 and dx reduces to plain world-frame x-delta.
        yaw = self._yaw_from_quat_xyzw(quat_xyzw)          # (L,)
        heading = yaw[:-gap]
        delta_xy = pos[gap:, :2] - pos[:-gap, :2]           # (L-gap, 2)
        dx = delta_xy[:, 0] * np.cos(heading) + delta_xy[:, 1] * np.sin(heading)
        dz = pos[gap:, 2] - pos[:-gap, 2]
        dyaw = self._wrap_to_pi(yaw[gap:] - yaw[:-gap])
        return np.stack([dx, dz, dyaw], axis=-1)            # (L-gap, 3)

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
        # actions = g["actions"][t_start : t_start + H].astype(np.float32)  # (H, action_dim)
        # actions = self.action_normalizer.normalize(actions)              # -> [-1, 1]

        # Need H+1 strided states (stride raw frames apart) to compute H velocity steps
        states = g["states"][t_start : t_start + H * self.stride + 1 : self.stride].astype(np.float32)  # (H+1, state_dim)
        pos = states[:, self.pos_slice]                                     # (H+1, 3)
        quat = states[:, self.quat_slice] if self.action_mode == "xz_yaw" else None
        actions = self._actions_from_deltas(pos, quat, 1)                   # (H, action_dim), each spanning control_dt

        actions = self.action_normalizer.normalize(actions.astype(np.float32))
        conditions = {"obs_rgb": rgb}

        if self.include_returns:
            returns = np.array([0.0 / self.returns_scale], dtype=np.float32)
            return RewardBatch(actions, conditions, returns)

        return Batch(actions, conditions)
