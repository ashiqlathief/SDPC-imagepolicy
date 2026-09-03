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
_DEFAULT_DEPTH_NEAR = _scene_cfg_module.DEPTH_NEAR
_DEFAULT_DEPTH_FAR = _scene_cfg_module.DEPTH_FAR

# Keep the same “shape” of outputs as other datasets
Batch = namedtuple("Batch", "trajectories conditions")
RewardBatch = namedtuple("RewardBatch", "trajectories conditions returns")


def _quat_conjugate(q):
    """q: (..., 4) xyzw -> conjugate (inverse, for unit quaternions)."""
    q = q.copy()
    q[..., :3] *= -1.0
    return q


def _quat_rotate(q, v):
    """Rotate vector(s) v (..., 3) by quaternion(s) q (..., 4) xyzw:
    v' = q * v * q^-1, expanded without building a full rotation matrix.
    Used by action_mode="vxz_yawrate" to convert states' world-frame
    linvel/angvel into the body frame the RL controller's own
    [v_x, v_z, yaw_rate] action is expressed in (see execise_01_c.py's
    _pre_physics_step: lin_vel_b/ang_vel_b get quat_apply'd from body ->
    world before being written to sim, so this is the inverse of that step)."""
    q_xyz = q[..., :3]
    q_w = q[..., 3:4]
    t = 2.0 * np.cross(q_xyz, v)
    return v + q_w * t + np.cross(q_xyz, t)


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
        data_subdir="data1",         # isaac/dataset/avoiding_crazyflie/<data_subdir>/<zarr_subdir>
        use_depth=False,            # load and concat the depth channel collected via quadcopter.py --use_depth
        depth_near=None,            # metres; defaults to config/avoiding-crazyflie.py DEPTH_NEAR
        depth_far=None,             # metres; defaults to config/avoiding-crazyflie.py DEPTH_FAR
        stats_path=None,
        use_pose_cond=False,
        action_mode="xyz",
        **_legacy_kwargs,
    ):
        super().__init__()
        if _legacy_kwargs:
            print(f"[CrazyflieImageDataset] Ignoring legacy kwargs from an older "
                  f"dataset_config.pkl: {sorted(_legacy_kwargs)}")
        if action_mode not in ("xyz", "vxz_yawrate"):
            raise ValueError(f"action_mode must be 'xyz' or 'vxz_yawrate', got {action_mode!r}")
        self.action_mode = action_mode
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
        self.use_pose_cond = bool(use_pose_cond)
        self.pose_normalizer = None

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
                if self.use_pose_cond and stats_path is not None and os.path.isfile(stats_path):
                    ckpt = torch.load(stats_path, map_location="cpu")
                    if "pose_min" in ckpt and "pose_max" in ckpt:
                        pX = np.stack(
                            [np.asarray(ckpt["pose_min"]), np.asarray(ckpt["pose_max"])], axis=0
                        ).astype(np.float32)
                        self.pose_normalizer = LimitsNormalizer(pX)
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

        if self.use_pose_cond:
            for p, g in zip(zarr_paths, self.groups):
                if not bool(g.attrs.get("has_targets", False)) or "targets" not in g:
                    raise ValueError(
                        f"use_pose_cond=True but zarr store has no target data: {p}. "
                        "Re-collect with a target-recording run, or set use_pose_cond=False."
                    )

        # --- Sanity / dims from first store ---
        g0 = self.groups[0]
        raw_h, raw_w = int(g0["rgb"].shape[1]), int(g0["rgb"].shape[2])
        # Model was trained on config.avoiding-crazyflie's VIT_IMG_SIZE (224x224,
        # square, matching vit_small_patch8_224's own pretraining resolution).
        # Real-camera zarr stores (240x424) are a different, non-square
        # resolution -- __getitem__ center-crops to square then resizes to
        # this, so img_h/img_w below reflect what's actually served, not the
        # raw zarr resolution. No-op if the zarr is already this size.
        self.img_size = int(getattr(_scene_cfg_module, "VIT_IMG_SIZE", 96))
        self.img_h = self.img_w = self.img_size
        self._raw_img_h, self._raw_img_w = raw_h, raw_w

        self.pos_slice = slice(0, 3)   # CHANGE if needed
        self.action_dim = 3            # (dx,dy,dz) if action_mode=="xyz", else
                                        # (vx_body,vz_body,yaw_rate_body)
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
            s = g["states"]  # (T, state_dim): [pos(3), quat_xyzw(4), linvel_w(3), angvel_w(3)]
            pos_all = s[:, self.pos_slice]
            if self.action_mode == "vxz_yawrate":
                quat_all = s[:, 3:7]
                linvel_all = s[:, 7:10]
                angvel_all = s[:, 10:13]

            for (ep_gi, ep_start, ep_end) in self.episodes:
                if ep_gi != gi:
                    continue

                if self.action_mode == "xyz":
                    p = pos_all[ep_start : ep_end + 1]                    # fully within real episode
                    act = p[1:] - p[:-1]                                  # (>=0, action_dim)
                else:
                    q = quat_all[ep_start : ep_end + 1]
                    lv_b = _quat_rotate(_quat_conjugate(q), linvel_all[ep_start : ep_end + 1])
                    av_b = _quat_rotate(_quat_conjugate(q), angvel_all[ep_start : ep_end + 1])
                    act = np.stack([lv_b[:, 0], lv_b[:, 2], av_b[:, 2]], axis=-1)  # (>=0, 3)

                if act.shape[0] == 0:
                    continue

                cmin = act.min(axis=0)
                cmax = act.max(axis=0)

                a_mins = cmin if a_mins is None else np.minimum(a_mins, cmin)
                a_maxs = cmax if a_maxs is None else np.maximum(a_maxs, cmax)

        X = np.stack([a_mins, a_maxs], axis=0).astype(np.float32)
        self.action_normalizer = LimitsNormalizer(X)

        # --- Build pose normalizer (LimitsNormalizer) over current + target positions ---
        # Shared scale for both "pose_now" (drone position) and "pose_target" (episode goal),
        # since both live in the same world-frame coordinates.
        if self.use_pose_cond:
            p_mins = None
            p_maxs = None
            for gi, g in enumerate(self.groups):
                pos_all = g["states"][:, self.pos_slice]
                targets_all = g["targets"][:]

                for (ep_gi, ep_start, ep_end) in self.episodes:
                    if ep_gi != gi:
                        continue

                    pos_ep = pos_all[ep_start : ep_end + 1]
                    tgt_ep = targets_all[ep_start : ep_end + 1]
                    combined = np.concatenate([pos_ep, tgt_ep], axis=0)

                    cmin = combined.min(axis=0)
                    cmax = combined.max(axis=0)

                    p_mins = cmin if p_mins is None else np.minimum(p_mins, cmin)
                    p_maxs = cmax if p_maxs is None else np.maximum(p_maxs, cmax)

            pX = np.stack([p_mins, p_maxs], axis=0).astype(np.float32)
            self.pose_normalizer = LimitsNormalizer(pX)

    def __len__(self):
        return len(self.indices)

    def _center_crop_resize(self, frames, interpolation):
        """Center-crop to square (using the shorter side) then resize to
        self.img_size. Matches the eval-time preprocessing the real camera
        feed goes through (see scripts/eval_crazieflieros2.py), so a
        non-square native camera resolution (e.g. 240x424) still lands on
        the same square VIT_IMG_SIZE the model's vision encoder expects.
        No-op if frames are already img_size x img_size."""
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

        # --- condition: image window ending at t_start ---
        # rgb in zarr: (T, H, W, 3) uint8
        rgb = g["rgb"][t_start - To + 1 : t_start + 1]  # (To, H, W, 3) uint8
        rgb = self._center_crop_resize(rgb, cv2.INTER_AREA)  # (To, img_size, img_size, 3) uint8
        # convert to float32 in [0,1], and channel-first for PyTorch
        rgb = (rgb.astype(np.float32) / 255.0).transpose(0, 3, 1, 2)  # (To, 3, img_size, img_size)

        if self.use_depth:
            depth = g["depth"][t_start - To + 1 : t_start + 1]  # (To, H, W) float32 metres
            depth = self._center_crop_resize(depth, cv2.INTER_NEAREST)  # nearest: don't blend across depth edges
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
        # Need H+1 consecutive states; states[0] is only used below for pose_now.
        states = g["states"][t_start : t_start + H + 1].astype(np.float32)  # (H+1, state_dim)
        if self.action_mode == "xyz":
            pos = states[:, self.pos_slice]                                 # (H+1, 3)
            actions = pos[1:] - pos[:-1]                                    # (H, action_dim), each spanning one frame
        else:
            quat = states[1:, 3:7]           # (H, 4) xyzw
            lin_vel_w = states[1:, 7:10]     # (H, 3) world-frame
            ang_vel_w = states[1:, 10:13]    # (H, 3) world-frame
            lin_vel_b = _quat_rotate(_quat_conjugate(quat), lin_vel_w)   # (H, 3) body-frame
            ang_vel_b = _quat_rotate(_quat_conjugate(quat), ang_vel_w)   # (H, 3) body-frame
            # [vx_body, vz_body, yaw_rate_body] -- vy_body is omitted: the collection
            # controller never commands lateral velocity, so it's ~0 throughout.
            actions = np.stack([lin_vel_b[:, 0], lin_vel_b[:, 2], ang_vel_b[:, 2]], axis=-1)  # (H, 3)

        actions = self.action_normalizer.normalize(actions.astype(np.float32))
        conditions = {"obs_rgb": rgb}

        if self.use_pose_cond:
            pose_now = states[0, self.pos_slice]                  # (3,) position at t_start
            pose_target = g["targets"][t_start].astype(np.float32)  # (3,) this episode's goal
            conditions["pose_now"] = self.pose_normalizer.normalize(pose_now)
            conditions["pose_target"] = self.pose_normalizer.normalize(pose_target)

        if self.include_returns:
            returns = np.array([0.0 / self.returns_scale], dtype=np.float32)
            return RewardBatch(actions, conditions, returns)

        return Batch(actions, conditions)
