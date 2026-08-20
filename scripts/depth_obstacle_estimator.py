from __future__ import annotations
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors


# =============================================================================
# Camera geometry
# =============================================================================

def camera_intrinsics(width: int, height: int, focal_length: float, horizontal_aperture: float):
    """Pinhole intrinsics from IsaacLab's CameraCfg.PinholeCameraCfg fields.
    Isaac derives vertical_aperture from the aspect ratio (vertical_aperture =
    horizontal_aperture * height/width), so fx == fy here."""
    fx = fy = focal_length * width / horizontal_aperture
    cx, cy = width / 2.0, height / 2.0
    return fx, fy, cx, cy


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product, (w,x,y,z) convention (matches root_state_w's layout
    used throughout this repo, e.g. px4_style_controller.py)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dtype=np.float32)


def quat_apply(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector(s) v (...,3) by quaternion q (4,), (w,x,y,z)."""
    w, x, y, z = q
    qv = np.array([x, y, z], dtype=np.float32)
    uv = np.cross(qv, v)
    uuv = np.cross(qv, uv)
    return v + 2.0 * (w * uv + uuv)


# Fixed body->camera offset, copied from CrazyflieSceneCfg.FPV_CAMERA_CFG in
# isaac/scripts/crazyflie_env_cfg.py (pos=(0,0,0.02), rot=(0.5,-0.5,0.5,-0.5),
# convention="ros") -- kept as a constant here rather than imported so this
# module has no IsaacLab dependency. If that offset ever changes, update it
# here too.
CAM_OFFSET_POS = np.array([0.0, 0.0, 0.02], dtype=np.float32)
CAM_OFFSET_QUAT = np.array([0.5, -0.5, 0.5, -0.5], dtype=np.float32)  # (w,x,y,z)

# FPV_CAMERA_CFG's sensor params (height/width/focal_length/horizontal_aperture),
# also copied rather than imported for the same reason.
FPV_WIDTH, FPV_HEIGHT = 96, 96
FPV_FOCAL_LENGTH = 24.0
FPV_HORIZONTAL_APERTURE = 20.955


def camera_world_pose(pos_body_w: np.ndarray, quat_body_w: np.ndarray):
    """(3,), (4,) drone root pose (world) -> (3,), (4,) FPV camera pose (world)."""
    pos_cam_w = pos_body_w + quat_apply(quat_body_w, CAM_OFFSET_POS)
    quat_cam_w = quat_mul(quat_body_w, CAM_OFFSET_QUAT)
    return pos_cam_w, quat_cam_w


def crop_points_by_world_z(points_cam: np.ndarray, pos_cam_w: np.ndarray,
                            quat_cam_w: np.ndarray, z_lo: float, z_hi: float) -> np.ndarray:
    """points_cam: (N,3) points already in CAMERA frame (e.g. backprojected
    with an identity pose, as depth_camera_live_test.py does for its
    handheld-camera sources). Drops any point whose reconstructed WORLD-
    frame z falls outside [z_lo, z_hi], returning the survivors still in
    camera frame -- this only removes points, it never changes convention.

    For a caller that already has WORLD-frame points (e.g.
    eval_crazieflie1.py, which backprojects with the real pos_cam_w/
    quat_cam_w up front), just crop by z_bounds directly -- this function
    exists specifically for the camera-frame case, where a per-point world
    z has to be reconstructed first before it can be compared to a known
    floor/ceiling height. Meaningful only when the caller actually knows
    the scene's floor/ceiling geometry (e.g. CEILING_HEIGHT in
    crazyflie_env_cfg.py); a real handheld camera has no such knowledge,
    so this is opt-in rather than baked into filter_points()."""
    if len(points_cam) == 0:
        return points_cam
    world_z = pos_cam_w[2] + quat_apply(quat_cam_w, points_cam)[:, 2]
    return points_cam[(world_z >= z_lo) & (world_z <= z_hi)]


# =============================================================================
# Depth -> world point cloud
# =============================================================================

def backproject_depth_to_world(depth: np.ndarray, fx: float, fy: float, cx: float, cy: float,
                                pos_cam_w: np.ndarray, quat_cam_w: np.ndarray,
                                max_range: float, stride: int = 2) -> np.ndarray:
    """depth: (H,W) or (H,W,1) float32 metres, RADIAL distance from the camera
    origin (Isaac's "distance_to_camera" annotator -- not planar z-depth;
    env.get_depth() in crazyflie_env.py returns (H,W,1), keeping the trailing
    channel dim, so squeeze it here rather than assume callers do it). Returns
    (N,3) world-frame points, dropping no-hit / out-of-range pixels.

    `stride` subsamples the pixel grid before back-projecting (e.g. stride=2
    on a 96x96 image -> ~2.3k candidate points instead of ~9.2k), since the
    downstream voxel filter would collapse most of them anyway.
    """
    if depth.ndim == 3:
        depth = depth[..., 0]
    H, W = depth.shape
    us = np.arange(0, W, stride)
    vs = np.arange(0, H, stride)
    uu, vv = np.meshgrid(us, vs)
    d = depth[vv, uu]

    valid = np.isfinite(d) & (d > 1e-3) & (d < max_range)
    if not valid.any():
        return np.zeros((0, 3), dtype=np.float32)

    uu, vv, d = uu[valid], vv[valid], d[valid]
    x = (uu - cx) / fx
    y = (vv - cy) / fy
    z = np.ones_like(x, dtype=np.float32)
    ray = np.stack([x, y, z], axis=-1).astype(np.float32)
    ray = ray / np.linalg.norm(ray, axis=-1, keepdims=True)   # unit ray, camera/ROS frame
    pts_cam = ray * d[:, None]                                 # radial distance along ray

    pts_world = pos_cam_w[None, :] + quat_apply(quat_cam_w, pts_cam)
    return pts_world.astype(np.float32)


# =============================================================================
# Filtering (depth.md step 2, adapted -- known-geometry crop instead of RANSAC)
# =============================================================================

def filter_points(points: np.ndarray, x_bounds, y_bounds, z_bounds,
                   voxel_size: float = 0.05, outlier_radius: float = 0.15,
                   outlier_min_neighbors: int = 4, wall_pad: float = 0.15,
                   z_margin: float = 0.05, output_2d: bool = False) -> np.ndarray:
    """Crop obviously-invalid points (sensor noise/reflections far outside the
    flight envelope), voxel-downsample, then drop points with too few
    neighbors (isolated depth noise / "flying pixels" near occlusion edges).

    x/y ("wall") bounds are widened OUTWARD by `wall_pad` rather than cropped
    inward: x_bounds/y_bounds are the projector's tightened flight-envelope
    limits (e.g. y_bounds=(-0.95,0.95)), not the real wall surface, which
    physically sits ~0.05-0.10m further out (see CORRIDOR_WIDTH/WALL_THICKNESS
    in crazyflie_env_cfg.py) -- cropping AT x_bounds/y_bounds would discard
    every real wall-surface point before it ever reached clustering, meaning
    the walls could never be detected as obstacles. Widening outward instead
    lets real wall points through so they get clustered/tracked/enforced the
    same as any other detected surface (cylinders included).

    z ("floor/ceiling") stays cropped INWARD by z_margin, i.e. still excluded
    -- the drone's flight-envelope z_halfspaces already fence the top/bottom,
    and floor points are typically the noisiest source of false obstacles for
    a low-flying drone, so they're left out unless requested otherwise.

    output_2d: if True, drop the Z column AFTER all of the above 3D-based
    filtering (crop, voxel-downsample, outlier removal) and return (x, y)
    only. Motivation: real corridor obstacles (BOXES/CYLINDERS) are vertical
    columns spanning the full flight envelope height, so their points can
    easily be MORE than `eps` apart in Z alone once you get more than
    ~10-15cm off the floor -- 3D DBSCAN (see cluster_points) then chops one
    physical column into several separate clusters stacked at different
    heights, each getting its own track ID (this is very likely why you see
    hundreds of short-lived, high-numbered track IDs for what's really a
    handful of objects). Clustering in (x, y) only merges every height-slice
    of the same column back into one cluster. Safe to enable even where Z
    mattered downstream: tracks_to_constraints() and ObstacleTracker never
    read a point's Z value in the first place (the projector's own point
    constraints are ("sphere_outside", [0, 1], ...) -- horizontal-plane
    only), so nothing downstream actually loses information it was using.
    cluster_points() and ObstacleTracker are dimension-agnostic and need no
    changes to accept (N,2) input.
    """
    if len(points) == 0:
        return points

    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    keep = (
        (x > x_bounds[0] - wall_pad) & (x < x_bounds[1] + wall_pad) &
        (y > y_bounds[0] - wall_pad) & (y < y_bounds[1] + wall_pad) &
        (z > z_bounds[0] + z_margin) & (z < z_bounds[1] - z_margin)
    )
    points = points[keep]
    if len(points) == 0:
        return points[:, :2] if output_2d else points

    voxel_idx = np.round(points / voxel_size).astype(np.int64)
    _, unique_idx = np.unique(voxel_idx, axis=0, return_index=True)
    points = points[unique_idx]
    if len(points) < outlier_min_neighbors + 1:
        return points[:, :2].astype(np.float32) if output_2d else points

    nn = NearestNeighbors(radius=outlier_radius).fit(points)
    neighbor_lists = nn.radius_neighbors(points, return_distance=False)
    neighbor_counts = np.array([len(idxs) - 1 for idxs in neighbor_lists])  # exclude self
    points = points[neighbor_counts >= outlier_min_neighbors]

    if output_2d and len(points):
        points = points[:, :2].astype(np.float32)
    return points


# =============================================================================
# Clustering (depth.md step 3/4 -- DBSCAN, no obstacle count needed up front)
# =============================================================================

def cluster_points(points: np.ndarray, eps: float = 0.12, min_samples: int = 4,
                    xy_only: bool = False) -> list[np.ndarray]:
    """Returns a list of (N_i,3) point arrays, one per cluster (DBSCAN noise
    points, label -1, are dropped).

    xy_only: run DBSCAN's distance computation on (x, y) only, ignoring z --
    but each returned cluster still keeps every column `points` had (x,y,z),
    just grouped differently. Real corridor obstacles (BOXES/CYLINDERS) are
    vertical columns spanning the full flight envelope height, so two
    points on the same physical column can easily be MORE than `eps` apart
    in z alone once you're more than ~10-15cm off the floor -- 3D DBSCAN
    then chops one column into several clusters stacked at different
    heights, each becoming its own track (see filter_points' docstring for
    the same issue from the crop side, and tracks_to_constraints(), which
    then turns every one of those height-slices' points into its own
    near-duplicate (x,y) keep-out circle -- multiplying constraint count
    for what's really one object). Off by default since it's a real
    tradeoff, not a strict improvement: a genuinely wide horizontal object
    close to a narrow tall one at similar (x,y) would merge into one
    cluster here where 3D clustering would have kept them separate."""
    if len(points) < min_samples:
        return []
    cluster_input = points[:, :2] if xy_only else points
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit(cluster_input).labels_
    return [points[labels == lbl] for lbl in sorted(set(labels)) if lbl != -1]


# =============================================================================
# Frame-to-frame tracking: static/dynamic classification + velocity + memory
# =============================================================================

class ObstacleTracker:
    """Simplified from Chen & Lu's full 6-feature-vector cluster matching to
    centroid-distance nearest-neighbor matching (your cylinders don't change
    point count/color/volume the way their handheld test objects did, so the
    extra feature dimensions aren't needed here). Still implements their core
    idea: match clusters frame-to-frame, classify a track dynamic once its
    smoothed velocity exceeds a threshold for several consecutive frames, and
    keep static tracks alive (the "memory buffer") longer than dynamic ones
    after they stop being matched (e.g. left the narrow FOV).
    """

    def __init__(self, match_dist: float = 0.25, dynamic_speed_thresh: float = 0.05,
                 dynamic_confirm_frames: int = 3, max_missed_static: int = 700,
                 max_missed_dynamic: int = 10):
        self.match_dist = match_dist
        self.dynamic_speed_thresh = dynamic_speed_thresh
        self.dynamic_confirm_frames = dynamic_confirm_frames
        self.max_missed_static = max_missed_static
        self.max_missed_dynamic = max_missed_dynamic
        self.tracks: dict[int, dict] = {}
        self._next_id = 0

    def reset(self):
        self.tracks.clear()
        self._next_id = 0

    def update(self, clusters: list[np.ndarray], dt: float) -> list[tuple[int, dict]]:
        """clusters: point arrays seen this frame. dt: real elapsed time (sim
        seconds) since the previous call -- used for the velocity finite
        difference, NOT the projector's horizon step spacing.

        Returns [(track_id, track_dict), ...] for every currently-alive track,
        each with keys: centroid (3,), points (Ni,3) (current if matched this
        frame, else the last-known points -- the "memory" for static ones),
        vel (3,) world m/s, is_dynamic (bool).
        """
        centroids = [c.mean(axis=0) for c in clusters]
        unmatched = set(range(len(clusters)))

        for tid, tr in self.tracks.items():
            best_j, best_d = None, self.match_dist
            for j in unmatched:
                d = float(np.linalg.norm(centroids[j] - tr["centroid"]))
                if d < best_d:
                    best_j, best_d = j, d
            if best_j is not None:
                new_centroid = centroids[best_j]
                inst_vel = (new_centroid - tr["centroid"]) / max(dt, 1e-6)
                tr["vel"] = inst_vel if tr["seen"] == 0 else 0.5 * tr["vel"] + 0.5 * inst_vel
                tr["centroid"] = new_centroid
                tr["points"] = clusters[best_j]
                tr["missed"] = 0
                tr["seen"] += 1
                tr["dynamic_streak"] = (tr["dynamic_streak"] + 1
                                         if np.linalg.norm(tr["vel"]) > self.dynamic_speed_thresh else 0)
                tr["is_dynamic"] = tr["dynamic_streak"] >= self.dynamic_confirm_frames
                unmatched.discard(best_j)
            else:
                tr["missed"] += 1

        for j in unmatched:
            self.tracks[self._next_id] = dict(
                centroid=centroids[j], points=clusters[j],
                vel=np.zeros(3, dtype=np.float32), missed=0, seen=0,
                dynamic_streak=0, is_dynamic=False,
            )
            self._next_id += 1

        alive = []
        for tid in list(self.tracks.keys()):
            tr = self.tracks[tid]
            limit = self.max_missed_dynamic if tr["is_dynamic"] else self.max_missed_static
            if tr["missed"] > limit:
                del self.tracks[tid]
                continue
            alive.append((tid, tr))
        return alive


# =============================================================================
# Tracks -> projector inputs (point-based obstacle constraints)
# =============================================================================

def tracks_to_constraints(active_tracks: list[tuple[int, dict]], horizon: int, proj_dt: float,
                           max_points_per_obstacle: int = 12, rng=None):
    """Turns tracker output into the same-shaped data
    build_obstacle_constraint_list() already consumes for cylinders:
      static_points: [(x, y), ...] -- each becomes one fixed-radius keep-out sphere
      dynamic_predictions: {key: [(x, y), ...]} -- one linear-extrapolated
        horizon prediction per dynamic point, same shape
        env.predict_cylinder_positions() already returns, just keyed per-point
        instead of per-cylinder-index.
    """
    rng = rng or np.random.default_rng(0)
    static_points: list[tuple[float, float]] = []
    dynamic_predictions: dict[int, list[tuple[float, float]]] = {}
    key = 0
    for _tid, tr in active_tracks:
        pts = tr["points"]
        if len(pts) > max_points_per_obstacle:
            idx = rng.choice(len(pts), size=max_points_per_obstacle, replace=False)
            pts = pts[idx]
        if not tr["is_dynamic"]:
            static_points.extend((float(p[0]), float(p[1])) for p in pts)
        else:
            vx, vy = float(tr["vel"][0]), float(tr["vel"][1])
            for p in pts:
                dynamic_predictions[key] = [
                    (float(p[0] + vx * k * proj_dt), float(p[1] + vy * k * proj_dt))
                    for k in range(1, horizon + 1)
                ]
                key += 1
    return static_points, dynamic_predictions