"""Synthetic (no IsaacLab, no GPU) test of detect_obstacles_umap()'s size/radius math.

Builds a fake depth image with ONE cylinder of known real-world radius at a known
depth against a background, feeds it through the exact production functions from
depth_obstacle_estimator.py, and compares the reported half_width_m against ground
truth. Isolates whether the bug is in the near/far depth-bin bookkeeping in
_contour_to_camera_detection() vs. background bleeding into the same contour in
_umap_contours(), without needing to boot Isaac Sim (avoids the GPU-contention
hazard noted for the sim-based diagnostic).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from depth_obstacle_estimator import _umap_contours, _contour_to_camera_detection, camera_intrinsics

W = H = 96
fx, fy, cx, cy = camera_intrinsics(96, 96, 24.0, 20.955)
MAX_RANGE_M = 3.0
BIN_SIZE = 200
MIN_PIXEL_COUNT = 8

TRUE_R = 0.15   # true cylinder radius, matches CylinderCfg(radius=0.15) in env_cfg.py
TRUE_D = 1.5    # true depth to the cylinder, metres

width_px = 2.0 * TRUE_R * fx / TRUE_D
u_l_true = int(round(cx - width_px / 2))
u_r_true = int(round(cx + width_px / 2))
print(f"[SETUP] fx={fx:.2f} mm_per_bin={MAX_RANGE_M*1000/BIN_SIZE:.2f}mm "
      f"true width_px={width_px:.1f} true cols=[{u_l_true},{u_r_true})")


def make_depth(bg_depth_m):
    """bg_depth_m: background depth (metres). Use > MAX_RANGE_M to make it
    disappear from the U-map entirely (by design, see _umap_contours' docstring
    comment); use < MAX_RANGE_M to simulate background bleeding into range."""
    depth = np.full((H, W), bg_depth_m, dtype=np.float32)
    depth[:, u_l_true:u_r_true] = TRUE_D   # full-height pole, exact same depth every row
    return depth


def run(name, bg_depth_m):
    depth_m = make_depth(bg_depth_m)
    contours, areas, depth_mm, mm_per_bin, Wc = _umap_contours(
        depth_m, fx, BIN_SIZE, MAX_RANGE_M, t_poi=500.0, t_tho=1800.0, min_pixel_count=MIN_PIXEL_COUNT
    )
    print(f"\n=== {name} (bg_depth={bg_depth_m}m) ===")
    print(f"  {len(contours)} contour(s), areas={areas}")
    if not contours:
        print("  no detection")
        return
    order = np.argsort(areas)[::-1]
    for rank, i in enumerate(order):
        import cv2
        x, y, w, h = cv2.boundingRect(contours[int(i)])
        det = _contour_to_camera_detection(contours[int(i)], depth_mm, fx, fy, cx, cy, mm_per_bin, Wc)
        print(f"  contour[{rank}] bbox cols=[{x},{x+w}) depth-bins=[{y},{y+h}) "
              f"-> depth-range=[{y*mm_per_bin/1000:.2f},{(y+h)*mm_per_bin/1000:.2f}]m")
        if det is None:
            print("    -> None (rejected)")
            continue
        pos_cam, half_w, half_h = det
        print(f"    -> reported depth={pos_cam[2]:.3f}m half_width={half_w:.3f}m "
              f"(true R={TRUE_R}, true depth={TRUE_D}) ratio={half_w/TRUE_R:.2f}x")


run("clean (background out of range)", bg_depth_m=5.0)
run("background bleed (background in range, near max)", bg_depth_m=2.9)
run("background bleed (background mid-range)", bg_depth_m=2.0)


def make_depth_short_pole(bg_depth_m, wall_depth_m, row_lo, row_hi, jitter=0.003):
    """More realistic: a SHORT pole (rows row_lo:row_hi only, not full height --
    matches CYLINDERS' height=1.0m vs WALL_HEIGHT=3.0m in env_cfg.py) with a
    far wall visible above/below it in the SAME columns, at wall_depth_m (in
    range). `jitter` adds small per-pixel noise so nothing sits exactly on a
    bin edge (avoids the strict '>'/'<' boundary artifact seen above)."""
    rng = np.random.default_rng(0)
    depth = bg_depth_m + rng.uniform(-jitter, jitter, (H, W)).astype(np.float32)
    depth[row_lo:row_hi, u_l_true:u_r_true] = TRUE_D + rng.uniform(-jitter, jitter, (row_hi - row_lo, u_r_true - u_l_true))
    depth[:row_lo, u_l_true:u_r_true] = wall_depth_m + rng.uniform(-jitter, jitter, (row_lo, u_r_true - u_l_true))
    depth[row_hi:, u_l_true:u_r_true] = wall_depth_m + rng.uniform(-jitter, jitter, (H - row_hi, u_r_true - u_l_true))
    return depth.astype(np.float32)


def run_short_pole(name, bg_depth_m, wall_depth_m, row_lo, row_hi):
    depth_m = make_depth_short_pole(bg_depth_m, wall_depth_m, row_lo, row_hi)
    contours, areas, depth_mm, mm_per_bin, Wc = _umap_contours(
        depth_m, fx, BIN_SIZE, MAX_RANGE_M, t_poi=500.0, t_tho=1800.0, min_pixel_count=MIN_PIXEL_COUNT
    )
    print(f"\n=== {name} (bg={bg_depth_m}m wall={wall_depth_m}m pole_rows=[{row_lo},{row_hi})) ===")
    print(f"  {len(contours)} contour(s), areas={areas}")
    order = np.argsort(areas)[::-1] if contours else []
    for rank, i in enumerate(order):
        import cv2
        x, y, w, h = cv2.boundingRect(contours[int(i)])
        det = _contour_to_camera_detection(contours[int(i)], depth_mm, fx, fy, cx, cy, mm_per_bin, Wc)
        print(f"  contour[{rank}] bbox cols=[{x},{x+w}) depth-bins=[{y},{y+h}) "
              f"-> depth-range=[{y*mm_per_bin/1000:.2f},{(y+h)*mm_per_bin/1000:.2f}]m")
        if det is None:
            print("    -> None (rejected)")
            continue
        pos_cam, half_w, half_h = det
        print(f"    -> reported depth={pos_cam[2]:.3f}m half_width={half_w:.3f}m half_height={half_h:.3f}m "
              f"(true R={TRUE_R}, true depth={TRUE_D}) width_ratio={half_w/TRUE_R:.2f}x")


# cylinder occupies rows 40:60 of a 96-row frame (a "short pole" vs the corridor's
# full 3.0m wall height), with the far wall visible above/below it, in range.
run_short_pole("short pole, background out of range", bg_depth_m=5.0, wall_depth_m=5.0, row_lo=40, row_hi=60)
run_short_pole("short pole, wall in range behind it", bg_depth_m=5.0, wall_depth_m=2.8, row_lo=40, row_hi=60)
