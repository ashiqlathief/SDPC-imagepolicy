import os
import glob
import numpy as np

# --- choose ONE DCT backend ---
USE_SCIPY = True

if USE_SCIPY:
    from scipy.fftpack import dct  # usually available in scientific python envs
else:
    import cv2  # if you prefer OpenCV (must be installed)

def dct2_scipy(img2d: np.ndarray) -> np.ndarray:
    # 2D DCT (type-II), orthonormal
    return dct(dct(img2d, axis=0, norm="ortho"), axis=1, norm="ortho")

def dct2_cv2(img2d: np.ndarray) -> np.ndarray:
    # OpenCV expects float32
    import cv2
    return cv2.dct(img2d.astype(np.float32))

def to_gray_resize(rgb: np.ndarray, out_hw=(64, 64)) -> np.ndarray:
    """
    rgb: (H,W,3) uint8
    returns: (out_h,out_w) float32 in [0,1]
    """
    # grayscale (luma)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    gray = (0.2989 * r + 0.5870 * g + 0.1140 * b).astype(np.float32) / 255.0

    out_h, out_w = out_hw
    H, W = gray.shape

    # simple resize without extra deps: nearest-neighbor sampling
    ys = (np.linspace(0, H - 1, out_h)).astype(np.int32)
    xs = (np.linspace(0, W - 1, out_w)).astype(np.int32)
    gray_small = gray[ys][:, xs]  # (out_h, out_w)

    return gray_small

def image_to_dct_latent(rgb: np.ndarray, out_hw=(64, 64), k=8) -> np.ndarray:
    """
    returns latent vector of length k*k (float32)
    """
    gray = to_gray_resize(rgb, out_hw=out_hw)
    if USE_SCIPY:
        coeff = dct2_scipy(gray)
    else:
        coeff = dct2_cv2(gray)
    # keep low-frequency block
    low = coeff[:k, :k].reshape(-1)
    return low.astype(np.float32)

def convert_npz(npz_path: str, k=8, out_hw=(64, 64), out_suffix="_latent.npy"):
    data = np.load(npz_path)
    rgb = data["rgb"]  # expect (T,H,W,3) uint8
    if rgb.dtype != np.uint8:
        # if stored as float 0..1, convert
        rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)

    T = rgb.shape[0]
    z_dim = k * k
    latents = np.empty((T, z_dim), dtype=np.float16)

    for t in range(T):
        z = image_to_dct_latent(rgb[t], out_hw=out_hw, k=k)
        latents[t] = z.astype(np.float16)

    out_path = npz_path.replace(".npz", out_suffix)
    np.save(out_path, latents)
    print(f"✅ {os.path.basename(npz_path)} -> {os.path.basename(out_path)}  shape={latents.shape} dtype={latents.dtype}")

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="folder containing *_images.npz files")
    ap.add_argument("--k", type=int, default=8, help="low-freq block size (k x k). 8 -> 64 dims")
    ap.add_argument("--h", type=int, default=64, help="resize height before DCT")
    ap.add_argument("--w", type=int, default=64, help="resize width before DCT")
    args = ap.parse_args()

    npz_files = sorted(glob.glob(os.path.join(args.data_dir, "*_images.npz")))
    print(f"Found {len(npz_files)} image files in {args.data_dir}")

    for f in npz_files:
        convert_npz(f, k=args.k, out_hw=(args.h, args.w))

if __name__ == "__main__":
    main()
