"""
plot_frames.py — save each frame of a saved frames_*.npy array (from
scripts/eval_crazieflie1.py --save_frames, shape (T,H,W,3) uint8) as an
individual PNG, and/or one contact-sheet overview PNG.

Usage:
    python scripts/plot_frames.py path/to/frames_*.npy
    python scripts/plot_frames.py frames.npy --out frames_png/ --stride 5
    python scripts/plot_frames.py frames.npy --grid           # + one contact sheet
    python scripts/plot_frames.py frames.npy --grid-only       # contact sheet only
"""
import argparse
import math
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_individual(frames, out_dir, stride):
    os.makedirs(out_dir, exist_ok=True)
    idxs = range(0, len(frames), stride)
    for i in idxs:
        plt.imsave(os.path.join(out_dir, f"frame_{i:04d}.png"), frames[i])
    print(f"[INFO] Saved {len(list(idxs))} frames -> {out_dir}")


def save_grid(frames, out_path, stride, ncols):
    idxs = list(range(0, len(frames), stride))
    nrows = math.ceil(len(idxs) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.3, nrows * 1.3))
    axes = np.atleast_1d(axes).reshape(-1)
    for ax, i in zip(axes, idxs):
        ax.imshow(frames[i])
        ax.set_title(str(i), fontsize=6)
        ax.axis("off")
    for ax in axes[len(idxs):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Saved contact sheet ({len(idxs)} frames) -> {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npy_path", help="path to a frames_*.npy file, shape (T,H,W,3)")
    ap.add_argument("--out", default=None,
                     help="output dir for individual frames (default: <npy_dir>/<npy_stem>_png/)")
    ap.add_argument("--stride", type=int, default=1, help="save every Nth frame (default: 1, all frames)")
    ap.add_argument("--grid", action="store_true", help="also save one contact-sheet PNG")
    ap.add_argument("--grid-only", action="store_true", help="save only the contact sheet, skip individual PNGs")
    ap.add_argument("--ncols", type=int, default=8, help="columns in the contact sheet (default: 8)")
    args = ap.parse_args()

    frames = np.load(args.npy_path)
    assert frames.ndim == 4 and frames.shape[-1] == 3, f"expected (T,H,W,3), got {frames.shape}"
    print(f"[INFO] Loaded {frames.shape[0]} frames of size {frames.shape[1]}x{frames.shape[2]} from {args.npy_path}")

    stem = os.path.splitext(os.path.basename(args.npy_path))[0]
    out_dir = args.out or os.path.join(os.path.dirname(args.npy_path), f"{stem}_png")

    if not args.grid_only:
        save_individual(frames, out_dir, args.stride)
    if args.grid or args.grid_only:
        grid_path = os.path.join(os.path.dirname(args.npy_path), f"{stem}_grid.png")
        save_grid(frames, grid_path, args.stride, args.ncols)


if __name__ == "__main__":
    main()
