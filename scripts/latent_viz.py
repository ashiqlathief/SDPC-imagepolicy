import argparse
import os
import re
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from torch.utils.data import DataLoader, Subset

# sklearn for PCA and t-SNE
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report

import diffuser.utils as utils


# ─────────────────────────────────────────────────────────────────────────────
# Extract latent vectors from dataset
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_latents(encoder, dataset, n_samples, device, batch_size=64):
    """
    Pass n_samples images through the encoder and collect:
      - latent vectors  (n_samples, latent_dim)
      - x positions     (n_samples,)  — corridor progress label for coloring
    """
    # randomly subsample dataset indices
    total = len(dataset)
    n_samples = min(n_samples, total)
    indices = np.random.choice(total, n_samples, replace=False)
    subset  = Subset(dataset, indices)
    loader  = DataLoader(subset, batch_size=batch_size, shuffle=False,
                         num_workers=0, pin_memory=False)

    encoder.eval()
    all_latents = []
    all_x_pos   = []

    print(f"[latent_viz] Extracting {n_samples} latent vectors...")

    for batch in loader:
        # batch is a namedtuple: Batch(trajectories, conditions)
        # conditions["obs_rgb"]: (B, To, 3, H, W)
        # trajectories: (B, H, action_dim) — actions as delta-pos
        trajectories = batch.trajectories   # (B, H, action_dim)
        obs_rgb      = batch.conditions["obs_rgb"].to(device)  # (B, To, 3, H, W)

        # encode
        latent = encoder(obs_rgb)           # (B, latent_dim)
        all_latents.append(latent.cpu().numpy())

        # use cumulative x displacement as a proxy for corridor position
        # actions are delta-pos in real units (after unnormalize) but
        # here they are normalized — use sum of x actions as proxy
        x_displacement = trajectories[:, :, 0].sum(dim=1).numpy()  # (B,)
        all_x_pos.append(x_displacement)

        print(f"  processed {sum(len(l) for l in all_latents)}/{n_samples}", end="\r")

    print()

    latents = np.concatenate(all_latents, axis=0)  # (N, D)
    x_pos   = np.concatenate(all_x_pos,   axis=0)  # (N,)

    print(f"[latent_viz] Done. Latent shape: {latents.shape}")
    return latents, x_pos

@torch.no_grad()
def extract_latents_episode(encoder, dataset, episode_idx, device):
    """
    Extract latent vectors for one full episode in temporal order.
    Returns:
        latents:    (T, latent_dim)
        timesteps:  (T,)  step index 0,1,2,...T-1
        x_actions:  (T,)  cumulative x displacement (corridor progress)
    """
    # find all indices belonging to episode_idx
    episode_indices = [
        i for i, (gi, t_start) in enumerate(dataset.indices)
        if gi == 0  # single zarr store
    ]

    # get the episode boundaries
    gi_ep, ep_start, ep_end = dataset.episodes[episode_idx]

    # get all dataset indices that fall within this episode
    ep_indices = [
        i for i, (gi, t_start) in enumerate(dataset.indices)
        if gi == gi_ep and ep_start <= t_start <= ep_end
    ]

    print(f"[latent_viz] Episode {episode_idx}: "
          f"start={ep_start} end={ep_end} frames={len(ep_indices)}")

    latents    = []
    x_actions  = []
    timesteps  = []

    for step, idx in enumerate(ep_indices):
        batch = dataset[idx]
        obs_rgb = torch.tensor(
            batch.conditions["obs_rgb"]
        ).unsqueeze(0).to(device)  # (1, To, 3, H, W)

        latent = encoder(obs_rgb)   # (1, latent_dim)
        latents.append(latent.cpu().numpy()[0])

        # cumulative x displacement as corridor progress
        x_actions.append(float(batch.trajectories[:, 0].sum()))
        timesteps.append(step)

    latents   = np.stack(latents,   axis=0)
    x_actions = np.array(x_actions, dtype=np.float32)
    timesteps = np.array(timesteps, dtype=np.float32)

    print(f"[latent_viz] Extracted {len(latents)} frames")
    return latents, timesteps, x_actions

# ─────────────────────────────────────────────────────────────────────────────
# PCA explained variance plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_explained_variance(latents, save_path, encoder_type, latent_dim):
    """
    Shows how many PCA components are needed to explain the variance.
    Useful for justifying your chosen latent dimension.
    """
    scaler  = StandardScaler()
    scaled  = scaler.fit_transform(latents)

    pca_full = PCA()
    pca_full.fit(scaled)

    cumvar = np.cumsum(pca_full.explained_variance_ratio_) * 100
    dims   = np.arange(1, len(cumvar) + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(dims, cumvar, linewidth=2)
    ax.axhline(90, linestyle="--", color="gray", alpha=0.7, label="90% variance")
    ax.axhline(95, linestyle=":",  color="gray", alpha=0.7, label="95% variance")

    # mark where 90% and 95% variance is reached
    idx_90 = int(np.searchsorted(cumvar, 90))
    idx_95 = int(np.searchsorted(cumvar, 95))
    ax.axvline(idx_90, linestyle="--", color="tab:orange", alpha=0.5,
               label=f"90% at {idx_90} dims")
    ax.axvline(idx_95, linestyle=":",  color="tab:red",    alpha=0.5,
               label=f"95% at {idx_95} dims")

    ax.set_xlabel("Number of PCA components")
    ax.set_ylabel("Cumulative explained variance (%)")
    ax.set_title(f"PCA explained variance — {encoder_type}  latent={latent_dim}")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, min(latent_dim, 100))   # show first 100 dims max
    ax.set_ylim(0, 101)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"[latent_viz] Saved: {save_path}")
    print(f"  90% variance at {idx_90} components")
    print(f"  95% variance at {idx_95} components")


# ─────────────────────────────────────────────────────────────────────────────
# 2D scatter plot helper
# ─────────────────────────────────────────────────────────────────────────────

def scatter_2d(coords_2d, color_vals, title, xlabel, ylabel,
               save_path, cmap="viridis", clabel="x displacement"):
    fig, ax = plt.subplots(figsize=(8, 7))
    sc = ax.scatter(
        coords_2d[:, 0], coords_2d[:, 1],
        c=color_vals, cmap=cmap,
        s=8, alpha=0.6, linewidths=0,
    )
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(clabel, fontsize=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"[latent_viz] Saved: {save_path}")

# ─────────────────────────────────────────────────────────────────────────────
# LDA helpers
# ─────────────────────────────────────────────────────────────────────────────

# Class names and colors used across all LDA plots
REGION_NAMES  = ["Early corridor", "Mid corridor", "Near gate"]
REGION_COLORS = ["tab:blue", "tab:orange", "tab:green"]


def make_region_labels(x_pos: np.ndarray,
                       early_thresh: float = -0.5,
                       late_thresh:  float =  0.5) -> np.ndarray:
    """
    Convert x_pos (cumulative normalized x action, corridor progress proxy)
    into 3 discrete region labels:

        0 = "early"      x_pos <  early_thresh
        1 = "mid"        early_thresh <= x_pos <= late_thresh
        2 = "near gate"  x_pos >  late_thresh

    Thresholds are on the NORMALIZED action scale (LimitsNormalizer → [-1, 1]).
    Adjust via --lda_early_thresh / --lda_late_thresh if your distribution
    is heavily skewed toward one region.

    Why 3 classes?
        LDA produces at most (n_classes - 1) = 2 discriminant components,
        which is exactly what we need for a 2D scatter plot.
    """
    labels = np.ones(len(x_pos), dtype=np.int64)       # default: mid (1)
    labels[x_pos <  early_thresh] = 0                  # early
    labels[x_pos >  late_thresh]  = 2                  # near gate

    counts = np.bincount(labels, minlength=3)
    print(f"[LDA] Region label counts — "
          f"early={counts[0]}  mid={counts[1]}  near_gate={counts[2]}")
    return labels


def scatter_2d_classes(coords_2d, class_labels, class_names, class_colors,
                       title, xlabel, ylabel, save_path,
                       show_centroids=True):
    """
    2D scatter plot where each point is colored by its discrete class label.
    Draws a legend and optionally a star marker at each class centroid.

    Parameters
    ----------
    coords_2d     : (N, 2) projected coordinates  (e.g. from LDA)
    class_labels  : (N,)   integer labels 0, 1, 2, ...
    class_names   : list of str, one name per class
    class_colors  : list of color strings, one per class
    show_centroids: if True, draw a large star at each class centroid
    """
    fig, ax = plt.subplots(figsize=(8, 7))

    for cls_id, (name, color) in enumerate(zip(class_names, class_colors)):
        mask = class_labels == cls_id
        ax.scatter(
            coords_2d[mask, 0], coords_2d[mask, 1],
            c=color, label=name,
            s=10, alpha=0.55, linewidths=0,
        )
        if show_centroids and mask.sum() > 0:
            cx = coords_2d[mask, 0].mean()
            cy = coords_2d[mask, 1].mean()
            ax.scatter(cx, cy,
                       c=color, s=200, marker="*",
                       edgecolors="black", linewidths=0.8, zorder=5)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=10, markerscale=2)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"[latent_viz] Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# LDA main function
# ─────────────────────────────────────────────────────────────────────────────
 
def run_lda(latents_scaled, x_pos, out_dir, encoder_type, latent_dim,
            early_thresh, late_thresh):
    """
    Fit LDA on the scaled latents using corridor-region labels,
    project to 2D, and save two plots:
 
        lda_region_2d.pdf        — scatter of LDA components
        lda_separability.pdf     — between-class vs within-class variance bar chart
 
    Also prints a classification report so you know how linearly
    separable the three regions actually are in latent space.
    LDA finds the 2D projection that MAXIMALLY SEPARATES the three
    corridor regions (early / mid / near-gate). If the three clusters
    are well separated, your encoder has learned position-aware features.
    If they overlap completely, the encoder ignores corridor position —
    which would explain poor navigation.
 
    Compare CNN vs ViT side by side: whichever encoder gives cleaner
    cluster separation has a more task-relevant latent space.
    """
 
    # ── build region labels ───────────────────────────────────────────────────
    region_labels = make_region_labels(x_pos, early_thresh, late_thresh)
 
    # skip LDA if any class is empty (can't fit)
    counts = np.bincount(region_labels, minlength=3)
    if (counts == 0).any():
        print("[LDA] WARNING: one or more classes are empty — "
              "adjust --lda_early_thresh / --lda_late_thresh. Skipping LDA.")
        return
 
    # ── fit LDA ───────────────────────────────────────────────────────────────
    # n_components = n_classes - 1 = 2  →  perfect for 2D visualization
    lda = LinearDiscriminantAnalysis(n_components=2)
    coords_lda = lda.fit_transform(latents_scaled, region_labels)
 
    print(f"\n[LDA] Explained variance ratio (LD1, LD2): "
          f"{lda.explained_variance_ratio_}")
 
    # ── classification report ─────────────────────────────────────────────────
    # Predict class from LDA projection to measure linear separability
    preds = lda.predict(latents_scaled)
    print("\n[LDA] Linear separability report (on training data):")
    print(classification_report(
        region_labels, preds,
        target_names=REGION_NAMES,
        digits=3,
    ))

    scatter = compute_within_class_scatter(latents_scaled, region_labels)
    print("\n[LDA] Within-class scatter (lower = tighter clusters):")
    print(f"  {'Class':<18} {'Mean dist':>10} {'Std dist':>10} {'N points':>10}")
    print(f"  {'-'*52}")
    for name, vals in scatter.items():
        print(f"  {name:<18} {vals['mean_dist']:>10.3f} "
              f"{vals['std_dist']:>10.3f} {vals['n']:>10d}")

    # Fisher criterion per class pair
    print(f"\n[LDA] Fisher criterion per class pair (higher = better):")
    centroids = {}
    for cls_id, name in enumerate(REGION_NAMES):
        mask = region_labels == cls_id
        centroids[name] = latents_scaled[mask].mean(axis=0)

    pairs = [
        ("Early corridor", "Mid corridor"),
        ("Mid corridor",   "Near gate"),
        ("Early corridor", "Near gate"),
    ]
    for nameA, nameB in pairs:
        between = np.linalg.norm(centroids[nameA] - centroids[nameB])
        sA = scatter[nameA]["std_dist"]
        sB = scatter[nameB]["std_dist"]
        fisher = between / (sA + sB + 1e-8)
        print(f"  {nameA} vs {nameB}")
        print(f"    centroid_dist={between:.3f}  fisher={fisher:.3f}")
    # ── scatter plot ──────────────────────────────────────────────────────────
    ld1_var = lda.explained_variance_ratio_[0] * 100
    ld2_var = lda.explained_variance_ratio_[1] * 100
 
    scatter_2d_classes(
        coords_2d    = coords_lda,
        class_labels = region_labels,
        class_names  = REGION_NAMES,
        class_colors = REGION_COLORS,
        title        = (f"LDA latent space — {encoder_type}  latent={latent_dim}\n"
                        f"LD1={ld1_var:.1f}%  LD2={ld2_var:.1f}%  "
                        f"(★ = class centroid)"),
        xlabel       = f"LD1 ({ld1_var:.1f}% between-class variance)",
        ylabel       = f"LD2 ({ld2_var:.1f}% between-class variance)",
        save_path    = os.path.join(out_dir, "lda_region_2d.pdf"),
        show_centroids = True,
    )
 
    # ── separability bar chart ────────────────────────────────────────────────
    # Shows how much of the total variance is "between-class" (good)
    # vs "within-class" (bad = classes overlap)
    _plot_lda_separability(
        lda          = lda,
        latents      = latents_scaled,
        labels       = region_labels,
        encoder_type = encoder_type,
        latent_dim   = latent_dim,
        save_path    = os.path.join(out_dir, "lda_separability.pdf"),
    )

def compute_within_class_scatter(latents_scaled, region_labels):
    """
    For each class, compute the mean and std distance of points
    from their class centroid. Lower = tighter, more compact cluster.
    """
    results = {}
    for cls_id, name in enumerate(REGION_NAMES):
        mask     = region_labels == cls_id
        pts      = latents_scaled[mask]
        centroid = pts.mean(axis=0)
        dists    = np.linalg.norm(pts - centroid, axis=1)
        results[name] = {
            "mean_dist": float(dists.mean()),
            "std_dist":  float(dists.std()),
            "n":         int(mask.sum()),
            "dists":     dists,   # keep raw dists for Fisher computation
        }
    return results


def _plot_lda_separability(lda, latents, labels, encoder_type,
                            latent_dim, save_path):
    """
    Bar chart showing between-class variance (Fisher criterion).
 
    For each LDA component, a higher bar means the classes are better
    separated along that direction.  This is a compact single-number
    quality metric you can compare across CNN and ViT encoders.
    """
    ratios = lda.explained_variance_ratio_ * 100   # LD1, LD2
 
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(
        [f"LD{i+1}" for i in range(len(ratios))],
        ratios,
        color=["tab:blue", "tab:orange"],
        edgecolor="black", linewidth=0.8,
    )
 
    # annotate bar values
    for bar, val in zip(bars, ratios):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=11)
 
    ax.set_ylabel("Between-class variance explained (%)")
    ax.set_title(f"LDA separability — {encoder_type}  latent={latent_dim}")
    ax.set_ylim(0, max(ratios) * 1.2)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"[latent_viz] Saved: {save_path}")
# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir",   type=str, required=True,help="Path to trained run e.g. .../H16_K20_.../9")
    parser.add_argument("--n_samples", type=int, default=2000, help="How many dataset samples to encode")
    parser.add_argument("--device",    type=str, default="cuda:0")
    parser.add_argument("--tsne_perplexity", type=float, default=30.0)
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--episode_idx", type=int, default=0,help="Which episode from dataset to visualize")
    parser.add_argument("--n_episodes", type=int, default=95,help="Number of episodes to visualize")
        # LDA threshold arguments
    # These are on the NORMALIZED action scale (LimitsNormalizer → [-1,1])
    # Default: split into thirds  early<-0.5  mid  late>0.5
    parser.add_argument("--lda_early_thresh", type=float, default=-0.5,help="x_pos below this → 'early corridor' label")
    parser.add_argument("--lda_late_thresh",  type=float, default=0.5,help="x_pos above this → 'near gate' label")
    args = parser.parse_args()
    all_latents = []
    all_timesteps = []
    all_x_pos = []
    all_episode_ids = []

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    # ── load model and dataset ────────────────────────────────────────────────
    print(f"[latent_viz] Loading: {args.run_dir}")
    diff_exp  = utils.load_diffusion(args.run_dir, epoch="best", device=str(device))
    dataset   = diff_exp.dataset
    diffusion = diff_exp.diffusion.to(device)
    diffusion.eval()

    # get encoder from inside the diffusion model
    encoder = diffusion.model.encoder
    encoder.eval()

    # infer encoder type and latent dim from model
    latent_dim   = int(getattr(diffusion.model, "image_cond_dim", 256))
    encoder_type = getattr(diffusion.model, "encoder_type", "unknown")
    print(f"[latent_viz] encoder={encoder_type}  latent_dim={latent_dim}")

    # ── output directory ──────────────────────────────────────────────────────
    out_dir = os.path.join(args.run_dir, "latent_viz")
    os.makedirs(out_dir, exist_ok=True)

    # ── extract latents ───────────────────────────────────────────────────────
    # latents, x_pos = extract_latents(
    #     encoder, dataset, args.n_samples, device, batch_size=64
    # )
    for ep_idx in range(args.n_episodes):
        latents, timesteps, x_pos = extract_latents_episode(
            encoder, dataset, episode_idx=ep_idx, device=device
        )
        all_latents.append(latents)
        all_timesteps.append(timesteps)
        all_x_pos.append(x_pos)
        all_episode_ids.append([ep_idx] * len(latents))

    latents   = np.concatenate(all_latents,    axis=0)
    x_pos     = np.concatenate(all_x_pos,      axis=0)
    episode_ids = np.concatenate(all_episode_ids, axis=0)
    # print(f"Total frames: {len(latents)}")   # should be ~8014
    # print(f"episode_ids shape: {episode_ids.shape}")  # must match
    # ── standardize ──────────────────────────────────────────────────────────
    scaler  = StandardScaler()
    latents_scaled = scaler.fit_transform(latents)

    # # ─────────────────────────────────────────────────────────────────────────
    # # 1. PCA explained variance curve
    # # ─────────────────────────────────────────────────────────────────────────
    # print("\n[latent_viz] Running PCA explained variance...")
    # plot_explained_variance(
    #     latents_scaled,
    #     save_path    = os.path.join(out_dir, "pca_explained_variance.pdf"),
    #     encoder_type = encoder_type,
    #     latent_dim   = latent_dim,
    # )

    # # ─────────────────────────────────────────────────────────────────────────
    # # 2. PCA 2D scatter
    # # ─────────────────────────────────────────────────────────────────────────
    # print("\n[latent_viz] Running PCA 2D...")
    # pca_2d   = PCA(n_components=2)
    # coords_pca = pca_2d.fit_transform(latents_scaled)

    # var_explained = pca_2d.explained_variance_ratio_ * 100
    # print(f"  PC1 explains {var_explained[0]:.1f}%  "
    #       f"PC2 explains {var_explained[1]:.1f}%")

    # scatter_2d(
    #     coords_2d  = coords_pca,
    #     color_vals = episode_ids, 
    #     # color_vals = timesteps,
    #     # color_vals = x_pos,
    #     cmap       = "tab10", 
    #     title      = f"PCA latent space — {encoder_type}  latent={latent_dim}\n"
    #                  f"PC1={var_explained[0]:.1f}%  PC2={var_explained[1]:.1f}%",
    #     xlabel     = f"PC1 ({var_explained[0]:.1f}% variance)",
    #     ylabel     = f"PC2 ({var_explained[1]:.1f}% variance)",
    #     save_path  = os.path.join(out_dir, "pca_2d.pdf"),
    #     # clabel     = "cumulative x action (corridor progress proxy)",
    #     clabel     = "episode index",
    # )

    # # ─────────────────────────────────────────────────────────────────────────
    # # 3. t-SNE 2D scatter
    # # ─────────────────────────────────────────────────────────────────────────
    # print(f"\n[latent_viz] Running t-SNE (perplexity={args.tsne_perplexity})...")
    # print("  This may take a few minutes for large n_samples...")

    # # run t-SNE on PCA-reduced data (50 dims) for speed
    # # standard practice: PCA first to 50 dims, then t-SNE
    # n_pca_for_tsne = min(50, latent_dim)
    # pca_50 = PCA(n_components=n_pca_for_tsne)
    # latents_pca50 = pca_50.fit_transform(latents_scaled)
    # print(f"  PCA pre-reduction: {latent_dim} → {n_pca_for_tsne} dims "
    #       f"({pca_50.explained_variance_ratio_.sum()*100:.1f}% variance kept)")

    # tsne = TSNE(
    #     n_components = 2,
    #     perplexity   = args.tsne_perplexity,
    #     max_iter       = 1000,
    #     random_state = args.seed,
    #     verbose      = 1,
    # )
    # coords_tsne = tsne.fit_transform(latents_pca50)

    # scatter_2d(
    #     coords_2d  = coords_tsne,
    #     color_vals = x_pos,
    #     cmap       = "viridis",
    #     clabel     = "corridor progress",
    #     title      = f"t-SNE latent space — {encoder_type}  latent={latent_dim}  "
    #                  f"perplexity={args.tsne_perplexity}",
    #     xlabel     = "t-SNE dim 1",
    #     ylabel     = "t-SNE dim 2",
    #     save_path  = os.path.join(out_dir, "tsne_2d.pdf"),
    #     # clabel     = "cumulative x action (corridor progress proxy)",
    # )
    # ─────────────────────────────────────────────────────────────────────────
    # 4. LDA 2D scatter
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[latent_viz] Running LDA...")
    print(f"  Region thresholds: early < {args.lda_early_thresh} "
          f"< mid < {args.lda_late_thresh} < near_gate")
    print("  (adjust with --lda_early_thresh and --lda_late_thresh if needed)")
    run_lda(
        latents_scaled = latents_scaled,
        x_pos          = x_pos,
        out_dir        = out_dir,
        encoder_type   = encoder_type,
        latent_dim     = latent_dim,
        early_thresh   = args.lda_early_thresh,
        late_thresh    = args.lda_late_thresh,
    )
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n[latent_viz] All plots saved to: {out_dir}")
    print("Files:")
    for f in os.listdir(out_dir):
        print(f"  {f}")


if __name__ == "__main__":
    main()