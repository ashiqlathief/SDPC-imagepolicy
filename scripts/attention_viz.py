import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path

import diffuser.utils as utils


# ─────────────────────────────────────────────────────────────────────────────
# Frame sampling — pick representative frames spread across corridor progress
# ─────────────────────────────────────────────────────────────────────────────

def sample_representative_frames(dataset, n_frames, seed=42):
    """
    Pick n_frames indices spread evenly across corridor progress
    (early / mid / near-gate) so we get a representative cross-section.

    Returns list of dataset indices.
    """
    rng = np.random.default_rng(seed)
    total = len(dataset)

    # collect x_pos for all samples in a fast pass (no encoder needed)
    # x_pos = sum of x-actions in the trajectory chunk = corridor progress proxy
    x_pos_all = []
    for i in range(total):
        batch  = dataset[i]
        x_disp = float(batch.trajectories[:, 0].sum())
        x_pos_all.append(x_disp)
    x_pos_all = np.array(x_pos_all)

    # split into n_frames equal-width bins and pick one sample per bin
    bins   = np.linspace(x_pos_all.min(), x_pos_all.max(), n_frames + 1)
    chosen = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        candidates = np.where((x_pos_all >= lo) & (x_pos_all < hi))[0]
        if len(candidates) == 0:
            # fallback: nearest sample
            mid = (lo + hi) / 2
            candidates = np.array([np.argmin(np.abs(x_pos_all - mid))])
        chosen.append(int(rng.choice(candidates)))

    print(f"[attention_viz] Sampled {len(chosen)} frames "
          f"(x_pos range: {x_pos_all[chosen].min():.2f} → "
          f"{x_pos_all[chosen].max():.2f})")
    return chosen, x_pos_all[chosen]


def load_frame(dataset, idx, device):
    """
    Load one sample from the dataset.
    Returns:
        obs_rgb_t : (1, To, 3, H, W)  float32 in [0,1]  on device
        raw_img   : (H, W, 3)         uint8              for display
    """
    batch   = dataset[idx]
    obs_rgb = batch.conditions["obs_rgb"]         # (To, 3, H, W) float32

    # raw image for display: take last frame, convert to uint8 HWC
    raw = obs_rgb[-1]                             # (3, H, W) float32 in [0,1]
    raw_img = (raw.transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)

    obs_rgb_t = torch.tensor(obs_rgb).unsqueeze(0).to(device)  # (1,To,3,H,W)
    return obs_rgb_t, raw_img


# ─────────────────────────────────────────────────────────────────────────────
# ViT attention map extraction
# ─────────────────────────────────────────────────────────────────────────────

def _detect_vit_type(encoder):
    """
    Returns 'custom' for ViTObsEncoder (uses encoder.blocks / nn.TransformerEncoder)
    Returns 'timm'   for ViTObsEncoderPretrained (uses encoder.backbone / timm model)
    """
    if hasattr(encoder, "blocks") and hasattr(encoder, "patch"):
        return "custom"
    if hasattr(encoder, "backbone") and hasattr(encoder.backbone, "blocks"):
        return "timm"
    raise ValueError(
        f"Cannot detect ViT type for encoder class '{type(encoder).__name__}'. "
        "Expected either 'blocks'+'patch' (custom ViT) or 'backbone.blocks' (timm ViT)."
    )


def extract_vit_attention(encoder, obs_rgb_t):
    encoder.eval()
    vit_type = _detect_vit_type(encoder)

    B, To, C, H, W = obs_rgb_t.shape
    store = {}

    if vit_type == "custom":
        layers     = encoder.blocks.layers       # ModuleList of TransformerEncoderLayer
        last_layer = layers[-1]
        patch_emb  = encoder.patch
        grid_size  = patch_emb.grid_size
        N          = patch_emb.num_patches

        with torch.no_grad():
            x = obs_rgb_t.float().view(B * To, C, H, W)
            if x.max() > 1.5:
                x = x / 255.0
            x = (x - encoder.img_mean) / encoder.img_std

            tok = encoder.patch(x)               # (BT, N, D)
            BT, _N, D = tok.shape
            cls = encoder.cls_token.expand(BT, -1, -1)
            tok = torch.cat([cls, tok], dim=1)   # (BT, 1+N, D)
            tok = tok + encoder.pos_embed[:, :(1 + _N), :]
            tok = encoder.pos_drop(tok)

            # Run all layers except the last through normal path
            for layer in layers[:-1]:
                tok = layer(tok)

            # Last layer: manually execute norm_first self-attention
            # so we can pass need_weights=True directly — no fast path involved
            normed   = last_layer.norm1(tok)
            attn_out, weights = last_layer.self_attn(
                normed, normed, normed,
                need_weights=True,
                average_attn_weights=False,   # keep per-head weights
            )                                 # weights: (BT, n_heads, 1+N, 1+N)
            tok = tok + last_layer.dropout1(attn_out)
            tok = tok + last_layer._ff_block(last_layer.norm2(tok))

            tok = encoder.norm(tok)           # final LayerNorm

        weights  = weights                        # (BT, n_heads, 1+N, 1+N)
        cls_attn = weights[:, :, 0, 1:]          # CLS→patch: (BT, n_heads, N)
        cls_attn = cls_attn.mean(dim=1)           # avg over heads: (BT, N)
        cls_attn = cls_attn.view(B, To, N).mean(dim=1)  # avg over frames: (B, N)

    else:
        backbone   = encoder.backbone
        last_block = backbone.blocks[-1]
        attn_mod   = last_block.attn

        # grid size from timm patch_embed
        grid_size = int(backbone.patch_embed.grid_size[0])  # square grid
        N         = grid_size * grid_size

        # temporarily disable fused attention so the explicit path runs
        orig_fused = attn_mod.fused_attn
        attn_mod.fused_attn = False

        def attn_drop_hook(module, input, output):
            # input[0] is the softmax attention: (BT, n_heads, 1+N, 1+N)
            store["weights"] = input[0]

        hook_handle = attn_mod.attn_drop.register_forward_hook(attn_drop_hook)

        try:
            with torch.no_grad():
                # Use encoder.backbone directly — it handles normalisation
                # internally. We just need to reshape (B,To,C,H,W)->((B*To),C,H,W).
                x = obs_rgb_t.float().view(B * To, C, H, W)
                if x.max() > 1.5:
                    x = x / 255.0
                # timm models expect normalised input
                mean = torch.tensor([0.485, 0.456, 0.406],
                                     device=x.device).view(1, 3, 1, 1)
                std  = torch.tensor([0.229, 0.224, 0.225],
                                     device=x.device).view(1, 3, 1, 1)
                x = (x - mean) / std
                _ = backbone(x)   # triggers hook
        finally:
            hook_handle.remove()
            attn_mod.fused_attn = orig_fused   # always restore

        if "weights" not in store:
            raise RuntimeError(
                "timm ViT: attention weights were not captured. "
                "fused_attn may still be active or hook did not fire."
            )

        weights  = store["weights"]          # (BT, n_heads, 1+N, 1+N)
        # CLS token is index 0 in timm ViT as well
        cls_attn = weights[:, :, 0, 1:]     # (BT, n_heads, N)
        cls_attn = cls_attn.mean(dim=1)      # (BT, N)
        cls_attn = cls_attn.view(B, To, N).mean(dim=1)  # (B, N)

    # ── shared post-processing ────────────────────────────────────────────────
    attn = cls_attn[0].cpu().float().numpy()     # (N,)
    attn = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)

    attn_grid = attn.reshape(grid_size, grid_size)
    attn_map  = torch.tensor(attn_grid).unsqueeze(0).unsqueeze(0)
    attn_map  = F.interpolate(attn_map, size=(H, W), mode="bilinear",
                               align_corners=False)
    return attn_map.squeeze().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# CNN Grad-CAM extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_gradcam(encoder, obs_rgb_t):
    """
    Compute Grad-CAM heatmap for the CNN encoder targeting layer4 of ResNet18.

    Grad-CAM weights each feature map channel by the gradient of the
    encoder output's L2 norm w.r.t. that channel's activations, then
    takes a ReLU to keep only positively contributing regions.

    Returns:
        cam : (H_img, W_img)  float32  heatmap in [0,1]
    """
    encoder.eval()

    backbone = encoder.backbone        # ResNet18 with fc=Identity
    target_layer = backbone.layer4[-1]  # last residual block: output (B, 512, h, w)

    B, To, C, H, W = obs_rgb_t.shape

    # ── forward pass with gradient tracking ──────────────────────────────────
    activations = {}
    gradients   = {}

    def fwd_hook(module, input, output):
        activations["feat"] = output          # (B*To, 512, h, w)

    def bwd_hook(module, grad_input, grad_output):
        gradients["feat"] = grad_output[0]    # (B*To, 512, h, w)

    fwd_h = target_layer.register_forward_hook(fwd_hook)
    bwd_h = target_layer.register_full_backward_hook(bwd_hook)

    try:
        x = obs_rgb_t.float()
        x = x.view(B * To, C, H, W)
        if x.max() > 1.5:
            x = x / 255.0

        x = x.requires_grad_(True)

        # full encoder forward (backbone only — we want feature gradients)
        feat = backbone(x)                    # (B*To, 512)
        # use L2 norm of output as scalar for backprop
        score = feat.norm(dim=1).sum()
        score.backward()

    finally:
        fwd_h.remove()
        bwd_h.remove()

    # ── compute Grad-CAM ─────────────────────────────────────────────────────
    acts = activations["feat"].detach()       # (B*To, 512, h, w)
    grds = gradients["feat"].detach()         # (B*To, 512, h, w)

    # global-average-pool gradients over spatial dims → channel weights
    weights = grds.mean(dim=(2, 3), keepdim=True)   # (B*To, 512, 1, 1)

    # weighted combination of feature maps
    cam = (weights * acts).sum(dim=1)         # (B*To, h, w)
    cam = F.relu(cam)                         # keep positive contributions

    # average over To frames
    cam = cam.view(B, To, *cam.shape[1:]).mean(dim=1)  # (B, h, w)
    cam = cam[0].cpu().numpy()                # (h, w)

    # normalise to [0,1]
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    # upsample to original image size
    cam_t = torch.tensor(cam).unsqueeze(0).unsqueeze(0)   # (1,1,h,w)
    cam_t = F.interpolate(cam_t, size=(H, W), mode="bilinear",
                           align_corners=False)
    cam   = cam_t.squeeze().numpy()                       # (H, W)

    return cam


# ─────────────────────────────────────────────────────────────────────────────
# Overlay heatmap on image
# ─────────────────────────────────────────────────────────────────────────────

def overlay_heatmap(raw_img, heatmap, alpha=0.5, colormap="jet"):
    cmap    = cm.get_cmap(colormap)
    colored = (cmap(heatmap)[:, :, :3] * 255).astype(np.uint8)  # (H,W,3)
    blended = (alpha * colored + (1 - alpha) * raw_img).astype(np.uint8)
    return blended


# ─────────────────────────────────────────────────────────────────────────────
# Save one frame result
# ─────────────────────────────────────────────────────────────────────────────

def save_frame_result(raw_img, heatmap, frame_idx, x_pos_val,
                      out_dir, method_name, encoder_type):
    """
    Save:
      1. Side-by-side figure: original | heatmap-only | overlay
      2. The blended overlay as a standalone PNG
    """
    overlay = overlay_heatmap(raw_img, heatmap)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(raw_img)
    axes[0].set_title("Original frame", fontsize=11)
    axes[0].axis("off")

    im = axes[1].imshow(heatmap, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title(f"{method_name} heatmap", fontsize=11)
    axes[1].axis("off")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay (α=0.5)", fontsize=11)
    axes[2].axis("off")

    # fig.suptitle(
    #     f"{encoder_type} — frame {frame_idx:02d} — "
    #     f"x_pos={x_pos_val:.3f}  ({method_name})",
    #     fontsize=12
    # )
    fig.tight_layout()

    fname = os.path.join(out_dir, f"frame_{frame_idx:02d}_{method_name.lower()}.png")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[attention_viz] Saved: {fname}")

    return overlay


# ─────────────────────────────────────────────────────────────────────────────
# Summary grid
# ─────────────────────────────────────────────────────────────────────────────

def save_summary_grid(overlays, x_pos_vals, out_dir,
                      encoder_type, method_name):
    """
    Save all frame overlays in a single grid figure.
    Useful for putting directly into the thesis.
    """
    n = len(overlays)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(ncols * 3.5, nrows * 3.5))
    axes = np.array(axes).reshape(-1)

    for i, (overlay, xp) in enumerate(zip(overlays, x_pos_vals)):
        axes[i].imshow(overlay)
        # axes[i].set_title(f"x_pos={xp:.2f}", fontsize=9)
        axes[i].axis("off")

    # hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    # fig.suptitle(
    #     f"{encoder_type} — {method_name} — {n} representative frames\n"
    #     f"(ordered by corridor progress, left→right, top→bottom)",
    #     fontsize=12
    # )
    fig.tight_layout()

    fname = os.path.join(out_dir, f"summary_grid_{method_name.lower()}.png")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[attention_viz] Summary grid saved: {fname}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir",  type=str, required=True,
                        help="Path to trained run e.g. .../H16_K20_.../9")
    parser.add_argument("--n_frames", type=int, default=8,
                        help="Number of representative frames to visualize")
    parser.add_argument("--device",   type=str, default="cuda:0")
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--alpha",    type=float, default=0.5,
                        help="Heatmap overlay opacity (0=invisible, 1=fully opaque)")
    args = parser.parse_args()

    device = torch.device(args.device)

    # ── load model and dataset ────────────────────────────────────────────────
    print(f"[attention_viz] Loading: {args.run_dir}")
    diff_exp     = utils.load_diffusion(args.run_dir, epoch="best",
                                         device=str(device))
    dataset      = diff_exp.dataset
    diffusion    = diff_exp.diffusion.to(device)
    diffusion.eval()

    encoder      = diffusion.model.encoder
    encoder.eval()

    encoder_type = getattr(diffusion.model, "encoder_type", "unknown")
    latent_dim   = int(getattr(diffusion.model, "image_cond_dim", 256))
    print(f"[attention_viz] encoder={encoder_type}  latent_dim={latent_dim}")

    # ── output directory ──────────────────────────────────────────────────────
    out_dir = os.path.join(args.run_dir, "attention_viz")
    os.makedirs(out_dir, exist_ok=True)

    # ── sample representative frames ─────────────────────────────────────────
    print(f"\n[attention_viz] Sampling {args.n_frames} representative frames...")
    print("  (This scans x_pos for all dataset samples — may take ~1 min)")
    indices, x_pos_vals = sample_representative_frames(
        dataset, args.n_frames, seed=args.seed
    )

    # ── choose method based on encoder type ──────────────────────────────────
    is_vit = encoder_type.lower() in ("vit", "vitp", "vit_pretrained")
    is_cnn = encoder_type.lower() in ("cnn", "resnet")

    if not is_vit and not is_cnn:
        print(f"[attention_viz] WARNING: unknown encoder_type='{encoder_type}'")
        print("  Trying ViT attention first, then falling back to Grad-CAM...")

    # ── process each frame ────────────────────────────────────────────────────
    overlays    = []
    method_name = "Attn" if is_vit else "GradCAM"

    for frame_idx, (ds_idx, xp) in enumerate(zip(indices, x_pos_vals)):
        print(f"\n[attention_viz] Frame {frame_idx+1}/{args.n_frames}  "
              f"(dataset idx={ds_idx}  x_pos={xp:.3f})")

        obs_rgb_t, raw_img = load_frame(dataset, ds_idx, device)

        try:
            if is_vit or (not is_cnn):
                heatmap     = extract_vit_attention(encoder, obs_rgb_t)
                method_name = "Attn"
            else:
                heatmap     = extract_gradcam(encoder, obs_rgb_t)
                method_name = "GradCAM"

        except Exception as e:
            print(f"  [WARN] Extraction failed: {e}")
            print("  Filling with zero heatmap for this frame.")
            H, W = raw_img.shape[:2]
            heatmap = np.zeros((H, W), dtype=np.float32)

        overlay = save_frame_result(
            raw_img     = raw_img,
            heatmap     = heatmap,
            frame_idx   = frame_idx,
            x_pos_val   = xp,
            out_dir     = out_dir,
            method_name = method_name,
            encoder_type= encoder_type,
        )
        overlays.append(overlay)

    # ── summary grid ─────────────────────────────────────────────────────────
    print(f"\n[attention_viz] Saving summary grid...")
    save_summary_grid(
        overlays     = overlays,
        x_pos_vals   = x_pos_vals,
        out_dir      = out_dir,
        encoder_type = encoder_type,
        method_name  = method_name,
    )

    print(f"\n[attention_viz] Done. All outputs saved to: {out_dir}")
    for f in sorted(os.listdir(out_dir)):
        print(f"  {f}")


if __name__ == "__main__":
    main()