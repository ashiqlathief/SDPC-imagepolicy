# diffuser/models/vit_obs_encoder.py
import math
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


def _to_float_chw(x: torch.Tensor) -> torch.Tensor:
    """
    Accepts:
      - (B, To, H, W, 3) uint8 or float
      - (B, To, 3, H, W) uint8 or float
    Returns:
      - (B*To, 3, H, W) float32 in [0,1]
    """
    if x.dim() != 5:
        raise ValueError(f"Expected 5D tensor, got shape={tuple(x.shape)}")

    # (B,To,H,W,3) -> (B,To,3,H,W)
    if x.shape[-1] == 3 and x.shape[2] != 3:
        x = x.permute(0, 1, 4, 2, 3).contiguous()

    # now should be (B,To,3,H,W)
    if x.shape[2] != 3:
        raise ValueError(f"Expected channel dim=3 at index 2, got shape={tuple(x.shape)}")

    if x.dtype == torch.uint8:
        x = x.float() / 255.0
    else:
        x = x.float()
        # If your pipeline already gives [0,1], this is fine.
        # If it gives [0,255], normalize here:
        if x.max() > 1.5:
            x = x / 255.0

    B, To, C, H, W = x.shape
    x = x.view(B * To, C, H, W)
    return x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding using Conv2d."""
    def __init__(self, img_size=96, patch_size=16, embed_dim=256):
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size

        self.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B,3,H,W)
        x = self.proj(x)            # (B, D, Gh, Gw)
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)
        return x


class ViTObsEncoder(nn.Module):
    """
    Minimal ViT-style encoder

    Input:  images (B, To, H, W, 3) or (B, To, 3, H, W)
    Output: cond  (B, cond_dim)

    Strategy:
      - encode each frame with ViT -> CLS embedding (B*To, D)
      - reshape back -> (B, To, D)
      - pool over To (mean by default) -> (B, D)
      - project -> (B, cond_dim)
    """
    def __init__(
        self,
        image_size: int = 96,
        patch_size: int = 16,
        embed_dim: int = 256,
        depth: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        cond_dim: int = 512,
        To: int = 2,
        temporal_pool: str = "mean",   # "mean" or "last"
    ):
        super().__init__()
        self.To = To
        self.temporal_pool = temporal_pool

        # Convert image to patches
        self.patch = PatchEmbed(image_size, patch_size, embed_dim)
        num_patches = self.patch.num_patches

        # CLS token + positional embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + num_patches, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        # Transformer blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Linear(embed_dim, cond_dim)

        self._init_weights()

        # Optional: normalize like ViT/CLIP style (helps a bit). ImageNet stats.
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.register_buffer("img_mean", torch.tensor(mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("img_std",  torch.tensor(std).view(1, 3, 1, 1), persistent=False)

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        # proj/norm already have decent defaults

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # images -> (B*To,3,H,W) in [0,1]
        x = _to_float_chw(images)

        # normalize
        x = (x - self.img_mean) / self.img_std

        # patchify -> tokens
        tok = self.patch(x)  # (B*To, N, D)
        BT, N, D = tok.shape

        cls = self.cls_token.expand(BT, -1, -1)  # (B*To,1,D)
        tok = torch.cat([cls, tok], dim=1)       # (B*To,1+N,D)
        tok = tok + self.pos_embed[:, : (1 + N), :]
        tok = self.pos_drop(tok)
        # Transformer attention blocks That is where attention happens
        tok = self.blocks(tok)   # (B*To,1+N,D)
        tok = self.norm(tok)
        
        # Take CLS token as image feature
        frame_feat = tok[:, 0]   # CLS token (B*To,D)

        # back to (B,To,D)
        # If caller passes To different from self.To, infer it:
        B = images.shape[0]
        To = images.shape[1]
        frame_feat = frame_feat.view(B, To, D)

        if self.temporal_pool == "mean":
            feat = frame_feat.mean(dim=1)        # (B,D)
        elif self.temporal_pool == "last":
            feat = frame_feat[:, -1, :]          # (B,D)
        else:
            raise ValueError(f"Unknown temporal_pool={self.temporal_pool}")

        cond = self.proj(feat)                   # (B,cond_dim)
        return cond
    
class ViTObsEncoderPretrained(nn.Module):
    """
    Pretrained ViT encoder using timm

    Input:  (B, To, 3, H, W)  or  (B, To, H, W, 3)
    Output: (B, cond_dim)

    The backbone weights come from ImageNet pretraining
    Only the projection head is randomly initialized
    """
    def __init__(
        self,
        image_size: int = 96,
        patch_size: int = 16,
        cond_dim: int = 512,
        temporal_pool: str = "mean",      # "mean" or "last"
        pretrained: bool = True,          # set False to compare random-init
        model_name: str = "vit_small_patch8_224",  # timm model name
    ):
        super().__init__()
        self.temporal_pool = temporal_pool

        # Load pretrained ViT backbone from timm
        # num_classes=0 removes the classification head
        # global_pool='token' uses the CLS token as output (same as your current design)
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,           # no classification head
            global_pool='token',     # output is CLS token
            img_size=image_size,
            in_chans=3,
        )

        # Get the embed_dim from the loaded backbone
        embed_dim = self.backbone.embed_dim  # e.g. 384 for vit_small

        # # Freeze everything in backbone
        # for param in self.backbone.parameters():
        #     param.requires_grad = False
        
        # Same projection head as your current ViTObsEncoder
        self.proj = nn.Linear(embed_dim, cond_dim)

        # ImageNet normalization
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.register_buffer("img_mean", torch.tensor(mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("img_std",  torch.tensor(std).view(1, 3, 1, 1), persistent=False)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # images → (B*To, 3, H, W) in [0,1]
        x = _to_float_chw(images)

        # Normalize
        x = (x - self.img_mean) / self.img_std

        # Pretrained ViT backbone → CLS token → (B*To, embed_dim)
        frame_feat = self.backbone(x)

        # Recover B and To dimensions
        B = images.shape[0]
        To = images.shape[1]
        D = frame_feat.shape[-1]
        frame_feat = frame_feat.view(B, To, D)  # (B, To, embed_dim)

        # Temporal pooling over To frames
        if self.temporal_pool == "mean":
            feat = frame_feat.mean(dim=1)    # (B, embed_dim)
        elif self.temporal_pool == "last":
            feat = frame_feat[:, -1, :]      # (B, embed_dim)
        else:
            raise ValueError(f"Unknown temporal_pool={self.temporal_pool}")

        return self.proj(feat)               # (B, cond_dim)