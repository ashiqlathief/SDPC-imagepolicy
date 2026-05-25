import torch
import torch.nn as nn
from .image_encoder import ImageObsEncoder
from .vit_obs_encoder import ViTObsEncoder
from .transformer_denoiserDIT import Transformer1DDenoisingModel
from .image_encoder import ImageObsEncoder
from .vit_obs_encoder import ViTObsEncoder, ViTObsEncoderPretrained

class RawPixelEncoder(nn.Module):
    """
    True raw pixel baseline.
    Full 96x96 image flattened directly as conditioning vector.
    """
    def __init__(self):
        super().__init__()
        self.out_dim = 3 * 96 * 96  # 27648

    def forward(self, obs_rgb: torch.Tensor) -> torch.Tensor:
        if obs_rgb.shape[-1] == 3:
            obs_rgb = obs_rgb.permute(0, 1, 4, 2, 3).contiguous()

        B, To, C, H, W = obs_rgb.shape
        x = obs_rgb.float()
        if x.max() > 1.5:
            x = x / 255.0

        x = x.view(B, To, -1)   # (B, To, 27648)
        x = x.mean(dim=1)        # (B, 27648)
        return x
    
class ImageCondTransformer1DModel(nn.Module):
    """
    Drop-in replacement for ImageCondUNet1DTemporalCondModel.
    Encoder is identical. Only the denoiser changes: UNet -> Transformer.
    """

    def __init__(
        self,
        horizon: int,
        action_dim: int,
        image_cond_dim: int = 256,
        d_model: int = 256,
        n_heads: int = 8,
        depth: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        encoder_type: str = "vit",
        vit_img_size: int = 96,
        vit_patch_size: int = 8,
        vit_depth: int = 6,
        vit_heads: int = 8,
        vit_mlp_ratio: float = 4.0,
        vit_dropout: float = 0.1,
        vit_attn_dropout: float = 0.0,
        returns_condition: bool = False,
        condition_dropout: float = 0.1,
        device=None,
        **kwargs,
    ):
        super().__init__()
        self.horizon = horizon
        self.action_dim = action_dim
        self.encoder_type = encoder_type.lower()
        self.image_cond_dim = image_cond_dim

        if self.encoder_type == "vitp":
            self.encoder = ViTObsEncoderPretrained(
                image_size=vit_img_size,
                patch_size=vit_patch_size,
                cond_dim=image_cond_dim,
                pretrained=True,
            )
        elif self.encoder_type == "vit":
            self.encoder = ViTObsEncoder(
                image_size=vit_img_size,
                patch_size=vit_patch_size,
                cond_dim=image_cond_dim,
                embed_dim=image_cond_dim,#embed_dim=vit_width,
                depth=vit_depth,
                num_heads=vit_heads,
                mlp_ratio=vit_mlp_ratio,
                dropout=vit_dropout,
                attn_dropout=vit_attn_dropout,
            )
        elif self.encoder_type == "raw_pixels":
            self.encoder = RawPixelEncoder()
            self.image_cond_dim = self.encoder.out_dim 
        elif self.encoder_type == "cnn":
            self.encoder = ImageObsEncoder(  #CNN
                cond_dim=image_cond_dim,
                pretrained=True
            )
        else:
            raise ValueError(f"Unsupported encoder_type: {encoder_type}")

        # Transformer denoiser
        self.core = Transformer1DDenoisingModel(
            horizon=horizon,
            action_dim=action_dim,
            image_cond_dim=self.image_cond_dim,
            d_model=d_model,
            n_heads=n_heads,
            depth=depth,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            returns_condition=returns_condition,
            condition_dropout=condition_dropout,
        )
        # Share the encoder with core so forward() can call self.encoder
        self.core.encoder = self.encoder

        if device:
            self.to(device)

    def forward(self, x, cond, time, returns=None, use_dropout=True, force_dropout=False):
        return self.core(x, cond, time, returns, use_dropout, force_dropout)