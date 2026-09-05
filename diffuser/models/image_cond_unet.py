import torch
import torch.nn as nn

from .image_encoder import ImageObsEncoder
from .vit_obs_encoder import ViTObsEncoder, ViTObsEncoderPretrained
from .unet1d_temporal_cond import UNet1DTemporalCondModel

class RawPixelEncoder(nn.Module):
    """
    True raw pixel baseline.
    Full 96x96 image flattened directly as conditioning vector.
    """
    def __init__(self, img_size=96):
        super().__init__()
        self.out_dim = 3 * img_size * img_size  # 27648 for 3x96x96

    def forward(self, obs_rgb: torch.Tensor) -> torch.Tensor:
        if obs_rgb.shape[-1] == 3 and obs_rgb.shape[2] != 3:
            obs_rgb = obs_rgb.permute(0, 1, 4, 2, 3).contiguous()

        B, To, C, H, W = obs_rgb.shape
        x = obs_rgb.float()
        if x.max() > 1.5:
            x = x / 255.0

        x = x.view(B, To, -1)   # (B, To, 27648)
        x = x.mean(dim=1)        # (B, 27648)
        return x
    
class ImageCondUNet1DTemporalCondModel(nn.Module):
    def __init__(
        self,
        horizon: int,
        action_dim: int,
        image_cond_dim: int = 256,
        dim: int = 64,
        dim_mults=(1, 2, 4, 8),
        returns_condition: bool = False,
        condition_dropout: float = 0.1,
        encoder_type: str = "vit",   # "cnn" or "vit" or "raw_pixels"
        # ViT-only knobs (safe defaults)
        vit_img_size: int = 96,
        vit_patch_size: int = 8,
        vit_width: int = 256,
        vit_depth: int = 6,
        vit_heads: int = 8,
        vit_mlp_ratio: float = 4.0,
        vit_dropout: float = 0.1,
        vit_attn_dropout: float = 0.0,
        device: str = None,
        **kwargs,
    ):
        super().__init__()
        self.horizon = horizon
        self.action_dim = action_dim
        self.image_cond_dim = image_cond_dim
        self.encoder_type = encoder_type.lower()

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
            self.encoder = RawPixelEncoder(img_size=vit_img_size)
            self.image_cond_dim = self.encoder.out_dim
        elif self.encoder_type == "cnn":
            self.encoder = ImageObsEncoder(  #CNN
                cond_dim=image_cond_dim,
                pretrained=True,
            )
        else:
            raise ValueError(f"Unsupported encoder_type: {encoder_type}")

        # ------------------------------------------------------------------
        # 2) Temporal denoiser UNet denoises [action || image_condition] jointly.
        # ------------------------------------------------------------------
        self.core = UNet1DTemporalCondModel(
            horizon=horizon,
            transition_dim=action_dim + image_cond_dim,
            cond_dim=0,  # no separate "state condition"; image cond is concatenated
            dim=dim,
            dim_mults=dim_mults,
            returns_condition=returns_condition,
            condition_dropout=condition_dropout,
            **kwargs,  # forwards any extra compatible args
        )

        # optional compatibility (no-op unless you want to force module device)
        if device is not None:
            self.to(device)

    def forward(self, x, cond, time, returns=None, use_dropout=True, force_dropout=False):
        """
        x:      (B, H, action_dim) noisy action sequence
        cond:   dict with 'obs_rgb': (B, To, 3, Himg, Wimg)
        time:   diffusion timestep tensor/scalar
        """
        if not isinstance(cond, dict):
            raise TypeError(f"Expected cond to be dict, got {type(cond)}")
        if "obs_rgb" not in cond:
            raise KeyError("cond must contain key 'obs_rgb'")

        rgb = cond["obs_rgb"]  # (B, To, 3, H, W)
        B, H, A = x.shape
        if A != self.action_dim:
            raise ValueError(f"x action dim mismatch: got {A}, expected {self.action_dim}")
        if H != self.horizon:
            # allow mismatch if you intentionally use variable horizon, but warn by raising for now
            raise ValueError(f"x horizon mismatch: got {H}, expected {self.horizon}")

        # Encode image observations -> (B, image_cond_dim)
        cond_vec = self.encoder(rgb)

        # Repeat across prediction horizon -> (B, H, image_cond_dim)
        cond_seq = cond_vec.unsqueeze(1).expand(B, H, self.image_cond_dim)

        # Concatenate noisy actions and image condition
        x_in = torch.cat([x, cond_seq], dim=-1)

        # UNet predicts noise over the full concatenated channel dim
        eps_full = self.core(
            x_in,
            cond=None,
            time=time,
            returns=returns,
            use_dropout=use_dropout,
            force_dropout=force_dropout,
        )

        # Return only the action part (diffusion target is action noise)
        eps_action = eps_full[..., : self.action_dim]
        return eps_action


class ImagePoseCondUNet1DTemporalCondModel(nn.Module):
    """
    Same as ImageCondUNet1DTemporalCondModel, plus goal-relative pose conditioning.

    cond must additionally contain:
        "goal_rel": (B, pose_dim) relative displacement to the goal
                     (pose_target - pose_now, computed in CrazyflieImageDataset)
    (see CrazyflieImageDataset(use_pose_cond=True)).

    Fed raw (no MLP currently) alongside the image embedding, then widened across
    the horizon -- everything downstream (the UNet core) is identical to the
    image-only model, just with a wider transition_dim.
    """

    def __init__(
        self,
        horizon: int,
        action_dim: int,
        image_cond_dim: int = 256,
        pose_dim: int = 3,
        pose_cond_dim: int = 64,
        dim: int = 64,
        dim_mults=(1, 2, 4, 8),
        returns_condition: bool = False,
        condition_dropout: float = 0.1,
        encoder_type: str = "vit",
        vit_img_size: int = 96,
        vit_patch_size: int = 8,
        vit_width: int = 256,
        vit_depth: int = 6,
        vit_heads: int = 8,
        vit_mlp_ratio: float = 4.0,
        vit_dropout: float = 0.1,
        vit_attn_dropout: float = 0.0,
        device: str = None,
        **kwargs,
    ):
        super().__init__()
        self.horizon = horizon
        self.action_dim = action_dim
        self.image_cond_dim = image_cond_dim
        self.pose_dim = pose_dim
        self.pose_cond_dim = pose_cond_dim
        self.encoder_type = encoder_type.lower()

        if self.encoder_type == "vitp":
            self.encoder = ViTObsEncoderPretrained(
                image_size=vit_img_size,
                patch_size=vit_patch_size,
                cond_dim=image_cond_dim,
                pretrained=True,
            )
        else:
            raise ValueError(f"Unsupported encoder_type: {encoder_type}")

        self.pose_encoder = nn.Sequential(  # unused for now -- pose_target fed to the UNet raw, see forward()
            nn.Linear(pose_dim, 64),
            nn.ReLU(),
            nn.Linear(64, pose_cond_dim),
        )

        self.core = UNet1DTemporalCondModel(
            horizon=horizon,
            transition_dim=action_dim + self.image_cond_dim + pose_dim,
            cond_dim=0,  # image + pose cond is concatenated, not passed as a separate "state condition"
            dim=dim,
            dim_mults=dim_mults,
            returns_condition=returns_condition,
            condition_dropout=condition_dropout,
            **kwargs,
        )

        if device is not None:
            self.to(device)

    def forward(self, x, cond, time, returns=None, use_dropout=True, force_dropout=False):
        if not isinstance(cond, dict):
            raise TypeError(f"Expected cond to be dict, got {type(cond)}")
        if "obs_rgb" not in cond:
            raise KeyError("cond must contain key 'obs_rgb'")
        if "goal_rel" not in cond:
            raise KeyError("cond must contain key 'goal_rel' (relative displacement pose_target - pose_now)")

        rgb = cond["obs_rgb"]
        B, H, A = x.shape # B=batch size, H=horizon, A=action_dim x=noisy action sequence
        if A != self.action_dim:
            raise ValueError(f"x action dim mismatch: got {A}, expected {self.action_dim}")
        if H != self.horizon:
            raise ValueError(f"x horizon mismatch: got {H}, expected {self.horizon}")

        image_z = self.encoder(rgb)  # (B, image_cond_dim)
        goal_rel = cond["goal_rel"]  # (B, pose_dim) -- already pose_target - pose_now (see CrazyflieImageDataset)

        cond_vec = torch.cat([image_z, goal_rel], dim=-1)  # (B, image_cond_dim+pose_dim)
        cond_seq = cond_vec.unsqueeze(1).expand(B, H, cond_vec.shape[-1])

        x_in = torch.cat([x, cond_seq], dim=-1)

        eps_full = self.core(
            x_in,
            cond=None,
            time=time,
            returns=returns,
            use_dropout=use_dropout,
            force_dropout=force_dropout,
        )
        return eps_full[..., : self.action_dim]