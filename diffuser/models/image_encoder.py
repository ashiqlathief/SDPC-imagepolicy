import torch
import torch.nn as nn
import torchvision.models as tvm

class ImageObsEncoder(nn.Module):
    """
    obs_rgb: (B, To, 3, H, W) -> cond: (B, cond_dim)
    """
    def __init__(self, cond_dim=256, pretrained=True):
        super().__init__()
        net = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        feat_dim = 512
        net.fc = nn.Identity()
        self.backbone = net
        self.proj = nn.Sequential(
            nn.Linear(512, cond_dim),    # compress to cond_dim first
            nn.Mish(),
            nn.LayerNorm(cond_dim),
            nn.Linear(cond_dim, cond_dim),  # refine at cond_dim width
            nn.Mish(),
            nn.LayerNorm(cond_dim),
        )
        # self.proj = nn.Sequential(
        #     nn.Linear(feat_dim, cond_dim),
        #     nn.Mish(),
        #     nn.LayerNorm(cond_dim),
        # )

    def forward(self, obs_rgb):
        B, To, C, H, W = obs_rgb.shape
        x = obs_rgb.reshape(B * To, C, H, W)
        feat = self.backbone(x)                  # (B*To, 512)
        feat = feat.view(B, To, -1).mean(dim=1)  # (B, 512)  (avg over To frames)
        return self.proj(feat)                   # (B, cond_dim)
