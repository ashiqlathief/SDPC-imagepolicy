import torch
import torch.nn as nn
import torchvision.models as tvm

class ImageObsEncoder(nn.Module):
    """
    obs_rgb: (B, To, 3, H, W) -> cond: (B, cond_dim)
    """
    def __init__(self, cond_dim=256, pretrained=True, in_chans=3):
        super().__init__()
        self.in_chans = in_chans
        net = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        feat_dim = 512
        net.fc = nn.Identity()
        if in_chans != 3:
            old_conv1 = net.conv1
            new_conv1 = nn.Conv2d(in_chans, old_conv1.out_channels, kernel_size=old_conv1.kernel_size,
                                   stride=old_conv1.stride, padding=old_conv1.padding, bias=False)
            with torch.no_grad():
                new_conv1.weight[:, :3] = old_conv1.weight
                if in_chans > 3:
                    # init extra (e.g. depth) channels as the mean of the RGB filters
                    new_conv1.weight[:, 3:] = old_conv1.weight.mean(dim=1, keepdim=True)
            net.conv1 = new_conv1
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
