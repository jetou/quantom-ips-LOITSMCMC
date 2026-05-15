from quantom_ips.models.mlp import MLP
from quantom_ips.models.layers.conv2dstack import ConvTranspose2dStack
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf, listconfig
from typing import Tuple, Optional
import math

class TMDConv2DTGenerator(nn.Module):
    def __init__(
        self,
        mlp_in: int = 128,
        image_size: Tuple[int, int] = (20, 30),
        base_channels: int = 64,
        out_channels: int = 6,
        out_activation: Optional[str] = "softplus",
        x_min: float = 0.04,
        x_max: float = 1.0,
        bt_min: float = 1e-3,
        bt_max: float = 6.0,
        log_x: bool = True,
        log_bt: bool = False,
        ):
        super().__init__()
        H, W = image_size
        self.image_size = (H, W)

        seed_h, seed_w = max(1, H//2), max(1, W//2)
        self.seed_hw = (seed_h, seed_w)
        self.base_channels = base_channels

        self.fc = nn.Sequential(
            nn.Linear(mlp_in, base_channels * seed_h * seed_w),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.upsample = nn.Upsample(size=(H, W), mode="bilinear", align_corners=False)

        # +2 channels because we concat coord maps
        self.refine = nn.Sequential(
            nn.Conv2d(base_channels + 2, base_channels, 3, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, out_channels, 1),
        )

        self.out_act = None
        if out_activation is not None:
            if out_activation.lower() == "softplus":
                self.out_act = nn.Softplus(beta=1.0)
            elif out_activation.lower() == "sigmoid":
                self.out_act = nn.Sigmoid()
            else:
                raise ValueError(out_activation)

        # ---- build physical coord maps once ----
        x = torch.linspace(0, 1, H)
        bt = torch.linspace(0, 1, W)

        # map [0,1] -> physical grid
        if log_x:
            lx0, lx1 = math.log(x_min), math.log(x_max)
            x_phys = torch.exp(lx0 + (lx1 - lx0) * x)   # (H,)
            x_feat = torch.log(x_phys)                  # log(x) feature
        else:
            x_phys = x_min + (x_max - x_min) * x
            x_feat = x_phys

        if log_bt:
            lb0, lb1 = math.log(bt_min), math.log(bt_max)
            bt_phys = torch.exp(lb0 + (lb1 - lb0) * bt
            )
            bt_feat = torch.log(bt_phys)
        else:
            bt_phys = bt_min + (bt_max - bt_min) * bt
            bt_feat = bt_phys

        # normalize features to roughly [-1,1] for stable conditioning
        x_feat = 2 * (x_feat - x_feat.min()) / (x_feat.max() - x_feat.min() + 1e-12) - 1
        bt_feat = 2 * (bt_feat - bt_feat.min()) / (bt_feat.max() - bt_feat.min() + 1e-12) - 1

        # make 2D maps (B,1,H,W) at runtime via expand
        x_map  = x_feat.view(1, 1, H, 1).expand(1, 1, H, W).clone()
        bt_map = bt_feat.view(1, 1, 1, W).expand(1, 1, H, W).clone()

        self.register_buffer("x_map", x_map, persistent=False)
        self.register_buffer("bt_map", bt_map, persistent=False)


    def forward(self, z):
        B = z.shape[0]
        seed_h, seed_w = self.seed_hw

        x = self.fc(z).view(B, self.base_channels, seed_h, seed_w)
        x = self.upsample(x)

        # concat physical coord channels
        coord_scale = 1
        x = torch.cat([
            x,
            coord_scale * self.x_map.expand(B, -1, -1, -1),
            coord_scale * self.bt_map.expand(B, -1, -1, -1),
        ], dim=1)


        x = self.refine(x)
        if self.out_act is not None:
            x = self.out_act(x)
        return x
     





