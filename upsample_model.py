
from torch import nn
import torch

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, in_channels*2, kernel_size, stride, padding),
            # nn.BatchNorm3d(in_channels*2),
            nn.ReLU(),
            # nn.GELU(),
            nn.Conv3d(in_channels*2, out_channels, kernel_size, stride, padding),
            # nn.BatchNorm3d(out_channels),
        )
        if in_channels != out_channels:
            self.shortcut = nn.Conv3d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        return self.shortcut(x) + self.block(x)

class ResidualStack(nn.Module):
    def __init__(self, channels, kernel_size=3, stride=1, padding=1, num_blocks=3):
        super().__init__()
        self.blocks = nn.ModuleList([
            ResidualBlock(channels, channels, kernel_size, stride, padding) for _ in range(num_blocks)
        ])
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, scale_factor=2):
        super().__init__()
        self.block = nn.Sequential(
            # nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding),
            nn.Upsample(scale_factor=(1, scale_factor, scale_factor)),
            ResidualBlock(in_channels, out_channels, kernel_size, stride, padding),
        )

    def forward(self, x):
        return self.block(x) 

class UpsampleModel(nn.Module):
    def __init__(self, base_channels=256, num_blocks=1):
        super().__init__()
        self.in_conv = nn.Conv3d(3, base_channels, 3, 1, 1)

        self.up1 = UpSample(base_channels, base_channels//2)
        self.up2 = UpSample(base_channels//2, base_channels//4)

        self.res0 = ResidualStack(base_channels, num_blocks=num_blocks)
        self.res1 = ResidualStack(base_channels//2, num_blocks=num_blocks)
        self.res2 = ResidualStack(base_channels//4, num_blocks=num_blocks)

        self.uup = nn.Upsample(scale_factor=(1, 4, 4))

        self.out_conv = nn.Conv3d(base_channels//4, 3, 3, 1, 1)

    def forward(self, x):
        x0 = self.uup(x)
        x = self.in_conv(x)
        x = self.res0(x)
        x = self.up1(x)
        x = self.res1(x)
        x = self.up2(x)
        x = self.res2(x)
        return self.out_conv(x) + x0