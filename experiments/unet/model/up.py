import torch
import torch.nn as nn
import torch.nn.functional as F

from unet.model.double_conv import DoubleConv3d

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv3d(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # compute padding
        diffX = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffZ = x2.size()[4] - x1.size()[4]
        x1 = F.pad(x1, [diffZ // 2, diffZ - diffZ // 2,
                        diffY // 2, diffY - diffY // 2,
                        diffX // 2, diffX - diffX // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)