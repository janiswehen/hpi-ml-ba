import torch
import torch.nn as nn
import torch.nn.functional as F

from unet.model.double_conv import DoubleConv3d, DoubleConv2d
from unet.model.up import Up3d, Up2d
from unet.model.down import Down3d, Down2d
from unet.model.out_conv import OutConv3d, OutConv2d


class UNet3d(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, input_shape=None):
        super(UNet3d, self).__init__()
        # just used to compute the model shape (not needed)
        self.input_shape = input_shape
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = [32, 64, 128, 256, 512]

        self.inc = DoubleConv3d(in_channels, self.channels[0])
        self.down1 = Down3d(self.channels[0], self.channels[1])
        self.down2 = Down3d(self.channels[1], self.channels[2])
        self.down3 = Down3d(self.channels[2], self.channels[3])
        self.down4 = Down3d(self.channels[3], self.channels[4])
        self.up1 = Up3d(self.channels[4], self.channels[3])
        self.up2 = Up3d(self.channels[3], self.channels[2])
        self.up3 = Up3d(self.channels[2], self.channels[1])
        self.up4 = Up3d(self.channels[1], self.channels[0])
        self.outc = OutConv3d(self.channels[0], out_channels)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

def test():
    x = torch.randn((1, 3, 35, 72, 287))
    model = UNet3d(in_channels=3, out_channels=1, input_shape=x.shape)
    preds = model(x)
    
    assert preds.shape[-3:] == x.shape[-3:]

if __name__ == "__main__":
    test()