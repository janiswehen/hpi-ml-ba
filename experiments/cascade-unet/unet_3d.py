import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet3d(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, input_shape=None):
        super(UNet3d, self).__init__()
        # just used to compute the model shape (not needed)
        self.input_shape = input_shape
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = [32, 64, 128, 256, 512]

        self.inc = DoubleConv3d(in_channels, self.channels[0])
        self.down1 = Down(self.channels[0], self.channels[1])
        self.down2 = Down(self.channels[1], self.channels[2])
        self.down3 = Down(self.channels[2], self.channels[3])
        self.down4 = Down(self.channels[3], self.channels[4])
        self.up1 = Up(self.channels[4], self.channels[3])
        self.up2 = Up(self.channels[3], self.channels[2])
        self.up3 = Up(self.channels[2], self.channels[1])
        self.up4 = Up(self.channels[1], self.channels[0])
        self.outc = OutConv(self.channels[0], out_channels)

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
    
    def get_next_shape(self, shape):
        return (shape[0], shape[1]*2, shape[2]//2, shape[3]//2, shape[4]//2)
    
    def get_model_shape(self):
        if self.input_shape is None:
            return []
        in_shape = self.input_shape
        shape_1 = (in_shape[0], self.channels[0], in_shape[2], in_shape[3], in_shape[4])
        shape_2 = self.get_next_shape(shape_1)
        shape_3 = self.get_next_shape(shape_2)
        shape_4 = self.get_next_shape(shape_3)
        shape_5 = self.get_next_shape(shape_4)
        return [in_shape, shape_1, shape_2, shape_3, shape_4, shape_5]

    def print_model_shape(self):
        for index, shape in enumerate(self.get_model_shape()):
            if index == 0:
                print(f"In-Shape:\t{shape[0]}, {shape[1]}, {shape[2]}, {shape[3]}, {shape[4]}")
            else:
                print(f"Shape{index}:\t\t{shape[0]}, {shape[1]}, {shape[2]}, {shape[3]}, {shape[4]}")


class DoubleConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv3d(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


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


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


def test():
    x = torch.randn((1, 3, 35, 72, 287))
    model = UNet3d(in_channels=3, out_channels=1, input_shape=x.shape)
    model.print_model_shape()
    preds = model(x)
    
    assert preds.shape[-3:] == x.shape[-3:]

if __name__ == "__main__":
    test()