import torch.nn as nn

class OutConv3d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv3d, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)