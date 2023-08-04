from unet_3d import DoubleConv3d
import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_float32_matmul_precision('medium')


in1 = torch.randn((1, 3, 255, 255, 255))
in2 = torch.randn((1, 3, 255, 255, 180))
model = DoubleConv3d(in_channels=3, out_channels=1).to(DEVICE)
out1 = model(in1.to(DEVICE))
out2 = model(in2.to(DEVICE))