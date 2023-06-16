import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class Double3DConv(nn.Module):
    def __init__(self, in_chanels, out_chanels):
        super(Double3DConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_chanels, out_chanels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(out_chanels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_chanels, out_chanels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(out_chanels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.conv(x)

class FullUNet(nn.Module):
    def __init__(self, in_chanels=3, out_chanels=1, features=[64, 128, 256, 512]):
        super(FullUNet, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        
        # Down part of UNet
        for feature in features:
            self.downs.append(Double3DConv(in_chanels, feature))
            in_chanels = feature
        
        # Up part of UNet
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose3d(
                    feature*2, feature, kernel_size=2, stride=2
                )
            )
            self.ups.append(Double3DConv(feature*2, feature))
        
        self.bottleneck = Double3DConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv3d(features[0], out_chanels, kernel_size=1)
    
    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]
            
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])
            
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)
        
        return self.final_conv(x)

class PaddedUNet(nn.Module):
    def __init__(self, unet):
        super(PaddedUNet, self).__init__()
        self.unet = unet
    
    def forward(self, x):
        # Compute padding sizes
        depth = len(self.unet.downs)
        padding = [(0, (d % 2**depth != 0) * (2**depth - d % 2**depth)) for d in x.shape[-3:]]
        padding = [item for sublist in padding[::-1] for item in sublist]  # flatten the list

        # Pad input and pass it through the UNet
        x_padded = nn.functional.pad(x, padding)
        out_padded = self.unet(x_padded)

        # Remove padding from the output
        i = [slice(None)] * len(x.shape)  # slices to keep everything
        i[-3:] = [slice(0, d) for d in x.shape[-3:]]  # slices to remove padding from last 3 dims
        out = out_padded[i]

        return out

def test():
    batch_size = 3
    input_chanels = 3
    output_chanels = 3
    
    x = torch.randn((batch_size, input_chanels, 240, 240, 32))
    
    model = FullUNet(in_chanels=input_chanels, out_chanels=output_chanels)
    paddedModel = PaddedUNet(model)
    
    preds = paddedModel(x)
    
    print(preds.shape)
    print(x.shape)

if __name__ == "__main__":
    test()