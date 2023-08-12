import torch
from torch.nn import BatchNorm3d

from unet.model.unet_3d import UNet3d
from unet.evaluator.evaluator import Evaluator
from unet.model.double_conv import DoubleConv3d

class FullUnetEvaluator(Evaluator):
    MODEL_TYPE = 'full_unet'
    PROJECT_NAME = 'Full-3D-UNet'
    
    def initModel(self):
        self.model = UNet3d(
            in_channels=self.dataset.chanels[0],
            out_channels=self.dataset.chanels[1]
        ).to(self.DEVICE)
        self.model.load_state_dict(torch.load(self.model_loading_config['path']))
    
    def infer(self, scan):
        scan = scan.to(self.DEVICE)
        pred = self.model(scan).detach().cpu()
        return pred