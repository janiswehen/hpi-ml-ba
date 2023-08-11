import torch
from unet.model.unet_3d import UNet3d
from unet.evaluator.evaluator import Evaluator

class FullUnetEvaluator(Evaluator):
    MODEL_TYPE = 'full_unet'
    PROJECT_NAME = 'Full-3D-UNet'

    def initModel(self):
        self.model = UNet3d(
            in_channels=self.test_dataset.chanels[0],
            out_channels=self.test_dataset.chanels[1]
        ).to(self.DEVICE)
        
        try:
            self.model.load_state_dict(torch.load(self.model_loading_config['path']))
            print(f'Loaded checkpoint from {self.model_loading_config["path"]}')
        except Exception as e:
            print(f'Unable to load checkpoint. {e}')
    
    def infer(self, scan):
        return self.model(scan)