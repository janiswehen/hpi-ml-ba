import torch
import torch.nn.functional as F

from unet.model.unet_3d import UNet3d
from unet.evaluator.evaluator import Evaluator

class PatchUnetEvaluator(Evaluator):
    MODEL_TYPE = 'patch_unet'
    PROJECT_NAME = 'Patch-3D-UNet'
    
    def initModel(self):
        self.model = UNet3d(
            in_channels=self.dataset.chanels[0],
            out_channels=self.dataset.chanels[1]
        ).to(self.DEVICE)
        self.model.load_state_dict(torch.load(self.model_loading_config['path']))
    
    def infer(self, scan):
        if self.eval_config['slice_axis'] == 0:
            scan = scan.permute(0,1,4,3,2)
        elif self.eval_config['slice_axis'] == 1:
            scan = scan.permute(0,1,2,4,3)

        D = scan.shape[-1]
        patch_count = D // self.data_loading_config['patch_size'] + 1
        padding = patch_count * self.data_loading_config['patch_size'] - D
        if padding > 0:
            scan = F.pad(scan, (0, padding))
        preds = []
        for idx in range(patch_count):
            start = idx*self.data_loading_config['patch_size']
            end = start + self.data_loading_config['patch_size']
            patch = scan[..., start:end].to(self.DEVICE)
            pred = self.model(patch).detach().cpu()
            preds.append(pred)
        pred = torch.cat(preds, dim=-1)
        pred = pred[..., :D]
        if self.eval_config['slice_axis'] == 0:
            pred = pred.permute(0,1,4,3,2)
        elif self.eval_config['slice_axis'] == 1:
            pred = pred.permute(0,1,2,4,3)
            
        return pred[..., :D]