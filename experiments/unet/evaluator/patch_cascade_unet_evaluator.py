import torch
import torch.nn.functional as F
from monai.networks.utils import one_hot

from unet.model.unet import UNet3d
from unet.evaluator.evaluator import Evaluator

class PatchCascadeUnetEvaluator(Evaluator):
    MODEL_TYPE = 'patch_cascade_unet'
    PROJECT_NAME = 'Patch-Cascade-UNet'
    
    def initModel(self):
        self.stage1 = UNet3d(
            in_channels=self.dataset.chanels[0],
            out_channels=self.dataset.chanels[1]
        ).to(self.DEVICE)
        self.stage2 = UNet3d(
            in_channels=self.dataset.chanels[0] + self.dataset.chanels[1],
            out_channels=self.dataset.chanels[1]
        ).to(self.DEVICE)
        self.stage1.load_state_dict(torch.load(self.model_loading_config['path1']))
        self.stage2.load_state_dict(torch.load(self.model_loading_config['path2']))
        self.org_shape = self.dataset[0][0].shape[-3:]
        self.scaled_shape = (
            self.data_loading_config['scaling']['w'],
            self.data_loading_config['scaling']['h'],
            self.data_loading_config['scaling']['d']
        )
        self.down_scale = torch.nn.Upsample(scale_factor=self.scaled_shape, mode='trilinear', align_corners=True)
        self.up_scale = torch.nn.Upsample(size=self.org_shape, mode='trilinear', align_corners=True)

    def stage1_infer(self, scan):
        scan_low_res = self.down_scale(scan).to(self.DEVICE)
        pred_low_res = self.stage1(scan_low_res).detach().cpu()
        pred_low_res = self.one_hot_arg_max(pred_low_res)
        self.up_scale.size = scan.shape[-3:]
        pred = self.up_scale(pred_low_res)
        scan_guided = torch.cat([scan, pred], dim=1)
        return scan_guided

    def one_hot_arg_max(self, x: torch.Tensor) -> torch.Tensor:
        a = x.argmax(dim=1, keepdim=True)
        return one_hot(a, x.shape[1])

    def stage2_infer(self, scan_guided):
        D = scan_guided.shape[self.data_loading_config['slice_axis'] - 3]
        patch_count = D // self.data_loading_config['patch_size'] + 1
        padding = patch_count * self.data_loading_config['patch_size'] - D
        if padding > 0:
            if self.data_loading_config['slice_axis'] == 0:
                scan_guided = F.pad(scan_guided, (0, 0, 0, 0, 0, padding))
            elif self.data_loading_config['slice_axis'] == 1:
                scan_guided = F.pad(scan_guided, (0, 0, 0, padding))
            else:
                scan_guided = F.pad(scan_guided, (0, padding))
        preds = []
        for idx in range(patch_count):
            start = idx*self.data_loading_config['patch_size']
            end = start + self.data_loading_config['patch_size']
            if self.data_loading_config['slice_axis'] == 0:
                patch = scan_guided[..., start:end, :, :].to(self.DEVICE)
            elif self.data_loading_config['slice_axis'] == 1:
                patch = scan_guided[..., start:end, :].to(self.DEVICE)
            else:
                patch = scan_guided[..., start:end].to(self.DEVICE)
            pred = self.stage2(patch).detach().cpu()
            preds.append(pred)
        pred = torch.cat(preds, dim=self.data_loading_config['slice_axis'] - 3)
        if self.data_loading_config['slice_axis'] == 0:
            pred = pred[..., :D, :, :]
        elif self.data_loading_config['slice_axis'] == 1:
            pred = pred[..., :D, :]
        else:
            pred = pred[..., :D]
        
        return pred
    
    def infer(self, scan):
        scan_guided = self.stage1_infer(scan)
        pred = self.stage2_infer(scan_guided)
        return pred