import torch
import time
import tqdm
import wandb
from torch.utils.data import DataLoader
from monai.losses import DiceLoss
from math import floor

from unet.dataset.msd_dataset import MSDDataset, Split, MSDTask

class Evaluator():
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    MODEL_TYPE = 'not implemented'
    PROJECT_NAME = 'not implemented'

    def __init__(self, config):
        assert config['model_type'] == self.MODEL_TYPE
        torch.set_float32_matmul_precision('medium')
        self.run_name = config['name']
        self.data_loading_config = config['data_loading']
        self.model_loading_config = config['model_loading']
        self.eval_config = config['eval']
        self.logging_config = config['logging']

        if self.logging_config['enabled'] == True:
            wandb.init(project=self.PROJECT_NAME, name=self.run_name, config=config)

        self.task = MSDTask.fromStr(self.data_loading_config['task'])
        self.dataset = MSDDataset(
            msd_task=self.task,
            split = Split.Test,
            split_ratio = (
                self.data_loading_config['split_ratio']['train'],
                self.data_loading_config['split_ratio']['val'],
                self.data_loading_config['split_ratio']['test']
            ),
            seed=self.data_loading_config['seed'],
            normalize=True
        )

        self.loader = DataLoader(
            dataset=self.dataset,
            batch_size=self.data_loading_config['batch_size'],
            num_workers=self.data_loading_config['n_workers'],
            shuffle=False,
        )
        self.epochs = self.eval_config['n_steps'] // (len(self.dataset) // self.data_loading_config['batch_size'])

        self.initModel()
        self.loss_fn = DiceLoss(softmax=True, include_background=False)

    def initModel(self):
        raise NotImplementedError

    def infer(self, scan: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def log_predictions(self):
        if not self.logging_config['enabled']:
            return
        wandb_images = []
        for i in range(self.logging_config['prediction_log_count']):
            scan, ground_trouth = self.dataset[i]
            pred = self.infer(scan.unsqueeze(0))[0]

            class_labels = self.dataset.class_labels

            ground_trouth = ground_trouth.argmax(dim=0)
            pred = pred.argmax(dim=0)
            scan = scan[self.logging_config['modality']] if self.logging_config['modality'] < self.dataset.chanels[0] else scan[0]
            slice_idx = floor(scan.shape[-3:][self.logging_config['slice_axis']] * self.logging_config['rel_slice'])
            wandb_images.append(wandb.Image(
                torch.select(scan, self.logging_config['slice_axis'], slice_idx),
                caption=f"Scan {i}",
                masks={
                    "prediction": {
                        "mask_data": torch.select(pred, self.logging_config['slice_axis'], slice_idx).cpu().numpy(),
                        "class_labels": class_labels,
                    },
                    "ground-trouth": {
                        "mask_data": torch.select(ground_trouth, self.logging_config['slice_axis'], slice_idx).cpu().numpy(),
                        "class_labels": class_labels,
                    },
                }
            ))
        wandb.log({"predictions": wandb_images})

    def evaluate(self):
        self.log_predictions()
        losses = []
        infer_times = []
        for epoch in range(self.epochs):
            loop = tqdm.tqdm(
                self.loader,
                desc=f"Test-Epoch {epoch + 1}/{self.epochs}"
            )
            for batchIdx, (scan, label) in enumerate(loop):
                start = time.time()
                pred = self.infer(scan)
                end = time.time()
                loss = self.loss_fn(pred, label)
                losses.append(loss.item())
                infer_times.append(end - start)
                if self.logging_config['enabled'] == True:
                    wandb.log({'step_loss': loss.item()})
                    wandb.log({'step_time': end - start})

        max_memory = torch.cuda.max_memory_allocated(self.DEVICE) / pow(10,9)
        loss = sum(losses) / len(losses)
        infer_time = sum(infer_times) / len(infer_times)
        if self.logging_config['enabled'] == True:
            wandb.log({'loss': loss})
            wandb.log({'time': infer_time})
            wandb.log({'max_memory': max_memory})