import torch
import time
import tqdm
import wandb
from torch.utils.data import DataLoader
from monai.losses import DiceLoss
from math import floor
import csv

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
            batch_size=1,
            num_workers=self.data_loading_config['n_workers'],
            shuffle=False,
        )
        self.epochs = self.eval_config['n_steps'] // len(self.dataset)

        self.initModel()
        self.loss_fn = DiceLoss(softmax=True, include_background=False, reduction='none')
        self.softmax = torch.nn.Softmax(dim=1)

    def initModel(self):
        raise NotImplementedError

    def infer(self) -> torch.Tensor:
        raise NotImplementedError

    def log_predictions(self):
        if not self.logging_config['enabled']:
            return
        wandb_images = []
        for i in range(self.logging_config['prediction_log_count']):
            scan, ground_trouth = self.dataset[i]
            pred = self.infer(scan.unsqueeze(0))
            pred = self.softmax(pred)[0]

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

    def write_csv(self, losses, infer_times):
        with open(f'./csv-logs/{self.run_name}-{self.MODEL_TYPE}-{self.task.value[0]}', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow([f'loss_{self.dataset.class_labels[i+1]}' for i in range(len(losses))] + ['infer_time'])
            for values in zip(*losses, infer_times):
                writer.writerow([*values])

    def evaluate(self):
        self.log_predictions()
        losses = [[] for _ in range(1, self.dataset.chanels[1])]
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
                loss = torch.reshape(loss, (-1,))
                for i in range(len(loss)):
                    losses[i].append(loss[i].item())
                infer_times.append(end - start)
                if self.logging_config['enabled'] == True:
                    for i in range(len(loss)):
                        wandb.log({f'step_loss_{self.dataset.class_labels[i+1]}': loss[i].item()})
                    wandb.log({'step_time': end - start})

        max_memory = torch.cuda.max_memory_allocated(self.DEVICE) / pow(10,9)
        self.write_csv(losses, infer_times)
        loss = [sum(losses[i]) / len(losses[i]) for i in range(len(losses))]
        infer_time = sum(infer_times) / len(infer_times)
        if self.logging_config['enabled'] == True:
            for i in range(len(loss)):
                wandb.log({f'loss_{self.dataset.class_labels[i+1]}': loss[i]})
            wandb.log({'time': infer_time})
            wandb.log({'max_memory': max_memory})