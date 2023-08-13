import torch
import wandb
import tqdm
import time
import os

from torch.utils.data import DataLoader
from monai.losses import DiceLoss
from math import floor

from unet.dataset.msd_dataset import Split, MSDTask, MSDDataset
from unet.model.unet_3d import UNet3d

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class FullUnetTrainer():
    def __init__(self, config):
        assert config['model_type'] == 'full_unet'
        torch.set_float32_matmul_precision('medium')
        self.run_name = config['name']
        self.data_loading_config = config['data_loading']
        self.model_loading_config = config['model_loading']
        self.training_config = config['training']
        self.logging_config = config['logging']
        
        if self.logging_config['enabled'] == True:
            wandb.init(project="Full-3D-UNet", name=self.run_name, config=config)
        
        self.task = MSDTask.fromStr(self.data_loading_config['task'])
        split_ratio = (
            self.data_loading_config['split_ratio']['train'],
            self.data_loading_config['split_ratio']['val'],
            self.data_loading_config['split_ratio']['test']
        )
        self.train_dataset = MSDDataset(
            msd_task=self.task,
            split=Split.TRAIN,
            split_ratio=split_ratio,
            seed=self.data_loading_config['seed'],
            normalize=True
        )
        self.val_dataset = MSDDataset(
            msd_task=self.task,
            split=Split.VAL,
            split_ratio=split_ratio,
            seed=self.data_loading_config['seed'],
            normalize=True
        )
        self.test_dataset = MSDDataset(
            msd_task=self.task,
            split=Split.Test,
            split_ratio=split_ratio,
            seed=self.data_loading_config['seed'],
            normalize=True
        )
        self.train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=1,
            num_workers=self.data_loading_config['n_workers'],
            shuffle=False,
        )
        self.val_loader = DataLoader(
            dataset=self.val_dataset,
            batch_size=1,
            num_workers=self.data_loading_config['n_workers'],
            shuffle=False,
        )
        self.model = self.initModel()
        self.loss_fn = DiceLoss(softmax=True, include_background=False)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.training_config['learning_rate'])
        self.scalar = torch.cuda.amp.GradScaler()
        self.epochs = self.training_config['n_steps'] // len(self.train_dataset)

    def initModel(self):
        model = UNet3d(
            in_channels=self.train_dataset.chanels[0],
            out_channels=self.train_dataset.chanels[1]
        ).to(DEVICE)
        
        if self.model_loading_config['enabled']:
            try:
                model.load_state_dict(torch.load(self.model_loading_config['path']))
                print(f'Loaded checkpoint from {self.model_loading_config["path"]}')
            except Exception as e:
                print(f'Unable to load checkpoint. {e}')
        
        return model
    
    def log_prediction(self):
        if not self.logging_config['enabled']:
            return
        with torch.cuda.amp.autocast():
            wandb_images = []
            for i in range(self.logging_config['prediction_log_count']):
                scan, ground_trouth = self.test_dataset[i]
                pred = self.model(scan.unsqueeze(0).to(DEVICE))[0].cpu()

                class_labels = self.test_dataset.class_labels

                ground_trouth = ground_trouth.argmax(dim=0)
                pred = pred.argmax(dim=0)
                scan = scan[self.logging_config['modality']] if self.logging_config['modality'] < self.train_dataset.chanels[0] else scan[0]
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
    
    def train_fn(self, loop: tqdm.tqdm, split: Split):
        loader = self.train_loader if split == Split.TRAIN else self.val_loader
        start_time = time.time()
        sum_loss = 0
        loop_len = len(loader)
        for batch_idx, (data, targets) in enumerate(loop):
            data = data.to(device=DEVICE)
            targets = targets.to(device=DEVICE)
            
            # forward
            with torch.cuda.amp.autocast():
                predictions = self.model(data)
                loss = self.loss_fn(predictions, targets)
            
            # backward
            if split == Split.TRAIN:
                self.optimizer.zero_grad()
                self.scalar.scale(loss).backward()
                self.scalar.step(self.optimizer)
                self.scalar.update()
            
            # update tqdm loop
            loop.set_postfix(loss=loss.item())
            sum_loss += loss.item()
        end_time = time.time()
        epoch_time = end_time - start_time
        if self.logging_config['enabled']:
            wandb.log({f"{split.value}_epoch_time": epoch_time})
            wandb.log({f"{split.value}_epoch_loss": sum_loss / loop_len})
    
    def save_model(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
    
    def fit(self):
        self.log_prediction()
        for epoch in range(self.epochs):
            for split in [Split.TRAIN, Split.VAL]:
                loop = tqdm.tqdm(
                    self.train_loader if split == Split.TRAIN else self.val_loader,
                    desc=f"{split.value}-Epoch {epoch + 1}/{self.epochs}"
                )
                self.train_fn(loop, split)
            self.save_model(f'checkpoints/{self.run_name}/full-unet/epoch_{epoch}.pth')
            self.log_prediction()
        self.save_model(f'final/full-unet/{self.run_name}_unet_weights.pth')