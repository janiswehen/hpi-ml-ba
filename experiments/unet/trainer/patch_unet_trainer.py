import torch
import wandb
import tqdm
import time
import os

from torch.utils.data import DataLoader
from monai.losses import DiceLoss
from math import floor

from unet.dataset.msd_dataset import Split, MSDTask, MSDDataset
from unet.dataset.sliced_dataset import SlicedDataset
from unet.model.unet import UNet3d

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class PatchUnetTrainer():
    def __init__(self, config):
        assert config['model_type'] == 'patch_unet'
        torch.set_float32_matmul_precision('medium')
        torch.autograd.set_detect_anomaly(True)
        self.run_name = config['name']
        self.data_loading_config = config['data_loading']
        self.model_loading_config = config['model_loading']
        self.training_config = config['training']
        self.logging_config = config['logging']
        
        if self.logging_config['enabled'] == True:
            wandb.init(project="Patch-3D-UNet", name=self.run_name, config=config)
        
        self.task = MSDTask.fromStr(self.data_loading_config['task'])
        split_ratio = (
            self.data_loading_config['split_ratio']['train'],
            self.data_loading_config['split_ratio']['val'],
            self.data_loading_config['split_ratio']['test']
        )
        org_train_dataset = MSDDataset(
            msd_task=self.task,
            split=Split.TRAIN,
            split_ratio=split_ratio,
            seed=self.data_loading_config['seed'],
            normalize=True
        )
        org_val_dataset = MSDDataset(
            msd_task=self.task,
            split=Split.VAL,
            split_ratio=split_ratio,
            seed=self.data_loading_config['seed'],
            normalize=True
        )
        org_test_dataset = MSDDataset(
            msd_task=self.task,
            split=Split.Test,
            split_ratio=split_ratio,
            seed=self.data_loading_config['seed'],
            normalize=True
        )
        self.train_dataset = SlicedDataset(
            dataset=org_train_dataset,
            patch_size=self.data_loading_config['patch_size'],
            slice_axis=self.data_loading_config['slice_axis'],
        )
        self.val_dataset = SlicedDataset(
            dataset=org_val_dataset,
            patch_size=self.data_loading_config['patch_size'],
            slice_axis=self.data_loading_config['slice_axis'],
        )
        self.test_dataset = SlicedDataset(
            dataset=org_test_dataset,
            patch_size=self.data_loading_config['patch_size'],
            slice_axis=self.data_loading_config['slice_axis'],
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
        self.epochs = self.training_config['n_steps'] // (len(self.train_dataset) * self.data_loading_config['mean_slice_count'])

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
                pred_ps = []
                for i in range(scan.shape[0]):
                    scan_p = scan[i,...].unsqueeze(0).to(DEVICE)
                    pred_p = self.model(scan_p)
                    scan_p = scan_p.detach().cpu()
                    pred_p = pred_p.detach().cpu()
                    pred_ps.append(pred_p[0])
                pred = torch.stack(pred_ps)
                pred = self.test_dataset.get_original(pred)
                pred = pred.argmax(dim=0)
                ground_trouth = self.test_dataset.get_original(ground_trouth)
                ground_trouth = ground_trouth.argmax(dim=0)
                scan = self.test_dataset.get_original(scan)[self.logging_config['modality']]
                
                class_labels = self.test_dataset.class_labels
                slice_idx = floor(scan.shape[self.logging_config['slice_axis']] * self.logging_config['rel_slice'])
                wandb_images.append(wandb.Image(
                    torch.select(scan, self.logging_config['slice_axis'] , slice_idx),
                    caption=f"Scan {i}",
                    masks={
                        "prediction": {
                            "mask_data": torch.select(pred, self.logging_config['slice_axis'] , slice_idx).numpy(),
                            "class_labels": class_labels,
                        },
                        "ground-trouth": {
                            "mask_data": torch.select(ground_trouth, self.logging_config['slice_axis'] , slice_idx).numpy(),
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
            losses = []
            for i in range(data.shape[1]):
                data_patch = data[:,i,...].to(device=DEVICE)
                targets_patch = targets[:,i,...].to(device=DEVICE)
                
                # forward
                with torch.cuda.amp.autocast():
                    predictions_patch = self.model(data_patch)
                    loss = self.loss_fn(predictions_patch, targets_patch)
                    data_patch = data_patch.detach().cpu()
                    predictions_patch = predictions_patch.detach().cpu()
                    targets_patch = targets_patch.detach().cpu()
                if loss.item() > 0.999:
                    continue
                losses.append(loss.item())
                
                # backward
                if split == Split.TRAIN:
                    self.optimizer.zero_grad()
                    self.scalar.scale(loss).backward()
                    self.scalar.step(self.optimizer)
                    self.scalar.update()
            
            # update tqdm loop
            loss = sum(losses) / len(losses)
            loop.set_postfix(loss=loss)
            sum_loss += loss
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
            self.save_model(f'checkpoints/{self.run_name}/patch-unet/epoch_{epoch}.pth')
            self.log_prediction()
        self.save_model(f'final/patch_unet/{self.run_name}_unet_weights.pth')