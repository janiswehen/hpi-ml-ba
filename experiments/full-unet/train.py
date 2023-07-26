import os
import time
from datetime import timedelta
import numpy as np

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import yaml
import wandb

from unet_3d import UNet3d
from dataset import BratsDataset
from monai.losses import DiceLoss

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def train_fn(loader: DataLoader, model: nn.Module, optimizer: optim.Optimizer, loss_fn, scalar: torch.cuda.amp.GradScaler, loop: tqdm.tqdm, log_wandb: bool):
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.half().to(device=DEVICE)
        targets = targets.half().to(device=DEVICE)
        
        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)
        
        # backward
        optimizer.zero_grad()
        scalar.scale(loss).backward()
        scalar.step(optimizer)
        scalar.update()
        
        # update tqdm loop
        loop.set_postfix(loss=loss.item())
        if log_wandb:
            wandb.log({"dice-loss": loss.item()})

def try_load_checkpoint(model: nn.Module, checkpoint_path: str):
    try:
        model.load_state_dict(torch.load(checkpoint_path))
        print('Loaded checkpoint from {}'.format(checkpoint_path))
    except Exception as e:
        print('Unable to load checkpoint. {}'.format(e))

def log_prediction(model: nn.Module, dataset: Dataset, slice=90, n_predictions=5):
    with torch.cuda.amp.autocast():
        wandb_images = []
        for i in range(n_predictions):
            scan, ground_trouth = dataset[i]
            pred = model(torch.tensor(scan).half().unsqueeze(0).to(DEVICE))[0]
            
            class_labels = dataset.class_labels
            
            ground_trouth = torch.tensor(ground_trouth).argmax(dim=0)
            pred = pred.argmax(dim=0)
            wandb_images.append(wandb.Image(
                scan[0, :, :, slice],
                caption=f"Scan {i}",
                masks={
                    "prediction": {
                        "mask_data": pred[:, :, slice].cpu().numpy(),
                        "class_labels": class_labels,
                    },
                    "ground-trouth": {
                        "mask_data": ground_trouth[:, :, slice].cpu().numpy(),
                        "class_labels": class_labels,
                    },
                }
            ))
        wandb.log({"predictions": wandb_images})

def main():
    config = {}
    with open("config.yaml", "r") as stream:
        config = yaml.safe_load(stream)['run']
    run_name = config['name']
    if config['logging']['enabled'] == True:
        wandb.init(project="Full-3D-UNet", name=run_name, config=config)

    # check if weights folders exists
    if os.path.isdir(f'final') == False:
        os.mkdir(f'final')
    if os.path.isdir(f'checkpoints') == False:
        os.mkdir(f'checkpoints')
    if os.path.isdir(f'checkpoints/{run_name}') == False:
        os.mkdir(f'checkpoints/{run_name}')

    dataset = BratsDataset(config['data_loading']['path'])

    loader = DataLoader(
        dataset,
        batch_size=config['data_loading']['batch_size'],
        num_workers=config['data_loading']['n_workers'],
        shuffle=True,
    )

    model = UNet3d(in_channels=4, out_channels=4).to(DEVICE)
    if config['logging']['enabled']:
        wandb.watch(model, log="all")
    if config['model_loading']['enabled']:
        try_load_checkpoint(model, config['model_loading']['path'])

    loss_fn = DiceLoss(softmax=True, include_background=False)
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])

    scalar = torch.cuda.amp.GradScaler()


    if config['logging']['enabled']:
        log_prediction(model, dataset, n_predictions=config['logging']['prediction_log_count'])
    for epoch in range(config['training']['n_epochs']):
        start_time = time.time()
        loop = tqdm.tqdm(loader, desc=f"Epoch {epoch + 1}/{config['training']['n_epochs']}")
        train_fn(loader, model, optimizer, loss_fn, scalar, loop, config['logging']['enabled'])
        end_time = time.time()
        elapsed_time = end_time - start_time
        torch.save(model.state_dict(), f'checkpoints/{run_name}/epoch_{epoch}.pth')
        if config['logging']['enabled']:
            log_prediction(model, dataset, n_predictions=config['logging']['prediction_log_count'])
            wandb.log({"epoch_time_seconds": elapsed_time})
            wandb.log({"epoch_time": str(timedelta(seconds=elapsed_time))})

    torch.save(model.state_dict(), f'final/{run_name}_unet3d_weights.pth')

if __name__ == '__main__':
    main()
