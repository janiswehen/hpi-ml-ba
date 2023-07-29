import os
import time

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from torch.utils.data import DataLoader
from torch.nn.modules.loss import _Loss
import yaml
import wandb

from unet_3d import UNet3d
from dataset import BratsDataset, Split
from patched_dataset import PatchDataset
from monai.losses import DiceLoss

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def train_fn(loader: DataLoader, model: nn.Module, optimizer: optim.Optimizer, loss_fn: _Loss, scalar: torch.cuda.amp.GradScaler, loop: tqdm.tqdm, log_wandb: bool, split: Split):
    start_time = time.time()
    sum_loss = 0
    loop_len = len(loader)
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.to(device=DEVICE)
        
        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward
        if split == Split.TRAIN:
            optimizer.zero_grad()
            scalar.scale(loss).backward()
            scalar.step(optimizer)
            scalar.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())
        sum_loss += loss.item()
    end_time = time.time()
    epoch_time = end_time - start_time
    if log_wandb:
        wandb.log({f"{split.value}_epoch_time": epoch_time})
        wandb.log({f"{split.value}_epoch_loss": sum_loss / loop_len})

def try_load_checkpoint(model: nn.Module, checkpoint_path: str):
    try:
        model.load_state_dict(torch.load(checkpoint_path))
        print('Loaded checkpoint from {}'.format(checkpoint_path))
    except Exception as e:
        print('Unable to load checkpoint. {}'.format(e))

def log_prediction(model: nn.Module, dataset: PatchDataset, config: dict):
    with torch.cuda.amp.autocast():
        wandb_images = []
        for i in range(config['prediction_log_count']):
            scan_p = []
            ground_trouth_p = []
            pred_p = []
            for patch_idx in range(dataset.patch_count):
                idx = i * dataset.patch_count + patch_idx
                scan, ground_trouth = dataset[idx]
                pred = model(scan.unsqueeze(0).to(DEVICE))[0]
                
                class_labels = dataset.class_labels
                
                scan = scan.cpu()
                ground_trouth = ground_trouth.argmax(dim=0).unsqueeze(0).cpu()
                pred = pred.argmax(dim=0).unsqueeze(0).cpu()
                scan_p.append(scan)
                ground_trouth_p.append(ground_trouth)
                pred_p.append(pred)
            
            scan_p = dataset.get_original(torch.stack(scan_p))[config['modality']]
            ground_trouth_p = dataset.get_original(torch.stack(ground_trouth_p))[0]
            pred_p = dataset.get_original(torch.stack(pred_p))[0]
            
            wandb_images.append(wandb.Image(
                torch.select(scan_p, config['slice_axis'] , config['slice']),
                caption=f"Scan {i}",
                masks={
                    "prediction": {
                        "mask_data": torch.select(pred_p, config['slice_axis'] , config['slice']).numpy(),
                        "class_labels": class_labels,
                    },
                    "ground-trouth": {
                        "mask_data": torch.select(ground_trouth_p, config['slice_axis'] , config['slice']).numpy(),
                        "class_labels": class_labels,
                    },
                }
            ))
        wandb.log({"predictions": wandb_images})

def main():
    torch.set_float32_matmul_precision('medium')
    config = {}
    with open("config.yaml", "r") as stream:
        config = yaml.safe_load(stream)['run']
    run_name = config['name']
    if config['logging']['enabled'] == True:
        wandb.init(project="Patch-3D-UNet", name=run_name, config=config)

    # check if weights folders exists
    if os.path.isdir(f'final') == False:
        os.mkdir(f'final')
    if os.path.isdir(f'checkpoints') == False:
        os.mkdir(f'checkpoints')
    if os.path.isdir(f'checkpoints/{run_name}') == False:
        os.mkdir(f'checkpoints/{run_name}')

    train_dataset = BratsDataset(
        config['data_loading']['path'],
        Split.TRAIN,
        config['data_loading']['split_ratio'],
        config['data_loading']['seed']
    )
    val_dataset = BratsDataset(
        config['data_loading']['path'],
        Split.VAL,
        config['data_loading']['split_ratio'],
        config['data_loading']['seed']
    )
    train_patch_dataset = PatchDataset(
        train_dataset,
        patch_size=config['data_loading']['patch_size'],
    )
    val_patch_dataset = PatchDataset(
        val_dataset,
        patch_size=config['data_loading']['patch_size'],
    )

    train_loader = DataLoader(
        train_patch_dataset,
        batch_size=config['data_loading']['batch_size'],
        num_workers=config['data_loading']['n_workers'],
        shuffle=False,
    )
    
    val_loader = DataLoader(
        val_patch_dataset,
        batch_size=config['data_loading']['batch_size'],
        num_workers=config['data_loading']['n_workers'],
        shuffle=False,
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
        log_prediction(model, val_patch_dataset, config=config['logging'])
    for epoch in range(config['training']['n_epochs']):
        for split in [Split.TRAIN, Split.VAL]:
            loader = train_loader if split == Split.TRAIN else val_loader
            loop = tqdm.tqdm(loader, desc=f"{split.value}-Epoch {epoch + 1}/{config['training']['n_epochs']}")
            train_fn(loader, model, optimizer, loss_fn, scalar, loop, config['logging']['enabled'], split)
        torch.save(model.state_dict(), f'checkpoints/{run_name}/epoch_{epoch}.pth')
        if config['logging']['enabled']:
            log_prediction(model, val_patch_dataset, config=config['logging'])

    torch.save(model.state_dict(), f'final/{run_name}_unet3d_weights.pth')

if __name__ == '__main__':
    main()
