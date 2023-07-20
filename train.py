import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import yaml
import os
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

def try_load_checkpoint(model: nn.Module, config):
    try:
        #check if config has checkpoint path
        if config['checkpoint_path'] is None:
            return
        model.load_state_dict(torch.load(config['checkpoint_path']))
        print('Loaded checkpoint from {}'.format(config['checkpoint_path']))
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
        config = yaml.safe_load(stream)
    run_name = config['run_name']
    config['device'] = DEVICE
    if config['wandb'] == True:
        wandb.init(project="Full-3D-UNet", name=run_name, config=config)

    # check if weights folders exists
    if os.path.isdir(f'final') == False:
        os.mkdir(f'final')
    if os.path.isdir(f'checkpoints') == False:
        os.mkdir(f'checkpoints')
    if os.path.isdir(f'checkpoints/{run_name}') == False:
        os.mkdir(f'checkpoints/{run_name}')

    dataset = BratsDataset(config['dataset_dir'])

    loader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        num_workers=config['n_workers'],
        shuffle=True,
    )

    model = UNet3d(in_channels=4, out_channels=4).to(DEVICE)
    if config['wandb']:
        wandb.watch(model, log="all")
    try_load_checkpoint(model, config)

    loss_fn = DiceLoss(softmax=True, include_background=False)
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    scalar = torch.cuda.amp.GradScaler()


    if config['wandb']:
        log_prediction(model, dataset, n_predictions=config['wandb_prediction_log_count'])
    for epoch in range(config['n_epochs']):
        loop = tqdm.tqdm(loader, desc=f"Epoch {epoch + 1}/{config['n_epochs']}")
        train_fn(loader, model, optimizer, loss_fn, scalar, loop, config['wandb'])
        if epoch % 5 == 4:
            torch.save(model.state_dict(), f'checkpoints/{run_name}/epoch_{epoch}.pth')
        if config['wandb']:
            log_prediction(model, dataset, n_predictions=config['wandb_prediction_log_count'])

    torch.save(model.state_dict(), f'final/{run_name}_unet3d_weights.pth')

if __name__ == '__main__':
    main()
