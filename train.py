import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from torch.utils.data import DataLoader
import yaml
import os

from unet_3d import UNet3d
from dataset import BratsDataset

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def train_fn(loader: DataLoader, model: nn.Module, optimizer: optim.Optimizer, loss_fn, scalar: torch.cuda.amp.GradScaler):
    loop = tqdm.tqdm(loader)
    
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

def try_load_checkpoint(model: nn.Module, config):
    try:
        #check if config has checkpoint path
        if config['checkpoint_path'] is None:
            return
        model.load_state_dict(torch.load(config['checkpoint_path']))
        print('Loaded checkpoint from {}'.format(config['checkpoint_path']))
    except Exception as e:
        print('Unable to load checkpoint. {}'.format(e))

def main():
    config = {}
    with open("config.yaml", "r") as stream:
        config = yaml.safe_load(stream)
    run_name = config['run_name']

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

    model = UNet3d(in_channels=4, out_channels=3).to(DEVICE)
    try_load_checkpoint(model, config)

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    scalar = torch.cuda.amp.GradScaler()

    for epoch in range(config['n_epochs']):
        train_fn(loader, model, optimizer, loss_fn, scalar)
        if epoch % 10 == 9:
            torch.save(model.state_dict(), f'checkpoints/{run_name}/epoch_{epoch}.pth')
    torch.save(model.state_dict(), f'final/{run_name}_unet3d_weights.pth')

if __name__ == '__main__':
    main()
