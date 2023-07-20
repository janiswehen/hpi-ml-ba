import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from torch.utils.data import DataLoader
import yaml

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

def main():
    config = {}
    with open("config.yaml", "r") as stream:
        config = yaml.safe_load(stream)

    dataset = BratsDataset(config['dataset_dir'])

    loader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        num_workers=config['n_workers'],
        shuffle=True,
    )

    model = UNet3d(in_channels=4, out_channels=3).to(DEVICE)
    
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    scalar = torch.cuda.amp.GradScaler()

    for epoch in range(config['n_epochs']):
        train_fn(loader, model, optimizer, loss_fn, scalar)

if __name__ == '__main__':
    main()
