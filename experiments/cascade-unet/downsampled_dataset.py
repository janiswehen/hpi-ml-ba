import torch.utils.data as data
import torch.nn.functional as F
from dataset import BratsDataset
import torch

class DownsampledDataset(data.Dataset):
    def __init__(self, dataset: data.Dataset, scale_factor=3):
        super().__init__()
        self.dataset = dataset
        self.class_labels = dataset.class_labels
        self.org_shape = dataset[0][0].shape[-3:]
        self.down_shape = (self.org_shape[0] // scale_factor, self.org_shape[1] // scale_factor, self.org_shape[2] // scale_factor)
        self.up = torch.nn.Upsample(size=self.org_shape, mode='trilinear', align_corners=True)
        self.down = torch.nn.Upsample(size=self.down_shape, mode='trilinear', align_corners=True)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        return self.down(self.dataset[index][0].unsqueeze(0))[0], self.down(self.dataset[index][1].unsqueeze(0))[0]

if __name__ == '__main__':
    data_dir = '/dhc/home/janis.wehen/data/Task01_BrainTumour/'
    dataset = BratsDataset(data_dir)
    patch_dataset = DownsampledDataset(dataset)
    x = patch_dataset[0][0].unsqueeze(0)
    print(x.shape)
    x_down = patch_dataset.down(x)
    print(x_down.shape)
    x_recon = patch_dataset.up(x_down)
    print(x_recon.shape)