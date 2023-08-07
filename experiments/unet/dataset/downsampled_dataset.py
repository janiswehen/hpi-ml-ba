import torch.utils.data as data
import torch.nn.functional as F
import torch

from unet.dataset.msd_dataset import Split, MSDDataset, MSDTask

class DownsampledDataset(data.Dataset):
    def __init__(self, dataset: data.Dataset, scale_factor=3, rescale=False, normalize=False):
        super().__init__()
        self.normalize = normalize
        self.dataset = dataset
        self.rescale = rescale
        self.modalitys = dataset.modalitys
        self.class_labels = dataset.class_labels
        self.chanels = dataset.chanels
        self.org_shape = dataset[0][0].shape[-3:]
        self.scaled_shape = (self.org_shape[0] // scale_factor, self.org_shape[1] // scale_factor, self.org_shape[2] // scale_factor)
        self.up = torch.nn.Upsample(size=self.org_shape, mode='trilinear', align_corners=True)
        self.down = torch.nn.Upsample(size=self.scaled_shape, mode='trilinear', align_corners=True)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        if self.rescale:
            return self.up(self.down(self.dataset[index][0].unsqueeze(0)))[0], self.up(self.down(self.dataset[index][1].unsqueeze(0)))[0]
        img, label = self.down(self.dataset[index][0].unsqueeze(0))[0], self.down(self.dataset[index][1].unsqueeze(0))[0]
        
        if self.normalize:
            img_mean = img.mean(dim=(1,2,3), keepdim=True)
            img_std = img.std(dim=(1,2,3), keepdim=True)
            img = (img - img_mean) / img_std
        
        return img, label

if __name__ == '__main__':
    dataset = MSDDataset()
    patch_dataset = DownsampledDataset(dataset)
    x = patch_dataset[0][0].unsqueeze(0)
    print(x.shape)
    x_down = patch_dataset.down(x)
    print(x_down.shape)
    x_recon = patch_dataset.up(x_down)
    print(x_recon.shape)