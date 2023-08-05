import torch.utils.data as data
import torch.nn.functional as F
import torch

from unet.dataset.msd_dataset import Split, MSDDataset, MSDTask

class NormalizedDataset(data.Dataset):
    def __init__(self, dataset: data.Dataset):
        super().__init__()
        self.dataset = dataset
        self.chanels = dataset.chanels
        self.class_labels = dataset.class_labels
        self.modalitys = dataset.modalitys

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        img, label = self.dataset[index]
        
        img_mean = img.mean(dim=(1,2,3), keepdim=True)
        img_std = img.std(dim=(1,2,3), keepdim=True)
        img = (img - img_mean) / img_std
        
        return img, label