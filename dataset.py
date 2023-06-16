import os
import torch.utils.data as data
import nibabel as nib
import numpy as np
import json

data_dir = '/dhc/home/janis.wehen/data/Task01_BrainTumour/'

class BratsDataset(data.Dataset):
    def __init__(self, dataset_dir, transform=None):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.transform = transform
        
        with open(os.path.join(self.dataset_dir, 'dataset.json')) as f:
            self.data_json = json.load(f)
    
    def __len__(self):
        return len(self.data_json['training'])
    
    def __getitem__(self, index):
        image_path = os.path.join(self.dataset_dir, self.data_json['training'][index]['image'])
        img = nib.load(image_path).get_fdata()
        img = np.moveaxis(img, -1, 0) # convert from (W, H, D, M) to (M, W, H, D)
        
        label_path = os.path.join(self.dataset_dir, self.data_json['training'][index]['label'])
        label = nib.load(label_path).get_fdata()
        label = np.expand_dims(label, axis=0) # convert from (W, H, D) to (1, W, H, D)
        
        return img, label

if __name__ == '__main__':
    dataset = BratsDataset(data_dir)
    print(len(dataset))
    print(dataset[0])