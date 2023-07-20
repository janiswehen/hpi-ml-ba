import os
import torch.utils.data as data
import nibabel as nib
import numpy as np
import json

class BratsDataset(data.Dataset):
    def __init__(self, dataset_dir, transform=None):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.transform = transform
        
        with open(os.path.join(self.dataset_dir, 'dataset.json')) as f:
            self.data_json = json.load(f)
        self.data = []
        for index in range(len(self.data_json['training'])):
            image_path = os.path.join(self.dataset_dir, self.data_json['training'][index]['image'])
            img = nib.load(image_path).get_fdata()
            img.astype(np.float32)
            img = np.moveaxis(img, -1, 0) # convert from (W, H, D, M) to (M, W, H, D)
            
            label_path = os.path.join(self.dataset_dir, self.data_json['training'][index]['label'])
            label = nib.load(label_path).get_fdata()
            label.astype(np.float32)
            label = np.expand_dims(label, axis=0) # convert from (W, H, D) to (1, W, H, D)
            
            self.data.append((img, label))
            if index == len(self.data_json['training']) - 1:
                print(f'Dataset loading: {index+1}/{len(self.data_json["training"])}')
            else:
                print(f'Dataset loading: {index+1}/{len(self.data_json["training"])}', end='\r')
    
    def __len__(self):
        return len(self.data_json['training'])
    
    def __getitem__(self, index):
        scan, label = self.data[index]
        
        mean = np.mean(scan)
        std = np.std(scan)
        scan = (scan - mean) / std
        
        label = self.one_hot_encode(label)
        
        return scan, label

    def one_hot_encode(self, label):
        classes = np.array([1, 2, 3])
        one_hot = (classes == label[0,...,None]).astype(np.float32)
        one_hot = np.moveaxis(one_hot, -1, 0)
        return one_hot

if __name__ == '__main__':
    data_dir = '/dhc/home/janis.wehen/data/Task01_BrainTumour/'
    dataset = BratsDataset(data_dir)
    print(len(dataset))
    scan, label = dataset[0]
    print(np.min(scan), np.max(scan))
    print(np.min(label), np.max(label))