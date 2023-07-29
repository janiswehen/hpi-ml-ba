import os
import torch.utils.data as data
import nibabel as nib
import numpy as np
import json
import torch
from enum import Enum
import random

class Split(Enum):
    TRAIN = 'Train'
    VAL = 'Val'

class BratsDataset(data.Dataset):
    def __init__(self, dataset_dir, split=Split.TRAIN, split_ratio=0.8, seed=42):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.split = split
        
        with open(os.path.join(self.dataset_dir, 'dataset.json')) as f:
            self.data_json = json.load(f)
        self.data = self.data_json['training']
        self.class_labels = {
            1: self.data_json['labels']['1'],
            2: self.data_json['labels']['2'],
            3: self.data_json['labels']['3']
        }
        random.seed(seed)
        random.shuffle(self.data)
        self.data = self.data[:int(len(self.data) * split_ratio)] if self.split == Split.TRAIN else self.data[int(len(self.data) * split_ratio):]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        image_path = os.path.join(self.dataset_dir, self.data[index]['image'])
        img = nib.load(image_path).get_fdata()
        img.astype(np.float32)
        img = np.moveaxis(img, -1, 0)  # convert from (W, H, D, M) to (M, W, H, D)

        label_path = os.path.join(self.dataset_dir, self.data[index]['label'])
        label = nib.load(label_path).get_fdata()
        label.astype(np.float32)
        label = np.expand_dims(label, axis=0)  # convert from (W, H, D) to (1, W, H, D)

        mean = np.mean(img)
        std = np.std(img)
        img = (img - mean) / std

        label = self.one_hot_encode(label)
        img = img.astype(np.float32)

        return torch.from_numpy(img), torch.from_numpy(label)

    def one_hot_encode(self, label):
        classes = np.array([0, 1, 2, 3])
        one_hot = (classes == label[0,...,None]).astype(np.float32)
        one_hot = np.moveaxis(one_hot, -1, 0)
        return one_hot

if __name__ == '__main__':
    data_dir = '/dhc/home/janis.wehen/data/Task01_BrainTumour/'
    dataset = BratsDataset(data_dir, Split.VAL)
    print(len(dataset))
    print(dataset[0][0].shape, dataset[0][1].shape)