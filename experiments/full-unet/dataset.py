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
        self.data = self.data_json['training']
        self.class_labels = {
            1: self.data_json['labels']['1'],
            2: self.data_json['labels']['2'],
            3: self.data_json['labels']['3']
        }
    
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

        return img, label

    def one_hot_encode(self, label):
        one_hot = np.zeros((4, *label.shape[1:]), dtype=np.float32)
        one_hot[0] = label[0] == 0
        one_hot[1] = label[0] >= 1
        one_hot[2] = label[0] >= 2
        one_hot[3] = label[0] >= 3
        return one_hot

if __name__ == '__main__':
    data_dir = '/dhc/home/janis.wehen/data/Task01_BrainTumour/'
    dataset = BratsDataset(data_dir)
    print(len(dataset))
    scan, label = dataset[0]
    print(np.min(scan), np.max(scan))
    print(np.min(label), np.max(label))
    print(dataset[0][0].shape, dataset[0][1].shape)