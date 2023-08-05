import os
import torch.utils.data as data
import nibabel as nib
import numpy as np
import json
import torch
import torch.nn.functional as F
from enum import Enum
import random

BASE_DATA_DIR = '/dhc/home/janis.wehen/data/MSD/'

class Split(Enum):
    TRAIN = 'Train'
    VAL = 'Val'

class MSDTask(Enum):
    TASK01 = ('Task01', 'BrainTumour')
    TASK02 = ('Task02', 'Heart')
    TASK03 = ('Task03', 'Liver')
    TASK04 = ('Task04', 'Hippocampus')
    TASK05 = ('Task05', 'Prostate')
    TASK06 = ('Task06', 'Lung')
    TASK07 = ('Task07', 'Pancreas')
    TASK08 = ('Task08', 'HepaticVessel')
    TASK09 = ('Task09', 'Spleen')
    TASK10 = ('Task10', 'Colon')
    
    def fromStr(string: str):
        for task in MSDTask:
            if string in task.value:
                return task
        raise ValueError(f'No MSDTask with string {string} found.')

class MSDDataset(data.Dataset):
    def __init__(self, msd_task=MSDTask.TASK01, split=Split.TRAIN, split_ratio=0.8, seed=42):
        super().__init__()
        self.dataset_dir = os.path.join(BASE_DATA_DIR, f'{msd_task.value[0]}_{msd_task.value[1]}')
        self.split = split
        
        with open(os.path.join(self.dataset_dir, 'dataset.json')) as f:
            self.data_json = json.load(f)
        self.data = self.data_json['training']
        self.modalitys = {
            int(key): self.data_json['modality'][key] for key in self.data_json['modality'].keys()
        }
        self.class_labels = {
            int(key): self.data_json['labels'][key] for key in self.data_json['labels'].keys() if key != '0'
        }
        self.is_multy_modal = self.data_json['tensorImageSize'] == '4D'
        
        if seed is not None:
            random.seed(seed)
            random.shuffle(self.data)
        self.data = self.data[:int(len(self.data) * split_ratio)] if self.split == Split.TRAIN else self.data[int(len(self.data) * split_ratio):]
        
        self.chanels = self[0][0].shape[0], self[0][1].shape[0]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        image_path = os.path.join(self.dataset_dir, self.data[index]['image'])
        img = nib.load(image_path).get_fdata()
        img.astype(np.float32)
        img = np.moveaxis(img, -1, 0) if self.is_multy_modal else np.expand_dims(img, axis=0)

        label_path = os.path.join(self.dataset_dir, self.data[index]['label'])
        label = nib.load(label_path).get_fdata()
        label.astype(np.float32)
        label = np.expand_dims(label, axis=0)

        label = self.one_hot_encode(label)
        img = img.astype(np.float32)

        return self.pad_tensor(torch.from_numpy(img)), self.pad_tensor(torch.from_numpy(label))

    def one_hot_encode(self, label):
        classes = np.array([0] + [i for i in self.class_labels.keys()])
        one_hot = (classes == label[0,...,None]).astype(np.float32)
        one_hot = np.moveaxis(one_hot, -1, 0)
        return one_hot
    
    def pad_tensor(self, tensor):
        if tensor.shape[-1] >= 16 and tensor.shape[-2] >= 16 and tensor.shape[-3] >= 16:
            return tensor
        
        padding = []
        for i in range(-1, -4, -1):
            if tensor.shape[i] < 16:
                padding.extend([0, 16 - tensor.shape[i]]) # pad the last dimension
            else:
                padding.extend([0, 0])  # no padding needed for this dimension
        tensor = F.pad(tensor, padding, "constant", 0)
        return tensor


if __name__ == '__main__':
    for task in MSDTask:
        dataset = MSDDataset(msd_task=task, split=Split.TRAIN, split_ratio=1, seed=None)
        print('----------------------------------------------')
        print(f'{task.value[0]}-{task.value[1]}-Dataset')
        print(f'   length: {len(dataset)}')
        print(f'   modalitys: {dataset.modalitys.values()}')
        print(f'   class labels: {dataset.class_labels.values()}')
        print(f'   is multy modal: {dataset.is_multy_modal}')
        print(f'   first scan shape: {dataset[0][0].shape}')
        print(f'   first label shape: {dataset[0][1].shape}')
        # print(f'   test for anomalys:')
        # print(f'   ?/{len(dataset)}', end='\r')
        # shapes_img = []
        # shapes_label = []
        # for i in range(len(dataset)):
        #     print(f'   {i}/{len(dataset)}', end='\r')
        #     img, label = dataset[i]
        #     shapes_img.append(img.shape)
        #     shapes_label.append(label.shape)
        # print(f'   {len(dataset)}/{len(dataset)}')
        # min_shape_img = [min([shape[d] for shape in shapes_img]) for d in range(-1, -4, -1)]
        # max_shape_img = [max([shape[d] for shape in shapes_img]) for d in range(-1, -4, -1)]
        # min_shape_label = [min([shape[d] for shape in shapes_label]) for d in range(-1, -4, -1)]
        # max_shape_label = [max([shape[d] for shape in shapes_label]) for d in range(-1, -4, -1)]
        # print(f'   min img shape: {min_shape_img}')
        # print(f'   max img shape: {max_shape_img}')
        # print(f'   min label shape: {min_shape_label}')
        # print(f'   max label shape: {max_shape_label}')