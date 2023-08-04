import torch.utils.data as data
import torch.nn.functional as F
import torch

from unet.dataset.msd_dataset import MSDDataset

class CombinedDataset(data.Dataset):
    def __init__(self, dataset1: data.Dataset, dataset2: data.Dataset, rule=lambda data1, data2: (torch.cat((data1[1], data2[0]), dim=0), data2[1])):
        super().__init__()
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.rule = rule
        assert len(dataset1) == len(dataset2)

    def __len__(self):
        return len(self.dataset1)
    
    def __getitem__(self, index):
        return self.rule(self.dataset1[index], self.dataset2[index])

if __name__ == '__main__':
    dataset1 = MSDDataset()
    dataset2 = MSDDataset()
    com_dataset = CombinedDataset(dataset1, dataset2)
    x, y = com_dataset[0]
    print(x.shape, y.shape)