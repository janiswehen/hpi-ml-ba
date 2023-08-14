import torch.utils.data as data
import torch.nn.functional as F
import torch

from unet.dataset.msd_dataset import MSDDataset

concat_rule = {
    "data": lambda data1, data2: (torch.cat((data1[0], data2[1]), dim=0), data1[1]),
    "labels": lambda label1, label2: label1,
    "modalitys": lambda modalitys1, modalitys2: { key: value for key, value in enumerate(list(modalitys1.values()) + list(modalitys2.values()))},
    "chanels": lambda chanel1, chanel2: (chanel1[0] + chanel2[1], chanel1[1])
}

class CombinedDataset(data.Dataset):
    def __init__(self, dataset1: data.Dataset, dataset2: data.Dataset, rule=concat_rule, normalize=False):
        super().__init__()
        self.normalize = normalize
        self.chanels = rule["chanels"](dataset1.chanels, dataset2.chanels)
        self.class_labels = rule["labels"](dataset1.class_labels, dataset2.class_labels)
        self.modalitys = rule["modalitys"](dataset1.modalitys, dataset2.modalitys)
        
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.rule = rule["data"]
        assert len(dataset1) == len(dataset2)

    def __len__(self):
        return len(self.dataset1)
    
    def __getitem__(self, index):
        data1 = self.dataset1[index]
        data2 = self.dataset2[index]
        img, label = self.rule(self.dataset1[index], self.dataset2[index])
        
        if self.normalize:
            img_mean = img.mean(dim=(1,2,3), keepdim=True)
            img_std = img.std(dim=(1,2,3), keepdim=True)
            img = (img - img_mean) / img_std
        
        return img, label

if __name__ == '__main__':
    dataset1 = MSDDataset()
    dataset2 = MSDDataset()
    com_dataset = CombinedDataset(dataset1, dataset2)
    x, y = com_dataset[0]
    print(x.shape, y.shape)