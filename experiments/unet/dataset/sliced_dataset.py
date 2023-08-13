import torch.utils.data as data
import torch.nn.functional as F
import torch

from unet.dataset.msd_dataset import MSDDataset, MSDTask
from unet.utils.slicer import Slicer

class SlicedDataset(data.Dataset):
    def __init__(self, dataset: MSDDataset, patch_size=16, normalize=False, slice_axis=2):
        super().__init__()
        self.normalize = normalize
        self.dataset = dataset
        self.slicer = Slicer(slice_axis, patch_size)

        self.class_labels = dataset.class_labels
        self.modalitys = dataset.modalitys
        self.chanels = dataset.chanels

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        scan, seg = self.dataset[index]
        scan, seg = self.slicer.slice(scan), self.slicer.slice(seg)
        return scan, seg

    def get_original(self, tensor):
        tensor = self.slicer.merge(tensor)
        return tensor

# if __name__ == '__main__':
    # task = MSDTask.TASK08
    # dataset = MSDDataset(msd_task=task)
    # patch_dataset = SlicedDataset(dataset, patch_size=16, slice_axis=2)
    # patches = [], []
    # print(f"patch shape: {patch_dataset[0][0].shape}")
    # for i in range(patch_dataset.patch_count):
    #     patch = patch_dataset[i]
    #     patches[0].append(patch[0])
    #     patches[1].append(patch[1])
    # patches = torch.stack(patches[0]), torch.stack(patches[1])
    # print(f"stack shape: {patches[0].shape}, {patches[1].shape}")
    # scan_re, seg_re = patch_dataset.get_original(patches[0]), patch_dataset.get_original(patches[1])
    # print(scan_re.shape, seg_re.shape)
    # scan, seg = dataset[0]
    # print(torch.eq(scan, scan_re).all(), torch.eq(seg, seg_re).all())