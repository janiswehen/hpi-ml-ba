import torch.utils.data as data
import torch.nn.functional as F
import torch

from unet.dataset.msd_dataset import MSDDataset

class PatchDataset(data.Dataset):
    def __init__(self, dataset: data.Dataset, patch_size=16):
        super().__init__()
        self.dataset = dataset
        self.patch_size = patch_size
        self.shape = dataset[0][0].shape
        self.patch_count = self.shape[-1] // patch_size if self.shape[-1] % patch_size == 0 else self.shape[-1] // patch_size + 1
        self.last_data_idx = None
        self.last_data = None
        self.class_labels = dataset.class_labels

    def __len__(self):
        return len(self.dataset) * self.patch_count
    
    def __getitem__(self, index):
        idx = index // self.patch_count
        scan, seg = None, None
        if idx == self.last_data_idx:
            scan, seg = self.last_data
        else:
            scan, seg = self.dataset[idx]
            del self.last_data
            self.last_data = scan, seg
            self.last_data_idx = idx
        start = index % self.patch_count * self.patch_size
        end = start + self.patch_size
        if end > self.shape[-1]:
            padding_size = end - self.shape[-1]
            scan = F.pad(scan, (0, padding_size))
            seg = F.pad(seg, (0, padding_size))
        return scan[..., start:end], seg[..., start:end]
    
    def get_original(self, tensor):
        N, C, W, H, P = tensor.shape
        D = self.shape[-1]
        
        # reorder dimensions back to (C, W, H, N, P)
        tensor = tensor.permute(1, 2, 3, 0, 4)

        # reshape back to the original shape (C, W, H, D')
        tensor = tensor.contiguous().view(C, W, H, -1)  # D' = N*P, which may be larger than D if padding was added

        # If there was padding, slice the tensor to remove it
        tensor = tensor[..., :D]  # slice the last dimension to the original depth D, effectively removing any padding

        return tensor

if __name__ == '__main__':
    data_dir = '/dhc/home/janis.wehen/data/Task01_BrainTumour/'
    dataset = MSDDataset(data_dir)
    patch_dataset = PatchDataset(dataset)
    patches = [], []
    for i in range(patch_dataset.patch_count+5):
        patches[0].append(patch_dataset[i][0])
        patches[1].append(patch_dataset[i][1])
    patches = torch.stack(patches[0][:patch_dataset.patch_count]), torch.stack(patches[1][:patch_dataset.patch_count])
    print(patches[0].shape, patches[1].shape)
    scan_re, seg_re = patch_dataset.get_original(patches[0]), patch_dataset.get_original(patches[1])
    print(scan_re.shape, seg_re.shape)
    scan, seg = dataset[0]
    print(torch.eq(scan, scan_re).all(), torch.eq(seg, seg_re).all())   
    