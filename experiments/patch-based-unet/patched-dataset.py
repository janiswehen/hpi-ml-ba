import torch.utils.data as data
import torch.nn.functional as F
from dataset import BratsDataset

class PatchDataset(data.Dataset):
    def __init__(self, dataset: data.Dataset, patch_size=16):
        super().__init__()
        self.dataset = dataset
        self.patch_size = patch_size
        self.shape = dataset[0][0].shape

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        item = self.dataset[index]
        return self.get_patch(item[0]), self.get_patch(item[1])

    def get_patch(self, tensor):
        C, W, H, D = tensor.shape

        # Calculate the number of patches
        num_patches = D // self.patch_size
        last_patch_size = D % self.patch_size

        if last_patch_size != 0:
            num_patches += 1
            padding_size = self.patch_size - last_patch_size
            # pad the last patch
            tensor = F.pad(tensor, (0, padding_size), "constant", 0)
            
        # Reshape tensor to add patch dimension
        tensor = tensor.view(C, W, H, num_patches, self.patch_size)
        
        # Reorder dimensions to (patches, C, W, H, P)
        tensor = tensor.permute(3, 0, 1, 2, 4)

        return tensor
    
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
    dataset = BratsDataset(data_dir)
    patch_dataset = PatchDataset(dataset)
    print(len(patch_dataset))
    print(patch_dataset[0][0].shape, patch_dataset[0][1].shape)
    print(patch_dataset.get_original(patch_dataset[0][0]).shape, patch_dataset.get_original(patch_dataset[0][1]).shape)