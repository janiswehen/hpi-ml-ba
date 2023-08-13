import torch
from math import ceil
from torch.nn.functional import pad

class Slicer():
    def __init__(self, slice_axis, patch_size):
        self.slice_axis = slice_axis
        self.patch_size = patch_size

    def swap_slice_axis(self, tensor: torch.Tensor) -> torch.Tensor:
        if len(tensor.shape) == 4:
            if self.slice_axis == 0:
                tensor = tensor.permute(0, 3, 2, 1)
            elif self.slice_axis == 1:
                tensor = tensor.permute(0, 1, 3, 2)
            return tensor
        if len(tensor.shape) == 5:
            if self.slice_axis == 0:
                tensor = tensor.permute(0, 1, 4, 3, 2)
            elif self.slice_axis == 1:
                tensor = tensor.permute(0, 1, 2, 4, 3)
            return tensor
        raise Exception('Invalid tensor shape')

    def slice(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = self.swap_slice_axis(tensor)
        
        if self.patch_size == 1:
            return tensor.permute(3, 0, 1, 2)

        self.D  = tensor.shape[-1]
        self.patch_count = ceil(self.D / self.patch_size)

        padding_size = self.patch_count * self.patch_size - self.D
        if padding_size > 0:
            tensor = pad(tensor, (0, padding_size))
        
        slices = torch.stack(torch.split(tensor, self.patch_size, dim=-1))
        
        slices = self.swap_slice_axis(slices)
        
        return slices

    def merge(self, slices: torch.Tensor, D=None) -> torch.Tensor:
        if self.patch_size == 1:
            slices = slices.permute(1, 2, 3, 0)
            return self.swap_slice_axis(slices)
        
        slices = self.swap_slice_axis(slices)
        
        N, C, W, H, P = slices.shape
        
        slices = slices.permute(1, 2, 3, 0, 4)
        tensor = slices.contiguous().view(C, W, H, -1)
        tensor = tensor[..., :D] if D is not None else tensor[..., :self.D]
        
        tensor = self.swap_slice_axis(tensor)
        
        return tensor

if __name__ == '__main__':
    slicer = Slicer(slice_axis=1, patch_size=1)
    x = torch.randn(5, 31, 32, 33)
    slices = slicer.slice(x)
    y = slicer.merge(slices)
    print(slices.shape)
    print(y.shape)
    print(torch.eq(x, y).all())