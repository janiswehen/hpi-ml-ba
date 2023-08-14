import torch
import os
import numpy as np
import nibabel as nib

from unet.dataset.msd_dataset import MSDDataset, Split, MSDTask
from unet.dataset.downsampled_dataset import DownsampledDataset

N_SMPLES = 1

if __name__ == '__main__':
    for task in MSDTask:
        if task != MSDTask.TASK01:
            continue
        ds = MSDDataset(task, Split.Test)
        down_ds = DownsampledDataset(dataset=ds, scaling=(0.15,0.15,0.5))
        re_ds = DownsampledDataset(dataset=ds, scaling=(0.15,0.15,0.5), rescale=True)
        
        for i in range(N_SMPLES):
            img, mask = ds[i]
            down_img, down_mask = down_ds[i]
            re_img, re_mask = re_ds[i]
            
            img = img.permute(1, 2, 3, 0).numpy()
            mask = mask.permute(1, 2, 3, 0).numpy()
            down_img = down_img.permute(1, 2, 3, 0).numpy()
            down_mask = down_mask.permute(1, 2, 3, 0).numpy()
            re_img = re_img.permute(1, 2, 3, 0).numpy()
            re_mask = re_mask.permute(1, 2, 3, 0).numpy()
            
            print(np.unique(mask))

            # os.makedirs(f"data_examples/task_{task.value[0]}", exist_ok=True)

            # nib.save(nib.Nifti1Image(img, affine=np.eye(4)), f"data_examples/task_{task.value[0]}/{i}_img_n.nii.gz")
            # nib.save(nib.Nifti1Image(mask, affine=np.eye(4)), f"data_examples/task_{task.value[0]}/{i}_seg_n.nii.gz")
            # nib.save(nib.Nifti1Image(down_img, affine=np.eye(4)), f"data_examples/task_{task.value[0]}/{i}_img_d.nii.gz")
            # nib.save(nib.Nifti1Image(down_mask, affine=np.eye(4)), f"data_examples/task_{task.value[0]}/{i}_seg_d.nii.gz")
            # nib.save(nib.Nifti1Image(re_img, affine=np.eye(4)), f"data_examples/task_{task.value[0]}/{i}_img_r.nii.gz")
            # nib.save(nib.Nifti1Image(re_mask, affine=np.eye(4)), f"data_examples/task_{task.value[0]}/{i}_seg_r.nii.gz")
