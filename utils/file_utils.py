from pathlib import Path

import torch
import numpy as np
import nibabel as nib


def save_batch_to_dataset(b_img, b_gt, b_pred, batch_idx, save_dir="./dataset"):
    b_img = b_img.cpu().numpy()
    b_gt = b_gt.cpu().numpy()
    b_pred = b_pred.cpu().numpy()

    Path(f"{save_dir}/images").mkdir(parents=True, exist_ok=True)
    Path(f"{save_dir}/gt").mkdir(parents=True, exist_ok=True)
    Path(f"{save_dir}/pred").mkdir(parents=True, exist_ok=True)

    for i in range(len(b_img)):
        filename = f"{batch_idx}_{i}.nii.gz"
        affine = np.diag(np.asarray([1, 1, 1, 0]))
        hdr = nib.Nifti1Header()

        nifti_img = nib.Nifti1Image(b_img[i, ...], affine, hdr)
        nifti_img.to_filename(f"./{save_dir}/images/{filename}")

        nifti_gt = nib.Nifti1Image(np.expand_dims(b_gt[i, ...], 0), affine, hdr)
        nifti_gt.to_filename(f"./{save_dir}/gt/{filename}")

        nifti_pred = nib.Nifti1Image(b_pred[i, ...], affine, hdr)
        nifti_pred.to_filename(f"./{save_dir}/pred/{filename}")

