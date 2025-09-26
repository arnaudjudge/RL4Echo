from pathlib import Path

import numpy as np
import nibabel as nib
import pandas as pd
import skimage.exposure as exp
import torch
from scipy import ndimage
from torchio import Resize, ScalarImage, LabelMap
from tqdm import tqdm

from rl4echo.utils.Metrics import is_anatomically_valid
from vital.models.segmentation.unet import UNet

from matplotlib import pyplot as plt

CSV_PATH = "/data/icardio/subsets/full_3DRL_subset_norm_TESTONLY/subset.csv"
IMG_PATH = "/data/icardio/subsets/full_3DRL_subset_norm_TESTONLY/img/"

OUT_PATH = "/data/icardio/subsets/full_3DRL_subset_norm_TESTONLY/2D_initial_actor_segmentation/"
CAP = 100000

def clean(s):
    lbl, num = ndimage.label(s != 0)
    # Count the number of elements per label
    count = np.bincount(lbl.flat)
    # Select the largest blob
    maxi = np.argmax(count[1:]) + 1
    # Remove the other blobs
    s[lbl != maxi] = 0
    return s


if __name__ == "__main__":

    df = pd.read_csv("/data/icardio/subsets/full_3DRL_subset_norm/subset_official_splits.csv", index_col=0)
    df = df[df["split_official_test"] == "test"]
    # df['MICCAI2024_frame_AV'] = df.get("MICCAI2024_frame_AV", None)
    # df['MICCAI2024_overall_AV'] = df.get("MICCAI2024_overall_AV", None)

    resize_down = Resize((256, 256, 1))

    model = UNet(input_shape=(2, 256, 256), output_shape=(1, 256, 256))
    model.load_state_dict(torch.load("/data/rl_logs_3d/run_4/2/rewardnet.ckpt")) #"/data/rl_logs_3d/run_4/3/actor.ckpt"))

    def get_segmentation(img):
        img = torch.tensor(img.astype(np.float32))
        if torch.cuda.is_available():
            model.cuda()
            img = img.cuda()
        out = torch.sigmoid(model(img))
        return out.cpu().detach().numpy()[0]
    j = 0

    # df_subset = df[(df['passed'] == True) & (df['MICCAI2024_overall_AV'].isna())]
    df_subset = df
    paths = [p for p in Path(IMG_PATH).rglob("*.nii.gz")]
    for idx in tqdm(df_subset.index, total=min(CAP, len(df_subset.index))):
        row = df.loc[idx].to_dict()
        label_path = Path(OUT_PATH) / row['study'] / str(row['view']).lower() / f"{row['dicom_uuid']}.nii.gz"
        if not label_path.exists() or 1 == 1:
            img_path = f"{IMG_PATH}/{row['study']}/{str(row['view']).lower()}/{row['dicom_uuid']}_0000.nii.gz"
            nifti_img = nib.load(img_path)
            data = nifti_img.get_fdata()

            seg = nib.load(label_path).get_fdata()

            # for i in range(data.shape[-1]):
            #     data[..., i] = exp.equalize_adapthist(data[..., i], clip_limit=0.01)
            resize_up = Resize((data.shape[0], data.shape[1], 1))
            label_seq = np.zeros_like(data, dtype=np.uint32)

            for i in range(data.shape[-1]):
                d = data[..., i]
                se = seg[..., i]
                #d = exp.equalize_adapthist(d, clip_limit=0.01)
                d_small = resize_down(ScalarImage(tensor=d[None, ..., None], affine=nifti_img.affine))
                s_small = resize_down(LabelMap(tensor=se[None, ..., None], affine=nifti_img.affine))

                in_ = np.concatenate((d_small.numpy().transpose((0, 3, 1, 2)), s_small.numpy().transpose((0, 3, 1, 2))), axis=1)

                # segment and post-process
                s = get_segmentation(in_)
                # s = np.argmax(s, axis=0)

                # s = clean(s)

                label = LabelMap(tensor=s[..., None], affine=d_small.affine)
                label = resize_up(label)

                label_seq[..., i] = label.numpy()[0, ..., 0]

            # label_nifti = nib.Nifti1Image(label_seq, header=nib.Nifti1Header(), affine=nifti_img.affine)
            # label_path.parent.mkdir(parents=True, exist_ok=True)
            # nib.save(label_nifti, label_path)
            import h5py
            with h5py.File('../3dUNC_2dMICCAIRNET.h5', 'a') as f:
                dicom = row['dicom_uuid']
                if dicom not in f:
                    f.create_group(dicom)
                # f[dicom]['img'] = (data * 255).astype(np.uint8)
                # f[dicom]['gt'] = .cpu().numpy().astype(np.uint8)
                f[dicom]['pred2d'] = seg.astype(np.uint8)
                f[dicom]['reward_map2d'] = label_seq.astype(np.uint8)

        else:
            label_seq = nib.load(label_path).get_fdata()

        # anatomical metrics
        # anatomical_validities = is_anatomically_valid(label_seq.transpose(2, 0, 1))

        # row['MICCAI2024_frame_AV'] = anatomical_validities.cpu().numpy().tolist()
        # row['MICCAI2024_overall_AV'] = all(anatomical_validities)
        # df.loc[idx] = row
        # df.to_csv(CSV_PATH)

        j += 1
        if j > CAP:
            break
