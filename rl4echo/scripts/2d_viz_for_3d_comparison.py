from pathlib import Path
import nibabel as nib
from matplotlib import pyplot as plt, animation
import numpy as np
import pandas as pd
from rl4echo.inference_from_seq import IMG_PATH
from torchio import Resize, LabelMap
from scipy import ndimage
from rl4echo.utils.Metrics import dice_score
import torch

def as_batch(action):
    y_pred_np_as_batch = action.transpose((2, 0, 1))
    return y_pred_np_as_batch


def clean_blobs(action):
    for i in range(action.shape[-1]):
        try:
            lbl, num = ndimage.label(action[..., i] != 0)
            # Count the number of elements per label
            count = np.bincount(lbl.flat)
            # Select the largest blob
            maxi = np.argmax(count[1:]) + 1
            # Remove the other blobs
            action[..., i][lbl != maxi] = 0
        except:
            print("WARNING EMPTY SEGMENTATION FRAME")
    return action


def load_and_clean(p, img_size=None, skip_blobs=False):
    nifti = nib.load(p)
    img = nifti.get_fdata() if skip_blobs else clean_blobs(nifti.get_fdata())
    img = as_batch(img)
    if img_size:
        resize_up = Resize((img_size[0], img_size[1], img_size[2]))
        img = LabelMap(tensor=img[None,], affine=np.diag(nifti.header["pixdim"][1:3].tolist() + [1, 0]))
        img = resize_up(img).numpy().squeeze(0)
    return img


if __name__ == "__main__":
    GT_PATH = '/data/icardio/subsets/full_3DRL_subset_norm_TESTONLY/segmentation/'
    IMG_PATH = '/data/icardio/subsets/full_3DRL_subset_norm_TESTONLY/img/'
    SET_PATH = [
        ("/home/local/USHERBROOKE/juda2901/dev/RL4Echo/testing_raw_CARDINAL_NEW_TESTSET/", "Baseline 3D U-Net"),
        ('/home/local/USHERBROOKE/juda2901/dev/ASCENT/ICARDIO_152TEST/inference_raw/', "nnU-Net"),
        ('/home/local/USHERBROOKE/juda2901/dev/MedSAM/iCardio/preds/MedSAM/', "MedSAM"),
        ("/home/local/USHERBROOKE/juda2901/dev/SAMUS/iCardio_testset_flipped/merged/", "SAMUS"),
        ("/home/local/USHERBROOKE/juda2901/dev/MemSAM/SAVED_MASKS/", "MemSAM"),
        ('/home/local/USHERBROOKE/juda2901/dev/RL4Echo/testing_raw_CARDINAL_FROM_MASK-SSL/', "MaskedSSL"),
        ('/data/icardio/subsets/full_3DRL_subset_norm_TESTONLY/2DMICCAI_segmentation/', "RL4Seg (2D)"),
        ('/home/local/USHERBROOKE/juda2901/dev/RL4Echo/testing_raw_LM+ANAT_BEST_NARVAL/', "RL4Seg3D"),
    ]

    VIZ_PATH = "./viz/2D_worstframe"
    Path(VIZ_PATH).mkdir(exist_ok=True)
    df = pd.read_csv("/data/icardio/subsets/full_3DRL_subset_norm_TESTONLY/subset_official_splits.csv", index_col=0)
    df = df[df['split_official_test'] == 'test']
    dicoms = [p.name.replace(".nii.gz", "") for p in Path(SET_PATH[2][0]).rglob('*.nii.gz') if p.name.replace(".nii.gz", "") in df['dicom_uuid'].to_list()]

    print(len(dicoms))
    dicoms = ["di-3C91-E35C-906B", "di-7061-409C-0CAF", "di-8075-490B-DF12", "di-1ECA-EAC3-8EAE"]
    fig, ax = plt.subplots(len(dicoms), 10, figsize=(16, 4))
    for idx, d in enumerate(dicoms):
        print(d)
        img_list = [(load_and_clean(next(Path(IMG_PATH).rglob(f"*{d}*")), skip_blobs=True), "Image")]
        img_list += [(load_and_clean(next(Path(GT_PATH).rglob(f"*{d}*"))), "GT")]
        img_list += [(load_and_clean(next(Path(p[0]).rglob(f"*{d}*")), img_size=img_list[0][0].shape), p[1]) for p in SET_PATH]

        dices = dice_score(torch.tensor(img_list[1][0]), torch.tensor(img_list[2][0])).numpy()
        i = np.argmin(dices)

        ax[idx, 0].imshow(img_list[0][0][i, ...].T, cmap='gray', interpolation='none')
        ax[idx, 0].axis('off')
        if idx == 0:
            ax[idx, 0].set_title(img_list[0][1])
        for j in range(1, len(img_list)):
            ax[idx, j].imshow(img_list[j][0][i, ...].T, cmap='gray', interpolation='none')
            ax[idx, j].axis('off')
            if idx == 0:
                ax[idx, j].set_title(img_list[j][1])

        plt.subplots_adjust(left=0.005,
                            bottom=0.0,
                            right=0.995,
                            top=1.0,
                            wspace=0.015,
                            hspace=0.0)
        # plt.subplot_tool()

        # if idx % 3 == 0:
    plt.show()
    plt.savefig(f"{VIZ_PATH}/dicoms-merged.png")
    plt.close()
