from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage
from torchio import Resize, LabelMap
from matplotlib import pyplot as plt, animation
from skimage.measure import find_contours


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
        # ("/home/local/USHERBROOKE/juda2901/dev/RL4Echo/testing_raw_CARDINAL_NEW_TESTSET/", "Baseline 3D U-Net"),
        # ('/home/local/USHERBROOKE/juda2901/dev/ASCENT/ICARDIO_152TEST/inference_raw/', "nnU-Net"),
        # ("/home/local/USHERBROOKE/juda2901/dev/SAMUS/iCardio_testset_flipped/merged/", "SAMUS"),
        # ("/home/local/USHERBROOKE/juda2901/dev/MemSAM/SAVED_MASKS/", "MemSAM"),
        # ('/home/local/USHERBROOKE/juda2901/dev/RL4Echo/testing_raw_CARDINAL_FROM_MASK-SSL/', "MaskedSSL"),
        ('/data/icardio/subsets/full_3DRL_subset_norm_TESTONLY/2DMICCAI_segmentation/', "RL4Seg (2D)"),
        # ('/home/local/USHERBROOKE/juda2901/dev/RL4Echo/testing_raw_LM+ANAT_BEST_NARVAL_TTA/', "RL4Seg3D (Anat. + LM)"),
        ('/home/local/USHERBROOKE/juda2901/dev/RL4Echo/testing_raw_ANAT-LM-T_NARVAL_TTA_LAST/', "RL4Seg3D (Anat. + LM + T.Pen.)"),
    ]

    # d = 'di-C9E0-1668-D365' # # 'di-C10D-4D1E-390F' # 'di-755E-299F-FA05' #'di-89C0-9C48-1DFA' # 'di-C10D-4D1E-390F'
    d = 'di-E60E-9C74-8500'
    img = (load_and_clean(next(Path(IMG_PATH).rglob(f"*{d}*")), skip_blobs=True), "Image")
    gt = (load_and_clean(next(Path(GT_PATH).rglob(f"*{d}*")), img_size=img[0].shape), 'GT')
    segs = [(load_and_clean(next(Path(p[0]).rglob(f"*{d}*")), img_size=img[0].shape), p[1]) for p in
                     SET_PATH]

    idxes = [175]
    # l, r = 170, 395
    l, r = 50, 375
    for i in idxes:
        plt.figure(tight_layout=True)
        plt.imshow(img[0][0].T, cmap='gray')
        # plt.axvline(i, ymin=0, ymax=1, c='r')
        plt.axhline(i, xmin=0, xmax=1, c=(0.7, 0, 0), linewidth=3)
        plt.axis('off')

        fig, ax = plt.subplots(len(segs) + 1, 1, tight_layout=True)

        ax[0].imshow(img[0][:, l:r, i], cmap='gray')
        ax[0].set_title("Image slice")
        ax[0].axis('off')

        endos = find_contours((gt[0][:, l:r, i] == 1).squeeze(), level=0.9)
        epis = find_contours((gt[0][:, l:r, i] != 0).squeeze(), level=0.9)

        from matplotlib.colors import LinearSegmentedColormap
        custom_cmap = LinearSegmentedColormap.from_list("custom", [(0, 0, 0), (0, 0.7, 0), (0.7, 0, 0)], N=3)

        for idx, seg in enumerate(segs):
            # ax[idx + 1].imshow(img[0][:, l:r, i], cmap='gray')
            ax[idx + 1].imshow(seg[0][:, l:r, i], cmap=custom_cmap, interpolation='none')
            for endo in endos:
                ax[idx + 1].plot(endo[:, 1], endo[:, 0], c='white', linewidth=2)
            for epi in epis:
                ax[idx + 1].plot(epi[:, 1], epi[:, 0], c='white', linewidth=2)
            ax[idx + 1].set_title(seg[1], fontsize=12)
            ax[idx + 1].axis('off')
        plt.show()

