from cProfile import label

import numpy as np
import os
import warnings
from multiprocessing import Pool
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
# import matplotlib
# matplotlib.use("TkAgg")
from matplotlib import pyplot as plt, animation
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.pyplot import tight_layout
from scipy import ndimage
from skimage.measure import find_contours
from torchio import Resize, LabelMap
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import pandas as pd

from rl4echo.utils.test_metrics import full_test_metrics
from vital.data.camus.config import Label
from vital.utils.image.us.measure import EchoMeasure

warnings.simplefilter(action='ignore', category=UserWarning)

def resize_curve(arr, target_len=50):
    # Original and new sample positions
    x_old = np.linspace(0, 1, len(arr))
    x_new = np.linspace(0, 1, target_len)

    # Interpolate to match new length
    return np.interp(x_new, x_old, arr)

def as_batch(action):
    y_pred_np_as_batch = action.transpose((2, 0, 1))
    return y_pred_np_as_batch

def do(p):
    #
    # LOAD IMAGES
    #
    # print(p)
    pred_nifti = nib.load(p)
    pred = as_batch(pred_nifti.get_fdata())

    gt_p = next(Path(GT_PATH).rglob(f"*{p.name}"))  # only one exists
    gt = nib.load(gt_p).get_fdata()
    gt = as_batch(gt)

    img_p = gt_p.as_posix().replace(".nii.gz", "_0000.nii.gz").replace('segmentation', 'img')
    img_nifti = nib.load(img_p)
    img = as_batch(img_nifti.get_fdata())

    lv_area = EchoMeasure.structure_area(pred, Label.LV) / (pred.shape[-1]*pred.shape[-2])*100
    # myo_area = EchoMeasure.structure_area(p, Label.MYO)
    lv_area = resize_curve(lv_area)

    print(img.shape)

    pred_8 = as_batch(nib.load(p.as_posix().replace("/4/", "/8/")).get_fdata())
    pred_16 = as_batch(nib.load(p.as_posix().replace("/4/", "/16/")).get_fdata())
    pred_24 = as_batch(nib.load(p.as_posix().replace("/4/", "/24/")).get_fdata())

    custom_cmap = LinearSegmentedColormap.from_list("custom", [(0, 0, 0), (0, 1, 0), (1, 0, 0)], N=3)
    windows = [4, 8, 16, 24]
    fig, axes = plt.subplots(1, len(windows)+1, figsize=(12, 6), tight_layout=True)
    bk = []
    for ax in axes:
        bk += [ax.imshow(img[0].T, animated=True, cmap='gray')]

    axes[0].set_title("Image")
    axes[0].axis("off")
    im1 = axes[1].imshow(pred[0].T, animated=True,
                         cmap=custom_cmap,
                         alpha=0.35,
                         interpolation='none')
    axes[1].set_title(f"Window Length 4")
    axes[1].axis("off")
    im2 = axes[2].imshow(pred_8[0].T, animated=True,
                         cmap=custom_cmap,
                         alpha=0.35,
                         interpolation='none')
    axes[2].set_title(f"Window Length 8")
    axes[2].axis("off")
    im3 = axes[3].imshow(pred_16[0].T, animated=True,
                         cmap=custom_cmap,
                         alpha=0.35,
                         interpolation='none')
    axes[3].set_title(f"Window Length 16")
    axes[3].axis("off")
    im4 = axes[4].imshow(pred_24[0].T, animated=True,
                         cmap=custom_cmap,
                         alpha=0.35,
                         interpolation='none')
    axes[4].set_title(f"Window Length 24")
    axes[4].axis("off")

    def update(i):
        im1.set_array(pred[i].T)
        im2.set_array(pred_8[i].T)
        im3.set_array(pred_16[i].T)
        im4.set_array(pred_24[i].T)
        for b in bk:
            b.set_array(img[i].T)
        return bk[0], bk[1], bk[2], bk[3], bk[4], im1, im2, im3, im4

    animation_fig = animation.FuncAnimation(fig, update, frames=img.shape[0], interval=100, blit=False,
                                         repeat_delay=10, )
    animation_fig.save("baselines_smoothness.gif")
    plt.close()
    return
    l = 200
    r = 675
    i = 300

    endos = find_contours((gt[:, l:r, i] == 1).squeeze(), level=0.9)
    epis = find_contours((gt[:, l:r, i] != 0).squeeze(), level=0.9)

    custom_cmap = LinearSegmentedColormap.from_list("custom", [(0, 0, 0), (0, 0.7, 0), (0.7, 0, 0)], N=3)

    img2 = img.transpose(0, 2, 1)
    f2, ax2 = plt.subplots(1, 1, tight_layout=True)
    ax2.imshow(img2[0], cmap='gray', aspect='auto', interpolation='none')
    ax2.axis('off')
    ax2.axhline(i, xmin=0, xmax=1, c=(0.7, 0, 0), linewidth=5)
    f2.savefig('sw_img')

    f2, ax2 = plt.subplots(1, 1, tight_layout=True)
    ax2.imshow(img2[:, i, l:r], cmap='gray', aspect='auto', interpolation='none')
    ax2.axis('off')
    f2.savefig("sw_imgslice.png")

    f2, ax2 = plt.subplots(1, 1, tight_layout=True)
    ax2.imshow(img2[:, i, l:r], cmap='gray', aspect='auto', interpolation='none')
    ax2.imshow(pred[:, l:r, i], cmap=custom_cmap, alpha=0.5, aspect='auto', interpolation='none')
    for endo in endos:
        ax2.plot(endo[:, 1], endo[:, 0], c='white', linewidth=2)
    for epi in epis:
        ax2.plot(epi[:, 1], epi[:, 0], c='white', linewidth=2)
    ax2.axis('off')
    f2.savefig('sw_4rl.png')
    plt.show()
    # plt.close()

    return lv_area


if __name__ == "__main__":
    SET_PATH = '/home/local/USHERBROOKE/juda2901/dev/RL4Echo/testing_raw_SUPERVISED_T_ABLATION/4/'
    # SET_PATH = '/home/local/USHERBROOKE/juda2901/dev/RL4Echo/results/testing_raw_LM+ANAT_BEST_NARVAL_TTA/'
    GT_PATH = '/data/icardio/subsets/full_3DRL_subset_norm_TESTONLY/segmentation/'
    # SET_PATH = '/data/icardio/subsets/full_3DRL_subset_norm_TESTONLY/segmentation/'

    paths = [p for p in Path(SET_PATH).rglob('*di-9BBB*.nii.gz')]

    # if single thread for loop...
    all = []
    for idx, p in enumerate(tqdm(reversed(paths[::-1]), total=len(paths))):
        all += [do(p)]
        if idx > 5:
            break

    # arr = process_map(do, paths, max_workers=12, chunksize=1)
    # arr = np.asarray(arr)
    #
    # mean_curve = arr.mean(axis=0)
    # std = arr.std(axis=0)
    #
    # plt.figure()
    # for c in arr:
    #     plt.plot(c, alpha=0.2)
    #
    # plt.plot(mean_curve, linewidth=3)
    # plt.fill_between(range(len(mean_curve)),
    #                  mean_curve - std,
    #                  mean_curve + std,
    #                  alpha=0.3)
    # plt.title("Mean Curve + Std Band")
    # plt.show()
