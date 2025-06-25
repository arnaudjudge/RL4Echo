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
from scipy import ndimage
from torchio import Resize, LabelMap
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from rl4echo.utils.correctors import AEMorphoCorrector
from rl4echo.utils.temporal_metrics import get_temporal_consistencies, check_temporal_validity
from rl4echo.utils.test_metrics import full_test_metrics

warnings.simplefilter(action='ignore', category=UserWarning)


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


def dict_mean(dict_list):
    mean_dict = {}
    for key in dict_list[0].keys():
        mean_dict[key] = sum(d[key] for d in dict_list) / len(dict_list)
    return mean_dict


def do(p, show_plot=False):
    #
    # LOAD IMAGES
    #
    # print(p)
    pred_nifti = nib.load(p)
    pred = clean_blobs(pred_nifti.get_fdata())

    gt_p = next(Path(GT_PATH).rglob(f"*{p.name}"))  # only one exists
    gt = nib.load(gt_p).get_fdata()

    img_p = gt_p.as_posix().replace(".nii.gz", "_0000.nii.gz").replace('segmentation', 'img')
    img_nifti = nib.load(img_p)
    img = img_nifti.get_fdata()

    #
    # METRICS
    #

    # resize if needed
    if pred.shape != gt.shape:
        resize_up = Resize((gt.shape[0], gt.shape[1], gt.shape[2]))
        pred = LabelMap(tensor=pred[None,], affine=np.diag(pred_nifti.header["pixdim"][1:3].tolist() + [1, 0]))
        pred = resize_up(pred).numpy().squeeze(0)

    pred_b = as_batch(pred)
    gt_b = as_batch(gt)
    # print(pred_b.shape, gt_b.shape, img.shape)

    # get voxel spacing
    voxel_spacing = np.asarray([img_nifti.header["pixdim"][1:3]]).repeat(repeats=len(pred_b), axis=0)

    # compute metrics here
    _, measures = get_temporal_consistencies(pred_b.transpose((0, 2, 1)), (1, 1), skip_measurement_metrics=True)
    t_val, _ = check_temporal_validity(pred_b.transpose((0, 2, 1)),
                            voxel_spacing[0])
    #
    # FIGURES
    #
    if show_plot:
        plt.figure()
        plt.plot(measures['lv_area'])

        plt.figure()
        plt.plot(measures['myo_area'])

        plt.show()
        print("show")
    return {p.name : (t_val, measures)}

if __name__ == "__main__":
    GT_PATH = '/data/icardio/subsets/full_3DRL_subset_norm_TESTONLY/segmentation/'
    SET_PATHS = [
        # "/home/local/USHERBROOKE/juda2901/dev/MemSAM/SAVED_MASKS/",
        # "/home/local/USHERBROOKE/juda2901/dev/SAMUS/iCardio_testset_flipped/1/" # SET split_mask=False
        # "/home/local/USHERBROOKE/juda2901/dev/RL4Echo/testing_raw_CARDINAL_NEW_TESTSET/",
        "/data/icardio/subsets/full_3DRL_subset_norm_TESTONLY/2DMICCAI_segmentation/",
        # '/home/local/USHERBROOKE/juda2901/dev/ASCENT/ICARDIO_152TEST/inference_raw/',
    # SET_PATH = '/home/local/USHERBROOKE/juda2901/dev/MedSAM/iCardio/preds/MedSAM/'
    #     '/home/local/USHERBROOKE/juda2901/dev/RL4Echo/testing_raw_CARDINAL_FROM_MASK-SSL/',
    # SET_PATH = '/home/local/USHERBROOKE/juda2901/dev/RL4Echo/testing_raw_CARDINAL_NO_MASK-SSL/'
        '/home/local/USHERBROOKE/juda2901/dev/RL4Echo/testing_raw_LM+ANAT_BEST_NARVAL/']

    GIF_PATH = None  # './gifs_RL4Seg_corrected/' # './gifs/'
    if GIF_PATH:
        Path(GIF_PATH).mkdir(exist_ok=True)

    measures = []
    for i in range(len(SET_PATHS)):
        m = {}
        paths = [p for p in Path(SET_PATHS[i]).rglob('*.nii.gz')]

        # for idx, p in enumerate(tqdm(paths[::-1], total=len(paths))):
        #     m.update(do(p, show_plot=False))
        all_logs = process_map(do, paths, max_workers=12, chunksize=1)
        for a in all_logs:
            m.update(a)

        measures += [m]
    print(measures[0].keys())
    print(measures[1].keys())

    for k in measures[0].keys():
        if measures[1][k][0]:
            fig, ax1 = plt.subplots()

            ax1.plot(measures[0][k][1]['lv_area'], color='b')
            ax1.plot(measures[1][k][1]['lv_area'], color='r')
            # ax1.plot(measures[2][k][1]['lv_area'], color='y')
            # ax1.plot(measures[3][k][1]['lv_area'], color='g')

            # ax2 = ax1.twinx()
            ax1.plot(measures[0][k][1]['myo_area'], '--',color='b')
            ax1.plot(measures[1][k][1]['myo_area'], '--', color='r')
            # ax1.plot(measures[2][k][1]['myo_area'], '--', color='y', )
            # ax1.plot(measures[3][k][1]['myo_area'], '--', color='g', )

            ax1.set_xlabel("Frames")
            ax1.set_ylabel("Number of pixels")

            # Create custom handles
            from matplotlib.lines import Line2D
            attr_handles = [
                Line2D([], [], color='k', linestyle='-', label='lv_area'),
                Line2D([], [], color='k', linestyle='--', label='myo_area'),
            ]

            method_handles = [
                Line2D([], [], color='b', label='RL4Seg (2D)'),
                Line2D([], [], color='r', label='RL4Seg3D'),
            ]

            # Add both legends
            leg1 = ax1.legend(handles=attr_handles, title="Attribute", loc='upper left')
            leg2 = ax1.legend(handles=method_handles, title="Method", loc='upper right')
            ax1.add_artist(leg1)

            plt.show()


    # all_logs = process_map(do, paths, max_workers=12, chunksize=1)

