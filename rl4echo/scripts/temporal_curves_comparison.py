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
from rl4echo.utils.test_metrics import full_test_metrics, dice, hausdorff
from vital.data.camus.config import Label

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


    test_dice = dice(pred_b, gt_b, labels=(Label.BG, Label.LV, Label.MYO),
                     exclude_bg=True, all_classes=True)
    test_dice_epi = dice((pred_b != 0).astype(np.uint8), (gt_b != 0).astype(np.uint8),
                         labels=(Label.BG, Label.LV), exclude_bg=True, all_classes=False)
    test_hd = hausdorff(pred_b, gt_b, labels=(Label.BG, Label.LV, Label.MYO),
                        exclude_bg=True, all_classes=True, voxel_spacing=voxel_spacing)
    test_hd_epi = hausdorff((pred_b != 0).astype(np.uint8), (gt_b != 0).astype(np.uint8),
                            labels=(Label.BG, Label.LV), exclude_bg=True, all_classes=False,
                            voxel_spacing=voxel_spacing)['Hausdorff']
    # print(test_dice, test_dice_epi)
    # print(test_hd, test_hd_epi)

    # compute metrics here
    const, measures = get_temporal_consistencies(pred_b.transpose((0, 2, 1)), (1, 1), skip_measurement_metrics=True)
    t_val, errors = check_temporal_validity(pred_b.transpose((0, 2, 1)),
                            voxel_spacing[0])

    from vital.metrics.evaluate.attribute import compute_temporal_consistency_metric
    lv_ = compute_temporal_consistency_metric(measures['lv_area'])
    print(lv_)
    lv_errors = [i for i in range(len(const['lv_area'])) if const['lv_area'][i]]
    myo_ = compute_temporal_consistency_metric(measures['myo_area'])
    print(myo_)
    myo_errors = [i for i in range(len(const['myo_area'])) if const['myo_area'][i]]
    print(lv_errors, myo_errors)
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
    return {p.name : (t_val, measures, lv_errors, myo_errors)}

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
        '/home/local/USHERBROOKE/juda2901/dev/RL4Echo/testing_raw_LM+ANAT_BEST_NARVAL/',
        '/data/icardio/subsets/full_3DRL_subset_norm_TESTONLY/segmentation/']

    GIF_PATH = None  # './gifs_RL4Seg_corrected/' # './gifs/'
    if GIF_PATH:
        Path(GIF_PATH).mkdir(exist_ok=True)

    measures = []
    for i in range(len(SET_PATHS)):
        m = {}
        paths = [p for p in Path(SET_PATHS[i]).rglob('*di-E60E-*.nii.gz')]

        # for idx, p in enumerate(tqdm(paths[::-1], total=len(paths))):
        #     m.update(do(p, show_plot=True))
        all_logs = process_map(do, paths, max_workers=12, chunksize=1)
        for a in all_logs:
            m.update(a)

        measures += [m]
    # print(measures[0].keys())
    # print(measures[1].keys())

    # gt_lv = 0
    # gt_myo = 0
    # d2_lv = 0
    # d2_myo = 0
    # d3_lv = 0
    # d3_myo = 0
    for k in measures[0].keys():

        # gt_lv += measures[2][k][2]
        # gt_myo += measures[2][k][3]
        # d2_lv += measures[0][k][2]
        # d2_myo += measures[0][k][3]
        # d3_lv += measures[1][k][2]
        # d3_myo += measures[1][k][3]


        if measures[1][k][0]:
            fig, ax1 = plt.subplots(figsize=(5,4), tight_layout=True)
            print(k)

            ax1.plot(measures[0][k][1]['lv_area'], color='b')
            ax1.plot(measures[1][k][1]['lv_area'], color='r')
            ax1.plot(measures[2][k][1]['lv_area'], color='black')
            # ax1.plot(measures[3][k][1]['lv_area'], color='g')
            ax1.set_xlabel("Frames", fontsize=12)
            ax1.set_ylabel("Area (Nb. of pixels)", fontsize=12)

            f, ax2 = plt.subplots(figsize=(5, 4), tight_layout=True)
            # ax2 = ax1.twinx()
            ax2.plot(measures[0][k][1]['myo_area'], color='b')
            ax2.plot(measures[1][k][1]['myo_area'], color='r')
            ax2.plot(measures[2][k][1]['myo_area'], color='black')
            # ax1.plot(measures[3][k][1]['myo_area'], '--', color='g', )

            ax2.set_xlabel("Frames", fontsize=12)
            ax2.set_ylabel("Area (Nb. of pixels)", fontsize=12)

            # Create custom handles
            from matplotlib.lines import Line2D
            # attr_handles = [
            #     Line2D([], [], color='k', linestyle='-', label='lv_area'),
            #     Line2D([], [], color='k', linestyle='--', label='myo_area'),
            # ]

            method_handles = [
                Line2D([], [], color='b', label='RL4Seg (2D)'),
                Line2D([], [], color='r', label='RL4Seg3D'),
                Line2D([], [], color='black', label='GT'),
            ]

            # Add both legends
            # leg1 = ax1.legend(handles=attr_handles, title="Attribute", loc='upper left')
            leg1 = ax1.legend(handles=method_handles, title="Method", loc='lower left', fontsize=9)
            leg2 = ax2.legend(handles=method_handles, title="Method", loc='lower right', fontsize=9)
            ax1.add_artist(leg1)
            ax2.add_artist(leg2)

            plt.show()

    # all_logs = process_map(do, paths, max_workers=12, chunksize=1)

