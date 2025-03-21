import json
import os
import re
from multiprocessing import Pool
from pathlib import Path

import nibabel as nib
import numpy as np
from matplotlib import pyplot as plt
from medpy.metric import hd
from numpy import repeat
from scipy import ndimage
from tqdm import tqdm

from rl4echo.utils.Metrics import is_anatomically_valid, is_anatomically_valid_multiproc
from rl4echo.utils.temporal_metrics import get_temporal_consistencies, attr_thresholds
from rl4echo.utils.test_metrics import hausdorff, dice
from vital.data.camus.config import Label
from vital.metrics.camus.anatomical.lv_metrics import LeftVentricleMetrics
from vital.metrics.camus.anatomical.myo_metrics import MyocardiumMetrics
from vital.metrics.evaluate.segmentation import Segmentation2DMetrics


def as_batch(action):
    y_pred_np_as_batch = action.transpose((2, 0, 1))
    return y_pred_np_as_batch


def clean_blobs(action):
    for i in range(action.shape[-1]):
        lbl, num = ndimage.label(action[..., i] != 0)
        # Count the number of elements per label
        count = np.bincount(lbl.flat)
        # Select the largest blob
        maxi = np.argmax(count[1:]) + 1
        # Remove the other blobs
        action[..., i][lbl != maxi] = 0
    return action


if __name__ == "__main__":
    GT_PATH = '/data/icardio/processed/segmentation/'
    # SET_PATH = '../../testing_raw_10k16gpu2ndtry/'
    SET_PATH = '../../testing_raw_100_randompred/'


    GIF_PATH = './gifs/'
    Path(GIF_PATH).mkdir(exist_ok=True)

    paths = [p for p in Path(SET_PATH).rglob('*.nii.gz')]
    bad = ["di-BE5C-222A-A340.nii.gz", "di-34A8-C1E5-9259.nii.gz"]  # bad
    good = ["di-25F8-52C8-964B.nii.gz", "di-8B43-85EB-63D5.nii.gz"]
    print(len(paths))

    with open('/home/local/USHERBROOKE/juda2901/dev/RL4Echo/testing_raw_100_randompred/20250225_arnaud_qc_100pred2.0 (1).json', "r") as f:
        d = json.load(f)
        data = d[1]['data']

    passed = [d['filename'].replace("_0000", "") for d in data if 'Pass' in d['status']]
    failed = [d['filename'].replace("_0000", "") for d in data if 'Fail' in d['status']]
    warn = [d['filename'].replace("_0000", "") for d in data if 'Warn' in d['status']]

    print(f"P {len(passed)}")
    print(f"W {len(warn)}")
    print(f"F {len(failed)}")

    maxes = []
    dices = []
    human_passes = []
    human_aq = []

    for p in paths:
        # if p.name not in good:
        #     continue
        # #
        # LOAD IMAGES
        #
        print(p)
        dicom = p.stem.replace(".nii", "")
        print(dicom)
        pred1 = clean_blobs(nib.load(p).get_fdata())

        gt_p = GT_PATH + p.relative_to(SET_PATH).as_posix()
        if not Path(gt_p).exists():
            continue
        gt = nib.load(gt_p).get_fdata()

        img_p = gt_p.replace(".nii.gz", "_0000.nii.gz").replace('segmentation', 'img')
        img = nib.load(img_p).get_fdata()

        #
        # METRICS
        #

        d = dice(pred1, gt, labels=[Label.LV, Label.MYO])
        dices += [d]

        human_pass = dicom in passed
        human_passes += [human_pass]

        if dicom in passed:
            human_aq += ['pass']
        elif dicom in warn:
            human_aq += ['warn']
        elif dicom in failed:
            human_aq += ['fail']
        else:
            human_aq += ['?']

        pred1_b = as_batch(pred1)
        gt_b = as_batch(gt)

        # av = is_anatomically_valid_multiproc(pred1_b, nib.load(img_p).header['pixdim'][1:3])
        # print(av)

        voxel_spacing = nib.load(img_p).header['pixdim'][1:3]

        # t_consistencies, measures = get_temporal_consistencies(pred1_b, voxel_spacing)

        # print(t_consistencies)
        # print(measures)
        #
        # total_errors = []
        # for attr in t_consistencies.keys():
        #     thresh = attr_thresholds[attr]
        #     t_consistency = t_consistencies[attr]
        #     total_errors += [t_consistency.sum()]
        #
        # print(np.array(total_errors).sum())

        # test = (np.ones_like(pred1_b).T * np.arange(len(pred1_b))).T

        prev_neigh = pred1_b[:-2]  # Previous neighbors of non-edge instants
        next_neigh = pred1_b[2:]  # Next neighbors of non-edge instants
        tuples = [(pred1_b[1:-1][i], prev_neigh[i], next_neigh[i]) for i in range(len(pred1_b[1:-1]))]

        def temporal_hd(tpls, voxel_spacing, label=None):
            curr, bckw, forw = tpls
            if label:
                return (hd(curr == label, forw == label, voxel_spacing) +
                        hd(curr == label, bckw == label, voxel_spacing)) /2
            return (hd(curr, forw, voxel_spacing) + hd(curr, bckw, voxel_spacing)) / 2

        with Pool(processes=os.cpu_count()) as pool:
            out = list(
                pool.starmap(
                    temporal_hd,
                    zip(
                        tuples,
                        repeat(voxel_spacing, repeats=len(tuples)),
                    )
                )
            )
        print(f"{human_aq[-1]} vs {max(out)} ({out})")
        maxes += [max(out)]

    dices = np.array(dices)
    maxes = np.array(maxes)
    human_aq = np.asarray(human_aq)


    def pltmarker(lst):
        mrk = []
        for l in lst:
            if l == 'pass':
                mrk.append('$p$')
            elif l == 'warn':
                mrk.append('$w$')
            elif l == 'fail':
                mrk.append('$f$')
            else:
                mrk.append('$unk$')
        return mrk
    mrks = pltmarker(human_aq)


    def pltcolor(lst):
        mrk = []
        for l in lst:
            if l == 'pass':
                mrk.append('green')
            elif l == 'warn':
                mrk.append('red')
            elif l == 'fail':
                mrk.append('red')
            else:
                mrk.append('red')
        return mrk
    clrs = pltcolor(human_aq)

    for i in range(len(maxes)):
        plt.scatter(x=dices[i], y=maxes[i], c=clrs[i], marker=mrks[i])
    plt.title(f"Dice vs hd_temporal")
    plt.ylabel("HD temporal mean")
    plt.xlabel("Dice mean")

    plt.axhline(5, linestyle='--')
    plt.axhline(5.5, linestyle='--')
    plt.axhline(6, linestyle='--')
    plt.axhline(6.5, linestyle='--')
    plt.axhline(7, linestyle='--')

    plt.show()

