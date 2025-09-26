import time

import nibabel as nib
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.ndimage
from matplotlib import pyplot as plt, animation
from matplotlib.colors import LinearSegmentedColormap
from scipy import ndimage
from torchio import Resize, ScalarImage

from rl4echo.utils.Metrics import is_anatomically_valid, mitral_valve_distance
from rl4echo.utils.test_metrics import dice, hausdorff
from rl4echo.utils.cardiac_cycle_utils import extract_cycle_points

from vital.metrics.evaluate.attribute import check_temporal_consistency_errors, compute_temporal_consistency_metric
from vital.utils.image.us.measure import EchoMeasure
from vital.data.camus.config import Label

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

def get_test_metrics(pred_b, gt_b, voxel_spacing):
    test_dice = dice(pred_b, gt_b, labels=(Label.BG, Label.LV, Label.MYO),
                     exclude_bg=True, all_classes=True)
    test_dice_epi = dice((pred_b != 0).astype(np.uint8), (gt_b != 0).astype(np.uint8),
                         labels=(Label.BG, Label.LV), exclude_bg=True, all_classes=False)

    test_hd = hausdorff(pred_b, gt_b, labels=(Label.BG, Label.LV, Label.MYO),
                        exclude_bg=True, all_classes=True, voxel_spacing=voxel_spacing)
    test_hd_epi = hausdorff((pred_b != 0).astype(np.uint8), (gt_b != 0).astype(np.uint8),
                            labels=(Label.BG, Label.LV), exclude_bg=True, all_classes=False,
                            voxel_spacing=voxel_spacing)['Hausdorff']

    return {**test_dice, **{"dice/epi": test_dice_epi}, **test_hd, **{"hd/epi": test_hd_epi}}


if __name__ == "__main__":
    GT_PATH = '/data/icardio/processed/segmentation/'
    SET1_PATH = '../../icardio_5000_miccai_testset/'
    SET2_PATH = '/data/icardio/MICCAI2024/segmentation/'

    GIF_PATH = './gifs/'
    Path(GIF_PATH).mkdir(exist_ok=True)

    results = {}

    es_ed_df = pd.read_csv("/home/local/USHERBROOKE/juda2901/dev/data/icardio/ES_ED_train_subset_affine/subset_official_test.csv")
    es_ed_dicoms = list(es_ed_df[es_ed_df['split_0'] == 'test']['dicom_uuid'].unique())

    count = 0
    for p in Path(GT_PATH).rglob('*.nii.gz'):
        if p.stem.replace(".nii", "") not in es_ed_dicoms:
            continue
        count += 1
        # if count > 2:
        #     break
        #
        # LOAD IMAGES
        #
        print(p)
        p1 = SET1_PATH + p.relative_to(GT_PATH).as_posix()
        pred1 = clean_blobs(nib.load(p1).get_fdata())

        p2 = SET2_PATH + p.relative_to(GT_PATH).as_posix()
        pred2 = clean_blobs(nib.load(p2).get_fdata())

        gt_p = GT_PATH + p.relative_to(GT_PATH).as_posix()
        if not Path(gt_p).exists():
            continue
        gt = nib.load(gt_p).get_fdata()

        area_curve = EchoMeasure.structure_area(gt.transpose((2, 0, 1)), labels=1)
        ED = np.argmax(area_curve)
        ES = np.argmin(area_curve)

        img_p = gt_p.replace(".nii.gz", "_0000.nii.gz").replace('segmentation', 'img')
        img = nib.load(img_p).get_fdata()

        #
        # METRICS
        #
        pred1_b = as_batch(pred1)[[ES, ED], ...]
        pred2_b = as_batch(pred2)[[ES, ED], ...]
        gt_b = as_batch(gt)[[ES, ED], ...]

        voxel_spacing = np.asarray([nib.load(img_p).header['pixdim'][1:3]]).repeat(repeats=len(gt_b), axis=0)

        av_1 = is_anatomically_valid(pred1_b)
        av_2 = is_anatomically_valid(pred2_b)
        pred1_b_metrics = {**get_test_metrics(pred1_b, gt_b, voxel_spacing), **{'av': av_1.mean().item()}}
        pred2_b_metrics = {**get_test_metrics(pred2_b, gt_b, voxel_spacing), **{'av': av_2.mean().item()}}

        print(pred2_b_metrics)

        results[f'{p.stem.replace(".nii", "")}'] = {k: v for d in [{f"3dRL/{k}": v} for k, v in pred1_b_metrics.items()] + \
                                                                [{f"MICCAI/{k}": v} for k, v in pred2_b_metrics.items()] for k, v in d.items()}

    print(count)
    df = pd.DataFrame.from_dict(results, orient='index')

    # print(df[df.columns[~df.columns.str.contains("av") & df.columns.str.contains("3dRL")]].mean())
    print(df[df.columns[df.columns.str.contains("3dRL")]].mean())
    print(df[df.columns[df.columns.str.contains("MICCAI")]].mean())
    # print(df[df.columns[~df.columns.str.contains("av") & df.columns.str.contains("baseline")]].mean())