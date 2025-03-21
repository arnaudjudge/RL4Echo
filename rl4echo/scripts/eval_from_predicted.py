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


def get_test_metrics_list(pred_b, gt_b, voxel_spacing):
    test_dict = {}

    for i in range(len(pred_b)):
        d = dice(pred_b[i][None,...], gt_b[i][None,...], labels=(Label.BG, Label.LV, Label.MYO),
             exclude_bg=True, all_classes=True)
        d_epi = dice((pred_b[i][None,...] != 0).astype(np.uint8), (gt_b[i][None,...] != 0).astype(np.uint8),
                             labels=(Label.BG, Label.LV), exclude_bg=True, all_classes=False)

        hd = hausdorff(pred_b[i][None,...], gt_b[i][None,...], labels=(Label.BG, Label.LV, Label.MYO),
                            exclude_bg=True, all_classes=True, voxel_spacing=voxel_spacing[i][None,...])
        hd_epi = hausdorff((pred_b[i][None,...] != 0).astype(np.uint8), (gt_b[i][None,...] != 0).astype(np.uint8),
                                labels=(Label.BG, Label.LV), exclude_bg=True, all_classes=False,
                                voxel_spacing=voxel_spacing[i][None,...])['Hausdorff']

        for k, v in d.items():
            test_dict[k] = np.append(test_dict.get(k, np.asarray([])), v)
        for k, v in hd.items():
            test_dict[k] = np.append(test_dict.get(k, np.asarray([])), v)

        test_dict['dice/epi'] = np.append(test_dict.get('dice/epi', np.asarray([])), d_epi)
        test_dict['hd/epi'] = np.append(test_dict.get('hd/epi', np.asarray([])), hd_epi)

    return test_dict


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


def interp1d(array: np.ndarray, new_len: int) -> np.ndarray:
    la = len(array)
    return np.interp(np.linspace(0, la - 1, num=new_len), np.arange(la), array)


def generate_bar_plots_for_cycle(metric, area_curve, interp_len=30, prefix=''):
    # peaks, valleys = extract_cycle_points(area_curve)
    metric_interp = []
    for i in range(len(metric)):
        peaks, _ = extract_cycle_points(area_curve[i])
        print(peaks)
        print(metric[i].shape)
        if len(peaks) <= 1:
            if peaks[0] > len(metric[i]) // 2:
                peaks += [0]
                peaks.sort()
            else:
                peaks += [-1]
        print(peaks)
        metric[i] = metric[i][peaks[0]: peaks[1]]
        area_curve[i] = area_curve[i][peaks[0]: peaks[1]]
        if interp_len:
            metric_interp += [interp1d(metric[i], interp_len)]
            area_curve[i] = interp1d(area_curve[i], interp_len)

    metric_interp_std = np.asarray(metric_interp).std(axis=0)
    metric_interp = np.asarray(metric_interp).mean(axis=0)
    area_curve = np.asarray(area_curve).mean(axis=0)

    f, ax = plt.subplots()
    f.suptitle(f"Area curve for first cycle vs {prefix}")
    color = 'tab:blue'
    ax.plot(area_curve, color=color)
    ax.tick_params(axis='y', labelcolor=color)

    color = "tab:red"
    ax2 = ax.twinx()
    ax2.plot(metric_interp, color=color)
    ax2.fill_between(np.arange(len(metric_interp)),
                     metric_interp - metric_interp_std, metric_interp + metric_interp_std,
                     facecolor='red', alpha=0.15, edgecolor='none')
    ax2.tick_params(axis='y', labelcolor=color)
    # xlims = ax2.get_xlim()
    ylims = ax2.get_ylim()

    color = "tab:green"
    for m in metric:
        ax3 = ax2.twiny()
        ax3.plot(m, color=color, alpha=0.05)
        ax3.set_xticks([])
    # ax2.set_ylim(ylims)
    ax2.set_ylim([metric_interp.min() - (0.1 * metric_interp.mean()),
                  metric_interp.max() + (0.1 * metric_interp.mean())])


if __name__ == "__main__":
    GT_PATH = '/data/icardio/processed/segmentation/'
    REF_PATH = '../../testing_raw_5000narval16gpu/'
    SET_PATH ='/data/icardio/MICCAI2024/segmentation/'
    # SET_PATH = '../../icardio_pretrained_testing_raw/'

    GIF_PATH = './gifs/'
    Path(GIF_PATH).mkdir(exist_ok=True)

    results = {}
    area_curves = []

    metrics_df = pd.DataFrame()
    count = 0
    for p in Path(REF_PATH).rglob('*.nii.gz'): #di-AC08-4356-8824.nii.gz
        count += 1
        # if count > 30:
        #     break
        #
        # LOAD IMAGES
        #
        print(p)
        # pred1 = clean_blobs(nib.load(p).get_fdata())

        p1 = SET_PATH + p.relative_to(REF_PATH).as_posix()
        pred1 = clean_blobs(nib.load(p1).get_fdata())
        #
        # p3 = SET3_PATH + p.relative_to(SET1_PATH).as_posix()
        # pred3 = clean_blobs(nib.load(p3).get_fdata())

        gt_p = GT_PATH + p.relative_to(REF_PATH).as_posix()
        if not Path(gt_p).exists():
            print(f"{gt_p} - DOES NOT EXIST")
            continue
        gt = nib.load(gt_p).get_fdata()

        img_p = gt_p.replace(".nii.gz", "_0000.nii.gz").replace('segmentation', 'img')
        img = nib.load(img_p).get_fdata()

        #
        # METRICS
        #
        pred1_b = as_batch(pred1)
        # pred2_b = as_batch(pred2)
        # pred3_b = as_batch(pred3)
        gt_b = as_batch(gt)

        area_curve = EchoMeasure.structure_area(gt.transpose((2, 0, 1)), labels=1)
        area_curves += [area_curve]
        voxel_spacing = np.asarray([nib.load(img_p).header['pixdim'][1:3]]).repeat(
            repeats=len(pred1_b), axis=0)

        av_1 = is_anatomically_valid(pred1_b)
        # results['3dRL']['AV'] += int(all(av_1))
        # av_2 = is_anatomically_valid(pred2_b)
        # results['MICCAI']['AV'] += int(all(av_1))
        # av_3 = is_anatomically_valid(pred3_b)

        lm_1 = mitral_valve_distance(pred1_b, gt_b, voxel_spacing[0], return_mean=False)
        # lm_2 = mitral_valve_distance(pred2_b, gt_b, voxel_spacing[0], return_mean=False)
        # lm_3 = mitral_valve_distance(pred3_b, gt_b, voxel_spacing[0], return_mean=False)

        pred1_b_metrics = {**get_test_metrics_list(pred1_b, gt_b, voxel_spacing), **{'av': av_1.numpy()}, **lm_1}
        # pred2_b_metrics = {**get_test_metrics_list(pred2_b, gt_b, voxel_spacing), **{'av': av_2}, **lm_2}
        # pred3_b_metrics = {**get_test_metrics_list(pred3_b, gt_b, voxel_spacing), **{'av': av_3}, **lm_3}

        # generate_bar_plots_for_cycle([pred1_b_metrics["Dice"]], [area_curve], prefix="3dRL/Dice", interp_len=None)

        # anatomical validity averages only
        # pred1_b_metrics = {'av_frames': av_1.mean().item(), 'av': int(av_1.all().item())}
        # pred2_b_metrics = {'av_frames': av_2.mean().item(), 'av': int(av_2.all().item())}
        # pred3_b_metrics = {'av_frames': av_3.mean().item(), 'av': int(av_3.all().item())}

        # print(pred1_b_metrics)
        # print(pred2_b_metrics)
        # print(pred3_b_metrics)

        results[f'{p.stem.replace(".nii", "")}'] = {k: v for d in
                                                    [{k: v} for k, v in pred1_b_metrics.items()] +\
                                                    [{f"{k}_mean": v.mean() if isinstance(v, np.ndarray) else v} if "av" not in k else {f"{k}_frames_mean": v.mean(), f"{k}_all_mean": int(v.all())} for k, v in pred1_b_metrics.items()] for k, v in d.items()}

        # results[f'{p.stem.replace(".nii", "")}'] = {k: v for d in
        #                                             [{f"3dRL/{k}": v} for k, v in pred1_b_metrics.items()] +\
        #                                             [{f"MICCAI/{k}": v} for k, v in pred2_b_metrics.items()] + \
        #                                             [{f"baseline/{k}": v} for k, v in pred3_b_metrics.items()] for k, v in d.items()}

    df = pd.DataFrame.from_dict(results, orient='index')

    dice_mean = list(df['Dice'])
    generate_bar_plots_for_cycle(dice_mean, area_curves.copy(), prefix="Dice")
    hd_mean = list(df['Hausdorff'])
    generate_bar_plots_for_cycle(hd_mean, area_curves.copy(), prefix="hd")
    mae_mean = list(df['mae'])
    generate_bar_plots_for_cycle([m.mean(axis=1) for m in mae_mean], area_curves.copy(), prefix="LM_mae")
    av_mean = list(df['av'])
    generate_bar_plots_for_cycle(av_mean, area_curves.copy(), prefix="AV")

    plt.show()

    # print(df[df.columns[~df.columns.str.contains("av") & df.columns.str.contains("3dRL")]].mean())
    print(df[df.columns[df.columns.str.contains("mean")]].mean())
    # print(df[df.columns[df.columns.str.contains("MICCAI")]].mean())
    # print(df[df.columns[df.columns.str.contains("baseline")]].mean())
