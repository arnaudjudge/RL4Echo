import nibabel as nib
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.ndimage
from matplotlib import pyplot as plt, animation
from matplotlib.colors import LinearSegmentedColormap
from scipy import ndimage
from torchio import Resize, ScalarImage

from rl4echo.utils.Metrics import is_anatomically_valid
from rl4echo.utils.temporal_metrics import get_temporal_hd_metric
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


def check_temporal_validity(segmentation_3d, voxelspacing, relaxed_factor=None, plot=False, verbose=False):
    measures_1d = {}

    # calculate 1d signals
    try:
        measures_1d["lv_area"] = EchoMeasure.structure_area(segmentation_3d, labels=1, voxelarea=voxelspacing[0]*voxelspacing[1])
        measures_1d["myo_area"] = EchoMeasure.structure_area(segmentation_3d, labels=2, voxelarea=voxelspacing[0]*voxelspacing[1])
        measures_1d['epi_center_x'] = EchoMeasure.structure_center(segmentation_3d, labels=[1, 2], axis=0)
        measures_1d['epi_center_y'] = EchoMeasure.structure_center(segmentation_3d, labels=[1, 2], axis=1)
        measures_1d["lv_base_width"] = EchoMeasure.lv_base_width(segmentation_3d, lv_labels=1, myo_labels=2, voxelspacing=voxelspacing)
        measures_1d["lv_length"] = EchoMeasure.lv_length(segmentation_3d, lv_labels=1, myo_labels=2, voxelspacing=voxelspacing)
    except RuntimeError as e:
        print(e)
        return False, np.asarray([1, 1, 1, 1, 1, 1]), measures_1d, {}

    attr_thresholds = {'lv_area': 0.2,
                        'lv_base_width': 0.45,
                        'lv_length': 0.25,
                       'myo_area': 0.3,
                       'epi_center_x': 0.3,
                       'epi_center_y': 0.2,
                       'hd_frames_myo': 5.5,
                       'hd_frames_epi': 6.0,
                       }

    total_errors = []
    temp_constistencies = {}
    for attr in measures_1d.keys():
        thresh = attr_thresholds[attr]
        if relaxed_factor:
            thresh = thresh * relaxed_factor
        a = check_temporal_consistency_errors(thresh, measures_1d[attr])
        total_errors += [a.sum()]

        temp_constistencies[attr] = compute_temporal_consistency_metric(measures_1d[attr])
        idx = [i for i in range(len(a)) if a[i]]
        idxall = range(len(a))
        prev_neigh = measures_1d[attr][:-2]  # Previous neighbors of non-edge instants
        next_neigh = measures_1d[attr][2:]  # Next neighbors of non-edge instants
        neigh_inter_diff = ((prev_neigh + next_neigh) / 2)
        # new
        if plot:
            if a.sum() > 0:
                plt.figure()
                plt.plot(measures_1d[attr])
                plt.plot(measures_1d[attr], 'o')
                plt.plot(idx, measures_1d[attr][idx], 'x')
                plt.plot(idxall[1:-1], neigh_inter_diff)
                # plt.plot(idxall[1:-1], neigh_inter_diff2)
                plt.title(attr)

                plt.figure()
                plt.plot(temp_constistencies[attr])
                plt.title(attr)
        if verbose:
            print(idx)
            print(f"{attr}: {a.sum()} - THRESH :{thresh}")
            print(f"{attr} - {['%.4f' % tc for tc in temp_constistencies[attr]]}")
            if a.sum() > 0:
                print(f"{attr} - {['%.4f' % tc for tc in temp_constistencies[attr] if abs(tc) > thresh]}")
    if plot:
        plt.show()

    hds_myo = np.asarray(get_temporal_hd_metric(segmentation_3d, voxelspacing, Label.MYO))
    hds_epi = np.asarray(get_temporal_hd_metric(segmentation_3d, voxelspacing))
    print(hds_myo)
    print(hds_myo)
    temp_constistencies['hd_frames_myo'] = hds_myo < attr_thresholds['hd_frames_myo']
    temp_constistencies['hd_frames_epi'] = hds_epi < attr_thresholds['hd_frames_epi']

    total_errors += [(hds_myo > attr_thresholds['hd_frames_myo']).sum()]
    total_errors += [(hds_epi > attr_thresholds['hd_frames_epi']).sum()]

    # allow for one metric to have one error in it if relaxed.
    return sum([e for e in total_errors]) <= 1 if relaxed_factor else sum([e != 0 for e in total_errors]) == 0, np.asarray(total_errors), measures_1d, temp_constistencies # np.asarray([int(e != 0) for e in total_errors]).sum()
    # return sum([e for e in total_errors]) <= 1, np.asarray(total_errors).sum(), measures_1d, temp_constistencies # np.asarray([int(e != 0) for e in total_errors]).sum()


if __name__ == "__main__":
    GT_PATH = '/data/icardio/processed/segmentation/'
    SET1_PATH = '../../testing_raw_10k16gpu2ndtry/'
    SET2_PATH = '/data/icardio/MICCAI2024/segmentation/'

    GIF_PATH = './gifs/'
    Path(GIF_PATH).mkdir(exist_ok=True)

    results = {'3dRL': {"AV": 0, "TV": 0},
               'MICCAI': {"AV": 0, "TV": 0}}
    gt_temp_validities = 0

    metrics_df = pd.DataFrame()
    total_mae1 = 0
    total_mae2 = 0
    count = 0

    paths = Path(SET1_PATH).rglob('*.nii.gz')

    for p in paths: #BE5C 8CEF 3EF8
        #
        # LOAD IMAGES
        #
        # if Path(f"{GIF_PATH}/{p.stem.split('.')[0]}.gif").exists():
        #     continue
        print(p)
        pred1 = clean_blobs(nib.load(p).get_fdata())

        p2 = SET2_PATH + p.relative_to(SET1_PATH).as_posix()
        pred2 = clean_blobs(nib.load(p2).get_fdata())

        gt_p = GT_PATH + p.relative_to(SET1_PATH).as_posix()
        if not Path(gt_p).exists():
            continue
        gt = nib.load(gt_p).get_fdata()

        img_p = gt_p.replace(".nii.gz", "_0000.nii.gz").replace('segmentation', 'img')
        img = nib.load(img_p).get_fdata()

        #
        # METRICS
        #
        pred1_b = as_batch(pred1)
        pred2_b = as_batch(pred2)
        gt_b = as_batch(gt)

        plot = False
        if plot:
            # GIF
            print(pred1_b.shape)
            fig, ax = plt.subplots()
            im = ax.imshow(pred1_b[0].T, animated=True)

            def update(i):
                im.set_array(pred1_b[i].T)
                # fig.title(f"{i}")
                ax.set_title(i)
                return im,


            animation_fig = animation.FuncAnimation(fig, update, frames=len(pred1_b), interval=200, blit=False,
                                                    repeat_delay=1000, )

            # animation_fig.save(f"gifs_RNet_anat_metric/{dicom}_{human_aq[-1]}.gif")
            # plt.show()
            # plt.close()

        av_1 = is_anatomically_valid(pred1_b)
        results['3dRL']['AV'] += int(all(av_1))
        av_2 = is_anatomically_valid(pred2_b)
        results['MICCAI']['AV'] += int(all(av_2))
        # temporal metrics
        # temp_validity1 = 0
        temp_validity1, total_v_err1, measures, consistencies = check_temporal_validity(pred1_b.transpose((0, 2, 1)),
                                                nib.load(img_p).header['pixdim'][1:3],
                                                verbose=True)
        results['3dRL']['TV'] += int(temp_validity1)
        # temp_validity2 = 0
        temp_validity2, total_v_err2, _, _= check_temporal_validity(pred2_b.transpose((0, 2, 1)),
                                                nib.load(img_p).header['pixdim'][1:3],
                                                )
        results['MICCAI']['TV'] += int(temp_validity2)
        gt_temp_validity, total_v_err_gt, _, _ = check_temporal_validity(gt_b.transpose((0, 2, 1)),
                                                nib.load(img_p).header['pixdim'][1:3],
                                                )
        # gt_temp_validity = 1
        gt_temp_validities += gt_temp_validity

        # landmarks
        mae1 = []
        mse1 = []
        mistakes1 = 0
        mae2 = []
        mse2 = []
        mistakes2 = 0
        for i in range(len(gt_b)):
            lv_points = np.asarray(
                EchoMeasure._endo_base(gt_b[i].T, lv_labels=Label.LV, myo_labels=Label.MYO))
            try:
                p1_points = np.asarray(
                    EchoMeasure._endo_base(pred1_b[i].T, lv_labels=Label.LV,
                                           myo_labels=Label.MYO))
                mae_values = np.asarray([np.linalg.norm(lv_points[0] - p1_points[0]),
                                         np.linalg.norm(lv_points[1] - p1_points[1])])
                mae1 += [mae_values.mean()]
                if (mae_values > 10).any():
                    mistakes1 += 1
                mse1 += [((lv_points - p1_points) ** 2).mean()]
            except Exception as e:
                print(f"except : {e}")
                mae1 += [pred1.shape[-1]]
                mse1 += [pred1.shape[-1] ** 2]
            try:
                p2_points = np.asarray(
                    EchoMeasure._endo_base(pred2_b[i].T, lv_labels=Label.LV,
                                           myo_labels=Label.MYO))
                mae_values = np.asarray([np.linalg.norm(lv_points[0] - p2_points[0]),
                                         np.linalg.norm(lv_points[1] - p2_points[1])])
                mae2 += [mae_values.mean()]
                if (mae_values > 10).any():
                    mistakes2 += 1
                mse2 += [((lv_points - p2_points) ** 2).mean()]
            except Exception as e:
                print(f"except : {e}")
                mae2 += [pred2.shape[-1]]
                mse2 += [pred2.shape[-1] ** 2]

        total_mae1 += np.asarray(mae1).mean()
        total_mae2 += np.asarray(mae2).mean()

        d = {'dicom': p.stem.split('.')[0],
             'valid': temp_validity1,
             'totally_valid': total_v_err1.sum() and all(av_1),
             # 'lv_area_min': measures['lv_area'].min(),
             # # 'lv_base_width_min': measures['lv_base_width'].min(),
             # # 'lv_length_min': measures['lv_length'].min(),
             # 'myo_area_min': measures['myo_area'].min(),
             # 'epi_center_x_min': measures['epi_center_x'].min(),
             # 'epi_center_y_min': measures['epi_center_y'].min(),
             # 'lv_area_max': measures['lv_area'].max(),
             # # 'lv_base_width_max': measures['lv_base_width'].max(),
             # # 'lv_length_max': measures['lv_length'].max(),
             # 'myo_area_max': measures['myo_area'].max(),
             # 'epi_center_x_max': measures['epi_center_x'].max(),
             # 'epi_center_y_max': measures['epi_center_y'].max()
             }
        for k, v in consistencies.items():
            d.update({f"{k}_constistency_min": min(abs(v)), f"{k}_constistency_max": max(abs(v))})
        metrics_df = pd.concat([pd.DataFrame(d, index=[0]), metrics_df], ignore_index=True)

        #
        # FIGURES
        #
        if count % 1 == 0:
            fig, axes = plt.subplots(1, 3, figsize=(12, 6))
            bk = []
            for ax in axes:
                bk += [ax.imshow(img[..., 0].T, animated=True, cmap='gray')]

            custom_cmap = LinearSegmentedColormap.from_list("custom", [(0, 0, 0), (0, 1, 0), (1, 0, 0)], N=3)
            im1 = axes[0].imshow(pred1[..., 0].T, animated=True,
                           cmap=custom_cmap,
                           alpha=0.35,
                           interpolation='none')
            axes[0].set_title(f"3dRL\nAV: {'True' if av_1.all() else 'False'}\nTempV: {total_v_err1}"
                              f"\nLM MAE: {np.asarray(mae1).mean(): .4f}\n MSE: {np.asarray(mse1).mean(): .4f}")
            axes[0].axis("off")
            im2 = axes[1].imshow(pred2[..., 0].T, animated=True,
                           cmap=custom_cmap,
                           alpha=0.35,
                           interpolation='none')
            axes[1].set_title(f"MICCAI\nAV: {'True' if av_2.all() else 'False'}\nTempV: {total_v_err2}"
                              f"\nLM MAE: {np.asarray(mae2).mean(): .4f}\n MSE: {np.asarray(mse2).mean(): .4f}")
            axes[1].axis("off")
            im3 = axes[2].imshow(gt[..., 0].T, animated=True,
                           cmap=custom_cmap,
                           alpha=0.35,
                           interpolation='none')
            axes[2].set_title(f"Pseudo-GT\nTempV: {total_v_err_gt}")
            axes[2].axis("off")

            def update(i):
                im1.set_array(pred1[..., i].T)
                im2.set_array(pred2[..., i].T)
                im3.set_array(gt[..., i].T)
                for b in bk:
                    b.set_array(img[..., i].T)
                return bk[0], bk[1], bk[2], im1, im2, im3
            animation_fig = animation.FuncAnimation(fig, update, frames=img.shape[-1], interval=100, blit=True,
                                                    repeat_delay=10, )
            animation_fig.save(f"{GIF_PATH}/{p.stem.split('.')[0]}.gif")
            # plt.show()
            plt.close()

        count += 1

    print(results)
    print(gt_temp_validities)

    num = len([p for p in Path(SET1_PATH).rglob('*.nii.gz')])
    print(total_mae1 / num)
    print(total_mae2 / num)

    metrics_df.to_csv("metrics_1allowed.csv")
