import warnings

import numpy as np
import torch
from matplotlib import pyplot as plt

from vital.data.camus.config import Label
from vital.metrics.camus.anatomical.utils import check_segmentation_validity
from vital.metrics.evaluate.attribute import compute_temporal_consistency_metric, check_temporal_consistency_errors
from vital.metrics.train.functional import differentiable_dice_score
from vital.utils.image.us.measure import EchoMeasure

warnings.simplefilter(action='ignore', category=FutureWarning)

def accuracy(pred, imgs, gt):
    actions = torch.round(pred)
    assert actions.shape == gt.shape, \
        print(f"Actions shape {actions.shape} vs GT shape {gt.shape}")
    simple = (actions == gt).float()
    return simple.mean(dim=(1, 2), keepdim=True)


def dice_score(output, target):
    classes = target.unique()
    out = torch.zeros(len(output), device=output.device)
    for i in range(len(output)):
        d = 0
        for c in classes:
            target_c = (target[i] == c)
            output_c = (output[i] == c)
            intersection = torch.sum(target_c * output_c)
            d += (2. * intersection) / (torch.sum(target_c) + torch.sum(output_c))
        out[i] = d / len(classes)
    return out


def is_anatomically_valid(output):
    out = torch.zeros(len(output))
    for i in range(len(output)):
        try:
            out[i] = int(check_segmentation_validity(output[i].T, (1.0, 1.0), [0, 1, 2]))
        except:
            out[i] = 0
    return out


def check_temporal_validity(segmentation_3d, voxelspacing, labels=None, relaxed_factor=4, skip_lv_bw_len=False):
    segmentation_3d = segmentation_3d.transpose((2, 0, 1))
    measures_1d = {}
    # calculate 1d signals
    try:
        measures_1d["lv_area"] = EchoMeasure.structure_area(segmentation_3d, labels=1)
        measures_1d["myo_area"] = EchoMeasure.structure_area(segmentation_3d, labels=2)
        measures_1d['epi_center_x'] = EchoMeasure.structure_center(segmentation_3d, labels=[1, 2], axis=1)
        measures_1d['epi_center_y'] = EchoMeasure.structure_center(segmentation_3d, labels=[1, 2], axis=0)
        if skip_lv_bw_len:
            measures_1d["lv_base_width"] = EchoMeasure.lv_base_width(segmentation_3d, lv_labels=1, myo_labels=2, voxelspacing=voxelspacing)
            measures_1d["lv_length"] = EchoMeasure.lv_length(segmentation_3d, lv_labels=1, myo_labels=2, voxelspacing=voxelspacing)
            # measures_1d["lv_orientation"] = EchoMeasure.structure_orientation(segmentation_3d, labels=labels, reference_orientation=90)
    except RuntimeError:
        return False

    attr_thresholds = {'lv_area': 2.5e-2,  # changed
                        'lv_base_width': 1.15e-1, #1.5e-1
                        'lv_length': 7e-2,
                        # 'lv_orientation': 2.29e-2,
                        'myo_area': 7.5e-2,  # changed
                        'epi_center_x': 3.5e-2,
                        'epi_center_y': 3.5e-2
                       }
    total_errors = []
    for attr in measures_1d.keys():
        thresh = attr_thresholds[attr]
        if relaxed_factor:
            thresh = thresh * relaxed_factor
        a = check_temporal_consistency_errors(thresh, measures_1d[attr],
                                              bounds=(measures_1d[attr].min(), measures_1d[attr].max()) if len(measures_1d[attr]) > 10 else None)
        total_errors += [a.sum()]
        idx = [i for i in range(len(a)) if a[i]]
        idxall = range(len(a))
        prev_neigh = measures_1d[attr][:-2]  # Previous neighbors of non-edge instants
        next_neigh = measures_1d[attr][2:]  # Next neighbors of non-edge instants
        neigh_inter_diff = ((prev_neigh + next_neigh) / 2)
        if a.sum() > 0:
            plt.figure()
            plt.plot(measures_1d[attr])
            plt.plot(measures_1d[attr], 'o')
            plt.plot(idx, measures_1d[attr][idx], 'x')
            plt.plot(idxall[1:-1], neigh_inter_diff)
            plt.title(attr)
        print(idx)
        print(compute_temporal_consistency_metric(measures_1d[attr])[idx])
        print(f"{attr}: {a.sum()} - THRESH :{thresh}")
        if a.sum() > 0:
            print(attr)
    plt.show()
    # allow for one metric to have one error in it if relaxed.
    return sum([e for e in total_errors]) <= 1 if relaxed_factor else sum([e != 0 for e in total_errors]) == 0


def mitral_valve_distance(segmentation, gt):
    mae = []
    mse = []
    mistakes = 0
    for i in range(len(gt)):
        try:
            lv_points = np.asarray(
                EchoMeasure._endo_base(gt[i].T, lv_labels=Label.LV, myo_labels=Label.MYO))
            p_points = np.asarray(
                EchoMeasure._endo_base(segmentation[i].T, lv_labels=Label.LV, myo_labels=Label.MYO))
            mae_values = np.asarray([np.linalg.norm(lv_points[0] - p_points[0]),
                                     np.linalg.norm(lv_points[1] - p_points[1])])
            mae += [mae_values.mean()]
            mse += [((lv_points - p_points) ** 2).mean()]

            # TODO: make this mm?
            if (mae_values > 15).all():
                mistakes += 2
            elif (mae_values > 15).any():
                mistakes += 1

            # plt.figure()
            # plt.imshow(gt[i].T)
            # plt.imshow(segmentation[i].T, alpha=0.35)
            # plt.title(mae_values)
            # plt.scatter(p_points[0, 1], p_points[0, 0], c='r')
            # plt.scatter(p_points[1, 1], p_points[1, 0], c='r')
            # plt.scatter(lv_points[0, 1], lv_points[0, 0], marker='x', c='g')
            # plt.scatter(lv_points[1, 1], lv_points[1, 0], marker='x', c='g')
            # plt.show()

        except Exception as e:
            print(f"except : {e}")
            mae += [segmentation.shape[-1]]
            mse += [segmentation.shape[-1] ** 2]
            mistakes += 2

    return np.asarray(mae), np.asarray(mse), np.asarray(mistakes)