import numpy as np
from matplotlib import pyplot as plt

from vital.metrics.evaluate.attribute import compute_temporal_consistency_metric, check_temporal_consistency_errors
from vital.utils.image.us.measure import EchoMeasure

attr_thresholds = {
    'lv_area': 0.2,
    'lv_base_width': 0.45,
    'lv_length': 0.25,
    'myo_area': 0.3,
    'epi_center_x': 0.3,
    'epi_center_y': 0.2,
}


def get_temporal_consistencies(segmentation_3d, voxelspacing):
    measures_1d = {}
    # calculate measures
    # if exception, make sure threshold is triggered
    try:
        measures_1d["lv_area"] = EchoMeasure.structure_area(segmentation_3d, labels=1, voxelarea=voxelspacing[0]*voxelspacing[1])
    except :
        print("lv_area extraction failed")
        measures_1d["lv_area"] = np.resize([1, 0], len(segmentation_3d))

    try:
        measures_1d["myo_area"] = EchoMeasure.structure_area(segmentation_3d, labels=2, voxelarea=voxelspacing[0]*voxelspacing[1])
    except:
        print("myo_area extraction failed")
        measures_1d["myo_area"] = np.resize([1, 0], len(segmentation_3d))

    try:
        measures_1d['epi_center_x'] = EchoMeasure.structure_center(segmentation_3d, labels=[1, 2], axis=0)
    except:
        print("epi_center_x extraction failed")
        measures_1d["epi_center_x"] = np.resize([1, 0], len(segmentation_3d))

    try:
        measures_1d['epi_center_y'] = EchoMeasure.structure_center(segmentation_3d, labels=[1, 2], axis=1)
    except:
        print("epi_center_y extraction failed")
        measures_1d["epi_center_y"] = np.resize([1, 0], len(segmentation_3d))

    try:
        measures_1d["lv_base_width"] = EchoMeasure.lv_base_width(segmentation_3d, lv_labels=1, myo_labels=2, voxelspacing=voxelspacing)
    except:
        print("lv_base_width extraction failed")
        measures_1d["lv_base_width"] = np.resize([1, 0], len(segmentation_3d))

    try:
        measures_1d["lv_length"] = EchoMeasure.lv_length(segmentation_3d, lv_labels=1, myo_labels=2, voxelspacing=voxelspacing)
    except:
        print("lv_length extraction failed")
        measures_1d["lv_length"] = np.resize([1, 0], len(segmentation_3d))

    t_consistencies = {}
    for attr in measures_1d.keys():
        thresh = attr_thresholds[attr]
        t_consistencies[attr] = check_temporal_consistency_errors(thresh, measures_1d[attr])

    return t_consistencies, measures_1d


def check_temporal_validity(segmentation_3d, voxelspacing, relaxed_factor=None, plot=False, verbose=False):
    total_errors = []
    temp_constistencies, measures_1d = get_temporal_consistencies(segmentation_3d, voxelspacing)
    for attr in temp_constistencies.keys():
        thresh = attr_thresholds[attr]
        t_consistency = temp_constistencies[attr]
        total_errors += [t_consistency.sum()]

        if plot:
            temp_constistencies[attr] = compute_temporal_consistency_metric(measures_1d[attr])
            idx = [i for i in range(len(t_consistency)) if t_consistency[i]]
            idxall = range(len(t_consistency))
            prev_neigh = measures_1d[attr][:-2]  # Previous neighbors of non-edge instants
            next_neigh = measures_1d[attr][2:]  # Next neighbors of non-edge instants
            neigh_inter_diff = ((prev_neigh + next_neigh) / 2)
            if t_consistency.sum() > 0:
                plt.figure()
                plt.plot(measures_1d[attr])
                plt.plot(measures_1d[attr], 'o')
                plt.plot(idx, measures_1d[attr][idx], 'x')
                plt.plot(idxall[1:-1], neigh_inter_diff)
                plt.title(attr)

                plt.figure()
                plt.plot(temp_constistencies[attr])
                plt.title(attr)
        if verbose:
            idx = [i for i in range(len(t_consistency)) if t_consistency[i]]
            print(idx)
            print(f"{attr}: {t_consistency.sum()} - THRESH :{thresh}")
            print(f"{attr} - {['%.4f' % tc for tc in temp_constistencies[attr]]}")
            if t_consistency.sum() > 0:
                print(f"{attr} - {['%.4f' % tc for tc in temp_constistencies[attr] if abs(tc) > thresh]}")
    if plot:
        plt.show()
    # allow for one metric to have one error in it if relaxed.
    print(sum([e != 0 for e in total_errors]))
    return sum([e for e in total_errors]) <= 1 if relaxed_factor else sum([e != 0 for e in total_errors]) == 0


