import h5py
import numpy as np
import scipy.ndimage
import skimage
from matplotlib import pyplot as plt
from scipy import ndimage

from vital.data.camus.config import Label
from vital.utils.image.us.measure import EchoMeasure


def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


if __name__ == "__main__":

    with h5py.File("./../../3d_lm_results_10.h5", "r") as h5:
        print(len(h5.keys()))
        mae = []
        reward_means = []
        mistakes = []
        for key in h5.keys():

            img = np.array(h5[key]['img']).transpose((2, 1, 0))
            pred = np.array(h5[key]['pred']).transpose((2, 1, 0))
            gt = np.array(h5[key]['gt']).transpose((2, 1, 0))
            reward = np.array(h5[key]['reward_map']).transpose((2, 1, 0))

            lv_area = EchoMeasure.structure_area(gt, labels=1)
            print(key)
            if key == "di-15DE-96DC-C5C0":
                continue
            for i in range(len(gt)):

                try:
                    lv_points = np.asarray(
                        EchoMeasure._endo_base(gt[i], lv_labels=Label.LV, myo_labels=Label.MYO))
                    p_points = np.asarray(
                        EchoMeasure._endo_base(pred[i], lv_labels=Label.LV, myo_labels=Label.MYO))
                    mae_values = np.asarray([np.linalg.norm(lv_points[0] - p_points[0]),
                                  np.linalg.norm(lv_points[1] - p_points[1])])
                    mae += [mae_values]

                    num_pixels = 5 / 0.37
                    if (mae_values > num_pixels).any():
                        mistakes += [1]
                    else:
                        mistakes += [0]


                    # #
                    # plt.figure()
                    # plt.imshow(reward[i])
                    #
                    # plt.figure()
                    # plt.imshow(reward[i] < 0.5)

                    r = reward[i] < 0.90
                    lbl, num = ndimage.label(reward[i] < 0.90)
                    # Count the number of elements per label
                    count = np.bincount(lbl.flat)
                    # Sort the largest blobs
                    blobs = np.sort(count[1:])[::-1]
                    # Remove the other blobs
                    # min_vals = []
                    # for j in range(4):
                    #     rj = r.copy()
                    #     if len(count[1:]) <= j:
                    #         break
                    #     rj[lbl != count[1:].argsort()[-2:][::-1][j] + 1] = 1
                    #     min_vals += [(reward[i] * rj).min()]


                    r1 = r.copy()
                    r1[lbl != count[1:].argsort()[-2:][::-1][0] + 1] = 0
                    props1 = skimage.measure.regionprops(r1.astype(np.uint8))[0]
                    r1_ = reward[i].copy()
                    r1_[lbl != count[1:].argsort()[-2:][::-1][0] + 1] = 1
                    c1 = ndimage.center_of_mass((1 - r1_))
                    #
                    # plt.figure()
                    # plt.imshow(reward[i])
                    #
                    # plt.figure()
                    # plt.imshow(1 - r1_)
                    # plt.scatter(c1[1], c1[0], marker='x')
                    # plt.show()


                    r2 = r.copy()
                    r2[lbl != count[1:].argsort()[-2:][::-1][1] + 1] = 0
                    props2 = skimage.measure.regionprops(r2.astype(np.uint8))[0]
                    r2_ = reward[i].copy()
                    r2_[lbl != count[1:].argsort()[-2:][::-1][1] + 1] = 1
                    c2 = ndimage.center_of_mass((1 - r2_))

                    wrong = False
                    if props2.area < (props1.area / 10):
                        props2 = props1
                        wrong = True

                    # https://stackoverflow.com/questions/40820955/numpy-average-distance-from-array-center
                    m1 = np.sqrt(((np.argwhere(r1 == 1) - np.array(c1)) ** 2).sum(1)).mean()
                    # m1 = (np.sqrt(((np.argwhere(r1 == 1) - np.array(props1.centroid)) ** 2).sum(1))*(1-reward[i][r1 == 1])).mean()
                    m2 = np.sqrt(((np.argwhere(r2 == 1) - np.array(c2)) ** 2).sum(1)).mean()
                    # m2 = (np.sqrt(((np.argwhere(r2 == 1) - np.array(props2.centroid)) ** 2).sum(1))*(1-reward[i][r2 == 1])).mean()

                    # if mistakes[-1] and max(m1, m2) < (5 / 0.37):
                    #     plt.figure()
                    #     plt.imshow(img[i])
                    #     plt.imshow(gt[i], alpha=0.2)
                    #     plt.imshow(pred[i], alpha=0.35)
                    #     plt.title(mistakes[-1])
                    #     plt.scatter(p_points[0, 1], p_points[0, 0], c='r')
                    #     plt.scatter(p_points[1, 1], p_points[1, 0], c='y')
                    #     plt.scatter(lv_points[0, 1], lv_points[0, 0], marker='x', c='g')
                    #     plt.scatter(lv_points[1, 1], lv_points[1, 0], marker='x', c='b')
                    #
                    #     plt.figure()
                    #     plt.imshow(reward[i])
                    #
                    #     plt.figure()
                    #     plt.title(f"{m1}")
                    #     plt.imshow(r1)
                    #     plt.scatter(c1[1], c1[0], marker='x')
                    #
                    #     plt.figure()
                    #     plt.title(f"{m2}")
                    #     plt.imshow(r2)
                    #     plt.scatter(c2[1], c2[0], marker='x')
                    #
                    #     plt.figure()
                    #     plt.imshow(1 - r1_)
                    #     plt.scatter(c1[1], c1[0], marker='x')
                    #     plt.scatter(props1.centroid[1], props1.centroid[0], marker='x')
                    #     plt.show()


                    # r = np.sum(reward[i] < 0.9) / (reward[i].shape[0] * reward[i].shape[1])
                    # print(props1.axis_major_length)
                    # print(props2.axis_major_length)
                    # reward_means += [max(props1.axis_major_length / props1.axis_minor_length,
                    #                      props2.axis_major_length / props2.axis_minor_length)]
                    reward_means += [max(m1, m2)]
                    # plt.show()
                except Exception as e:
                    print(f"LM exception: {e}")
                    mae += [pred.shape[-1]]
                    mistakes[i] = 1
                    plt.figure()
                    plt.imshow(r1)

                    plt.figure()
                    plt.imshow(r2)
                    plt.show()
        mae = np.asarray(mae)
        reward_means = np.asarray(reward_means)

        mae = mae.mean(axis=-1)
        mask = np.asarray(mistakes) == 0

        # reward_means = (reward_means - reward_means.min()) / (reward_means.max() - reward_means.min())

        fig = plt.figure()
        plt.bar(mae[mask], reward_means[mask], color='blue')
        plt.bar(mae[~mask], reward_means[~mask], color='red')

        plt.axhline(y=5 / 0.37, linewidth=1, color='k')

        plt.xlabel("MAE")
        plt.ylabel("Reward map area above 0.9 (normalized)")
        plt.title("Reward map mean vs MAE")
        plt.show()

        print(f"No mistake min: {reward_means[mask].min()}")
        print(f"No mistake max: {reward_means[mask].max()}")

        print(f"Mistake min: {reward_means[~mask].min()}")
        print(f"Mistake max: {reward_means[~mask].max()}")



