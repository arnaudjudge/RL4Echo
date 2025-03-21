import json
import random

import h5py
import numpy as np
import pandas as pd
import scipy.ndimage
import skimage
from matplotlib import pyplot as plt, animation
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import pearsonr
from scipy import ndimage
import cv2

from vital.data.camus.config import Label
from rl4echo.utils.test_metrics import dice


def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


if __name__ == "__main__":

    df = pd.read_csv("/data/icardio/processed/processed.csv")

    with open('/home/local/USHERBROOKE/juda2901/dev/qc/100pred_reward_test/20250204_arnaud_qc_100pred2.0 (3).json', "r") as f:
        d = json.load(f)
        data = d[1]['data']

    passed = [d['filename'].replace("_0000", "") for d in data if 'Pass' in d['status']]
    failed = [d['filename'].replace("_0000", "") for d in data if 'Fail' in d['status']]
    warn = [d['filename'].replace("_0000", "") for d in data if 'Warn' in d['status']]

    print(f"P {len(passed)}")
    print(f"W {len(warn)}")
    print(f"F {len(failed)}")

    with h5py.File("./../../3d_anatomical_reward_100randompred_wclean.h5", "r") as h5:

        # thresh = [0.95, 0.975, 0.9775, 0.98, 0.981, 0.982, 0.9825, 0.985, 0.9875, 0.99]
        prec = []
        rec = []
        thresh = [0.985]

        for t in thresh:
            dices = []
            reward_means = []
            error_means = []
            agree_human = []
            human_aq = []
            human_passes = []
            reward_passes = []
            view_conf = []

            plot = False

            for key in h5.keys():
                dicom = key.split("/")[-1].replace(".nii.gz", "")

                # if dicom not in warn:
                #     continue

                print(dicom)
                img = np.array(h5[key]['img']).transpose((2, 1, 0))
                pred = np.array(h5[key]['pred']).transpose((2, 1, 0))
                gt = np.array(h5[key]['gt']).transpose((2, 1, 0))
                reward = np.array(h5[key]['reward_map'])[0].transpose((2, 1, 0))

                error_map = (pred == gt).astype(np.uint8)

                view_conf += [df[df['dicom_uuid'] == dicom]['view_confidence'].item()]

                d = dice(pred, gt, labels=[Label.LV, Label.MYO])

                dices += [d]

                reward_means += [reward.mean()]
                reward_pass = reward.mean(axis=(1, 2)).min() > t
                reward_passes += [reward_pass]
                # reward_pass = reward.mean() > t
                human_pass = dicom in passed
                human_passes += [human_pass]
                error_means += [error_map.mean()]

                agree_human += [(human_pass == reward_pass)]

                if dicom in passed:
                    human_aq += ['pass']
                elif dicom in warn:
                    human_aq += ['warn']
                elif dicom in failed:
                    human_aq += ['fail']
                else:
                    human_aq += ['?']

                # if random.random() < 0.5:
                #     for i in range(len(reward)):
                #         plt.figure()
                #         plt.imshow(reward[i, ...] < 1)
                #         plt.show()

                if plot:  # and not (human_pass == reward_pass):
                    # GIF
                    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                    bk = axes[0].imshow(img[0, ...], animated=True, cmap='gray')
                    custom_cmap = LinearSegmentedColormap.from_list("custom", [(0, 0, 0), (0, 1, 0),  (1, 0, 0)], N=3)
                    im1 = axes[0].imshow(pred[0, ...], animated=True,
                                         cmap=custom_cmap,
                                         alpha=0.35,
                                         interpolation='none')
                    axes[0].set_title(
                        f"Prediction, dice : {d:.4f}")
                    axes[0].axis("off")
                    im2 = axes[1].imshow(reward[0, ...], animated=True,
                                         cmap='gray',
                                         interpolation='none')
                    axes[1].set_title(
                        f"Reward mean : {reward.mean():.4f}, reward frame min : {reward.mean(axis=(1,2)).min():.4f}, "
                        f"agree with human: {((dicom in passed) == reward_pass)}")
                    axes[1].axis("off")


                    def update(i):
                        im1.set_array(pred[i, ...])
                        im2.set_array(reward[i, ...])
                        bk.set_array(img[i, ...])
                        return bk, im1, im2


                    animation_fig = animation.FuncAnimation(fig, update, frames=img.shape[0], interval=100, blit=True,
                                                            repeat_delay=10, )
                    animation_fig.save(f"gifs_RNet_anat_metric/{dicom}_{human_aq[-1]}.gif")
                    # plt.show()
                    plt.close()

                # if d < 0.9:
                #     print(f"UNDER 80% DICE : {key}")
                #     plt.figure()
                #     plt.imshow(error_map[4, ...])
                #
                #     plt.figure()
                #     plt.imshow(reward[4, ...])
                #
                #     plt.show()


            dices = np.asarray(dices)
            reward_means = np.asarray(reward_means)
            error_means = np.asarray(error_means)
            view_conf = np.asarray(view_conf)
            agree_human = np.asarray(agree_human)
            reward_passes = np.asarray(reward_passes)
            human_passes = np.asarray(human_passes)

            print(agree_human.sum())

            human_aq = np.asarray(human_aq)

            # Calculate correlations
            corr_dice, _ = pearsonr(dices, reward_means)
            print(corr_dice)

            corr_view_conf, _ = pearsonr(dices, view_conf)
            print(corr_view_conf)

            corr_error, _ = pearsonr(error_means, reward_means)
            print(corr_error)

            tp = (reward_passes & (reward_passes == human_passes)).sum()
            tn = (~reward_passes & (reward_passes == human_passes)).sum()
            fp = (reward_passes & (reward_passes != human_passes)).sum()
            fn = (~reward_passes & (reward_passes != human_passes)).sum()

            print(tp)
            print(tn)
            print(fp)
            print(fn)

            prec += [tp / (tp + fp)]
            rec += [tp / (tp + fn)]


        plt.figure()
        plt.title("Precision recall curve")
        plt.plot(thresh, prec, c='g', marker='o', label='precision')
        plt.plot(thresh, rec, c='r', marker='o', label='recall')
        plt.legend()
        plt.show()


        #
        # PLOT
        #

        def pltcolor(lst):
            cols = []
            for l in lst:
                if l == 0:
                    cols.append('red')
                elif l == 1:
                    cols.append('green')
                else:
                    cols.append('blue')
            return cols

        # Create the colors list using the function above
        cols = pltcolor(agree_human)

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

        plt.figure()
        for i in range(len(reward_means)):
            plt.scatter(x=dices[i], y=reward_means[i], c=cols[i], marker=mrks[i])
        plt.title(f"Dice vs Reward correlation (Pearson corr: {corr_dice:.4f})")
        plt.ylabel("Reward mean")
        plt.xlabel("Dice")


        plt.figure()
        for i in range(len(reward_means)):
            plt.scatter(x=error_means[i], y=reward_means[i], c=cols[i], marker=mrks[i])
        plt.title(f"Error map vs Reward correlation (Pearson corr: {corr_error:.4f})")
        plt.ylabel("Reward mean")
        plt.xlabel("Error mean")

        plt.figure()
        for i in range(len(reward_means)):
            plt.scatter(x=dices[i], y=view_conf[i], c=cols[i], marker=mrks[i])
        plt.title(f"Dice vs View confidence correlation (Pearson corr: {corr_view_conf:.4f})")
        plt.ylabel("View confidence")
        plt.xlabel("Dice")

        plt.show()
