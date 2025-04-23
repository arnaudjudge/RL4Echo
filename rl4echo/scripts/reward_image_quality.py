import json
import random

import h5py
import natsort
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

    # with open('/home/local/USHERBROOKE/juda2901/dev/qc/20250407_arnaud_qc_100pred_imgquality.json', "r") as f:
    #     d = json.load(f)
    #     data = d[1]['data']
    #
    # passed = [d['filename'].replace("_0000", "") for d in data if 'Pass' in d['status']]
    # failed = [d['filename'].replace("_0000", "") for d in data if 'Fail' in d['status']]
    # warn = [d['filename'].replace("_0000", "") for d in data if 'Warn' in d['status']]
    #
    # print(f"P {len(passed)}")
    # print(f"W {len(warn)}")
    # print(f"F {len(failed)}")

    failed_rewards = []
    passed_rewards = []
    agree_human = []
    rewards = []
    status = []
    dices = []

    with h5py.File("./../../3d_500_pred_image_quality_validation.h5", "r") as h5:

        for key in natsort.natsorted(h5.keys()):
            dicom = key.split("/")[-1].replace(".nii.gz", "")

            # if dicom not in warn:
            #     continue

            img = np.array(h5[key]['img']).transpose((2, 1, 0))
            pred = np.array(h5[key]['pred']).transpose((2, 1, 0))
            gt = np.array(h5[key]['gt']).transpose((2, 1, 0))
            reward = np.array(h5[key]['reward_map'])[0].transpose((2, 1, 0))
            d = dice(pred, gt, labels=[Label.LV, Label.MYO])

            dices += [d]
            reward_meanframe = reward.mean(axis=(1, 2))
            idx = np.argpartition(reward_meanframe, 5)[:5]

            if reward.max() > 1:
                reward = reward / 256
            reward = (1 - reward) * (gt.astype(np.uint8) == 0)
            criteria = (reward > 0.1).sum() / len(reward.flatten())

            print(f"{dicom} - {criteria < 0.08}")
            # criteria = reward_meanframe[idx].mean() < 0.985
            rewards += [criteria]

            # if dicom in passed:
            #     status += ['pass']
            # elif dicom in warn:
            #     status += ['warn']
            # elif dicom in failed:
            #     status += ['fail']
            # else:
            #     status += ['?']

            # human_pass = (dicom in passed) or (dicom in warn)
            # agree_human += [(human_pass != (criteria < 0.08))]


            #
            # dicom_in_failed = dicom in failed
            # if dicom_in_failed:
            #     failed_rewards += [reward_meanframe[idx].mean()]
            #
            # dicom_in_passed = dicom in passed
            # if dicom_in_passed:
            #     # print(reward_meanframe[idx].mean())
            #     passed_rewards += [reward_meanframe[idx].mean()]
            #
            # if criteria:
            #     print(dicom)
            #     print(reward_meanframe[idx])
            #     print(reward_meanframe[idx].mean())
            #
            #
            #     view_conf = df[df['dicom_uuid'] == dicom]['view_confidence'].item()
            #     print(view_conf)
            #     print("\n")

            plot = True
            if plot:
                # GIF
                fig, axes = plt.subplots(1, 1) #, figsize=(12, 6))
                bk = axes.imshow(img[0, ...], animated=True, cmap='gray')
                custom_cmap = LinearSegmentedColormap.from_list("custom", [(0, 0, 0), (0, 1, 0), (1, 0, 0)], N=3)
                # im1 = axes[0].imshow(pred[0, ...], animated=True,
                #                      cmap=custom_cmap,
                #                      alpha=0.2,
                #                      interpolation='none')
                axes.set_title(
                    f"Prediction, criteria : {criteria:.4f}, pass : {criteria < 0.08}")
                axes.axis("off")
                # im2 = axes[1].imshow(reward[0, ...], animated=True,
                #                      cmap='gray',
                #                      interpolation='none')
                # axes[1].set_title(
                #     f"Reward mean : {reward.mean():.4f}, criteria : {criteria:.4f}, pass : {criteria < 0.08}")
                # axes[1].axis("off")

                def update(i):
                    #im1.set_array(pred[i, ...])
                    # im2.set_array(reward[i, ...])
                    bk.set_array(img[i, ...])
                    return bk, #, im2


                animation_fig = animation.FuncAnimation(fig, update, frames=img.shape[0], interval=100, blit=True,
                                                        repeat_delay=10, )
                animation_fig.save(f"gifs_500/{dicom}.gif")
                # plt.show()
                plt.close()

        #
        # failed_rewards = np.asarray(failed_rewards)
        # passed_rewards = np.asarray(passed_rewards)
        agree_human = np.asarray(agree_human)
        rewards = np.asarray(rewards)
        print((rewards > 0.08).sum())

        dices = np.asarray(dices)

        def pltcolor(lst):
            cols = []
            for l in lst:
                if l == 'fail':
                    cols.append('red')
                elif l == 'pass':
                    cols.append('green')
                else:
                    cols.append('yellow')
            return cols

        # Create the colors list using the function above
        cols = pltcolor(status)

        plt.figure()
        plt.scatter(x=dices, y=rewards, c=cols)
        plt.show()


        print(agree_human.sum())