from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import json
import torch
from skimage.measure import find_contours

from vital.models.segmentation.unet import UNet

if __name__ == "__main__":

    rn_path = '/data/rl_logs/1_entropy/3/rewardnet.ckpt'
    temp_factor = 4.1778

    rnet = UNet(input_shape=(2, 256, 256), output_shape=(1, 256, 256))
    rnet.load_state_dict(torch.load(rn_path))

    def get_reward(pred, img):
        pred = torch.tensor(pred.T)
        img = torch.tensor(img.T)
        if torch.cuda.is_available():
            rnet.cuda()
            pred = pred.cuda()
            img = img.cuda()
        stack = torch.stack((img.squeeze(0), pred), dim=0).unsqueeze(0)
        out = torch.sigmoid(rnet(stack)/temp_factor).squeeze(1)
        return out.cpu().detach().numpy()[0].T

    dicoms = ["di-D743-7A76-069A_ES", "di-DF19-2A64-5781_ES", "di-FC9F-91EC-F095_ED", "di-BCDC-817F-CE7A_ED",
             "di-7B00-899D-99F7_ED", "di-3E16-CB66-5804_ED", "di-16AD-0764-4FB4_ED", "di-A29A-55AC-FE06_ES",
             "di-16AD-0764-4FB4_ES", "di-7E01-DF08-B284_ED", "di-E883-D53A-F51C_ES", "di-FC9F-91EC-F095_ES"]
    path_dict = {"MCDropout": ["/home/local/USHERBROOKE/juda2901/dev/TEMP/RL4Echo/outputs/2024-03-01/14-08-46/RESULTS.h5", "di-FC9F-91EC-F095_ED"],
                 "Ensemble": ["/home/local/USHERBROOKE/juda2901/dev/TEMP/RL4Echo/outputs/2024-03-01/10-01-17/RESULTS.h5", "di-7B00-899D-99F7_ED"],
                 "TTA": ["/home/local/USHERBROOKE/juda2901/dev/TEMP/RL4Echo/outputs/2024-03-01/06-24-13/RESULTS.h5", "di-16AD-0764-4FB4_ES"],
                 "PWA": ["/home/local/USHERBROOKE/juda2901/dev/TEMP/RL4Echo/outputs/2024-03-01/08-26-29/RESULTS.h5", "di-BCDC-817F-CE7A_ED"],
                 }

    f, ax = plt.subplots(4, 4, figsize=(8, 7.5))
    i = 0
    for k, v in path_dict.items():
        with h5py.File(v[0], "r") as h5:
            dicom = v[1]
            d = h5[dicom]
            img = np.array(d['img']).T
            pred = np.array(d['pred']).T
            gt = np.array(d['gt']).T
            uncertainty_map = np.array(d['reward_map']).T
            error = np.array(d['accuracy_map']).T

            contour = find_contours((pred == 1).squeeze(), level=0.9)[0]
            ax[0, i].plot(contour[:, 1], contour[:, 0], c='r')
            contours = find_contours((pred == 2).squeeze(), level=0.9)
            for contour in contours:
                ax[0, i].plot(contour[:, 1], contour[:, 0], c='r')
            reward = get_reward(pred, img)
            ax[0, i].imshow(img, cmap='grey')
            ax[1, i].imshow(error, cmap='grey')
            ax[2, i].imshow(uncertainty_map, cmap='grey')
            ax[3, i].imshow(1 - reward, cmap='grey')

            ax[0, i].axis('off')
            ax[1, i].axis('off')
            ax[2, i].axis('off')
            ax[3, i].axis('off')


        i += 1

    ax[0, 0].axis("on")
    ax[0, 0].set_xlabel("MCDropout", fontsize=10)
    ax[0, 0].set_ylabel("Image/Segmentation", fontsize=10)
    ax[0, 0].xaxis.set_label_position('top')
    ax[0, 0].set_yticklabels([])
    ax[0, 0].set_xticklabels([])
    ax[0, 0].set_yticks([])
    ax[0, 0].set_xticks([])
    plt.setp(ax[2, 0].spines.values(), visible=False)
    ax[0, 0].tick_params(left=False, labelleft=False)
    ax[0, 0].patch.set_visible(False)

    ax[1, 0].axis("on")
    ax[1, 0].set_ylabel("Error Map", fontsize=10)
    ax[1, 0].xaxis.set_visible(False)
    plt.setp(ax[1, 0].spines.values(), visible=False)
    ax[1, 0].tick_params(left=False, labelleft=False)
    ax[1, 0].patch.set_visible(False)

    ax[2, 0].axis("on")
    ax[2, 0].set_ylabel("SOTA Uncertainty", fontsize=10)
    ax[2, 0].xaxis.set_visible(False)
    plt.setp(ax[2, 0].spines.values(), visible=False)
    ax[2, 0].tick_params(left=False, labelleft=False)
    ax[2, 0].patch.set_visible(False)

    ax[3, 0].axis("on")
    ax[3, 0].set_ylabel("RL4Seg Uncetainty", fontsize=10)
    ax[3, 0].xaxis.set_visible(False)
    plt.setp(ax[3, 0].spines.values(), visible=False)
    ax[3, 0].tick_params(left=False, labelleft=False)
    ax[3, 0].patch.set_visible(False)


    ax[0, 1].axis("on")
    ax[0, 1].set_xlabel("Ensemble", fontsize=10)
    ax[0, 1].xaxis.set_label_position('top')
    ax[0, 1].set_yticklabels([])
    ax[0, 1].set_xticklabels([])
    ax[0, 1].set_yticks([])
    ax[0, 1].set_xticks([])

    ax[0, 2].axis("on")
    ax[0, 2].set_xlabel("TTA", fontsize=10)
    ax[0, 2].xaxis.set_label_position('top')
    ax[0, 2].set_yticklabels([])
    ax[0, 2].set_xticklabels([])
    ax[0, 2].set_yticks([])
    ax[0, 2].set_xticks([])

    ax[0, 3].axis("on")
    ax[0, 3].set_xlabel("PWA", fontsize=10)
    ax[0, 3].xaxis.set_label_position('top')
    ax[0, 3].set_yticklabels([])
    ax[0, 3].set_xticklabels([])
    ax[0, 3].set_yticks([])
    ax[0, 3].set_xticks([])

    # plt.subplots_adjust(left=0.025,
    #                     bottom=0.0,
    #                     right=0.975,
    #                     top=0.975,
    #                     wspace=0.025,
    #                     hspace=0.025)
    plt.subplot_tool()

    # plt.savefig(f"/data/rl_figure/fig_unc_supp/additional_unc.png")

    plt.show()
