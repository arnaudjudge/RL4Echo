from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import json

if __name__ == "__main__":

    dicoms = ["di-7B00-899D-99F7_ED", "di-D743-7A76-069A_ES", "di-DF19-2A64-5781_ES", "di-FC9F-91EC-F095_ED", "di-BCDC-817F-CE7A_ED",
             "di-3E16-CB66-5804_ED", "di-16AD-0764-4FB4_ED", "di-A29A-55AC-FE06_ES",
             "di-16AD-0764-4FB4_ES", "di-7E01-DF08-B284_ED", "di-E883-D53A-F51C_ES", "di-FC9F-91EC-F095_ES"]
    path_dict = {"RL4Seg (ours)": [
                     "/home/local/USHERBROOKE/juda2901/dev/RL4Echo/reward_uncertainty_RL4Seg/", 0.0829],
                 "MCDropout": ["/home/local/USHERBROOKE/juda2901/dev/TEMP/RL4Echo/results_mcdo2/",
                             0.1651],
                 "Ensemble": ["/home/local/USHERBROOKE/juda2901/dev/TEMP/RL4Echo/results_ensemble2/",
                              0.0913],
                 "TTA": ["/home/local/USHERBROOKE/juda2901/dev/TEMP/RL4Echo/results_TTA2/",
                             0.1135],
                 "PWA": ["/home/local/USHERBROOKE/juda2901/dev/TEMP/RL4Echo/results_aleatoric2/",
                             0.0901],
                 }

    for dicom in dicoms: #Path(path_dict['TTA'][0]).rglob("*_err*"):
        dicom = dicom.split("/")[-1][:20]
        print(dicom)
        f, ax = plt.subplots(3, len(path_dict),figsize=(10,6))
        i = 0

        for k, v in path_dict.items():
            values = np.load(f"{v[0]}/bins_conf.npy")
            error = np.load(f"{v[0]}/{dicom}_error.npy")
            unc = np.load(f"{v[0]}/{dicom}_unc.npy")

            ax[0, i].set_title(f"{k}: {v[1]}")
            ax[0, i].title.set_size(10)
            ax[0, i].plot([0, 1], [0, 1], "--", c="k", label="Perfect calibration")
            ax[0, i].plot(values[0], values[1], linewidth=2, label=f"{k}: {v[1]}", zorder=10 if "RL" in k else 0)
            ax[0, i].set_xticks([0, 1])
            ax[0, i].set_yticks([0, 1])
            ax[0, 0].set_ylabel("Accuracy", fontsize=10, labelpad=-10)
            ax[0, 0].set_xlabel("Confidence", fontsize=10, labelpad=-10)
            # ax[0, i].axis('off')

            ax[1, i].imshow(error[:225, :], cmap='grey')
            ax[1, i].axis('off')
            ax[2, i].imshow(unc[:225, :], cmap='grey')
            ax[2, i].axis('off')

            i += 1


        ax[1, 0].axis("on")
        ax[1, 0].set_ylabel("Error Map", fontsize=10)
        ax[1, 0].xaxis.set_visible(False)
        plt.setp(ax[1, 0].spines.values(), visible=False)
        ax[1, 0].tick_params(left=False, labelleft=False)
        ax[1, 0].patch.set_visible(False)
        #
        ax[2, 0].axis("on")
        ax[2, 0].set_ylabel("Uncertainty", fontsize=10)

        ax[2, 0].xaxis.set_visible(False)
        plt.setp(ax[2, 0].spines.values(), visible=False)
        ax[2, 0].tick_params(left=False, labelleft=False)
        ax[2, 0].patch.set_visible(False)

        plt.subplots_adjust(left=0.05,
                            bottom=0.0,
                            right=0.95,
                            top=0.95,
                            wspace=0.15,
                            hspace=0.115)

        # only plots
        # plt.subplots_adjust(left=0.05,
        #                     bottom=0.25,
        #                     right=0.975,
        #                     top=0.875,
        #                     wspace=0.15,
        #                     hspace=0.30)
        plt.subplot_tool()
        # plt.savefig(f"/data/rl_figure/fig_uncertainty/reliabilitydiag_{dicom}.png")

        plt.show()
        break


