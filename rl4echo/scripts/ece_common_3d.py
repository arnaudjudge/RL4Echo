from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import json

from matplotlib import gridspec

if __name__ == "__main__":

    path_dict = {"RL4Seg3D (ours)": [
                     "/home/local/USHERBROOKE/juda2901/dev/RL4Echo/unc_results_RNET_TScale_New/", 0.057],
                 "MCDropout": ["/home/local/USHERBROOKE/juda2901/dev/RL4Echo/unc_results_MCDropout/",
                             0.109],
                 "Ensemble": ["/home/local/USHERBROOKE/juda2901/dev/RL4Echo/unc_results_Ensemble/",
                              0.070],
                 "TTA": ["/home/local/USHERBROOKE/juda2901/dev/RL4Echo/unc_results_TTA/",
                             0.105],
                 "RL4Seg (2D)": ["/home/local/USHERBROOKE/juda2901/dev/RL4Echo/unc_results_RNET_TScale_2D/",
                             0.088],
                 }
    i = -1
    plt.rcParams['font.size'] = 12
    fig = plt.figure(figsize=(18, 3)) #, constrained_layout=True)
    gs = gridspec.GridSpec(1, 5, figure=fig) #, height_ratios=[2, 1, 1]) #, hspace=0.5, wspace=0.3)

    ax_large = fig.add_subplot(gs[0, 0])

    # Four smaller plots below it (row 2)
    axs_small = []

    for k, v in path_dict.items():
        values = np.load(f"{v[0]}/bins_conf_w_size.npy")

        if k == "RL4Seg3D (ours)":
            ax_large.set_title(f"{k}: {v[1]:.3f}")
            # ax_large.title.set_size(10)
            ax_large.plot([0, 1], [0, 1], "--", c="k", label="Perfect calibration")
            ax_large.plot(values[0], values[1], linewidth=2, label=f"{k}: {v[1]}", color='#1C84C4')
            ax_large.set_xticks([0, 1])
            ax_large.set_yticks([0, 1])
            ax_large.set_ylabel("Accuracy", fontsize=12, labelpad=-10)
            ax_large.set_xlabel("Confidence", fontsize=12, labelpad=-10)
            ax_large2 = ax_large.twinx()
            try:
                ax_large2.bar(values[0], values[2], alpha=0.7, width=np.min(np.diff(values[0])) / 2, color='darkgray')
            except:
                ax_large2.bar(values[0], values[2], alpha=0.7, color='darkgray')
            ax_large2.axis("off")

            # ax_large.set_aspect('equal')
        else:
            # ax = fig.add_subplot(gs[(1 + int(i >= 2)), i % 2])
            ax = fig.add_subplot(gs[0, i + 1])

            ax.set_title(f"{k}: {v[1]:.3f}")
            # ax.title.set_size(10)
            ax.plot([0, 1], [0, 1], "--", c="k", label="Perfect calibration")
            ax.plot(values[0], values[1], linewidth=2, label=f"{k}: {v[1]}", color='#1C84C4')
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])

            ax2 = ax.twinx()
            try:
                ax2.bar(values[0], values[2], alpha=0.7, width=np.min(np.diff(values[0])) / 2, color='darkgray')
            except:
                ax2.bar(values[0], values[2], alpha=0.7, color='darkgray')
            ax2.axis("off")
            axs_small.append(ax)
            axs_small.append(ax2)


        # ax[0, i].axis('off')
        i += 1

    # ax[0, 0].set_ylabel("Accuracy", fontsize=10, labelpad=-10)
    # ax[0, 0].set_xlabel("Confidence", fontsize=10, labelpad=-10)
    # ax[1, 0].axis("on")
    # ax[1, 0].set_ylabel("Error Map", fontsize=10)
    # ax[1, 0].xaxis.set_visible(False)
    # plt.setp(ax[1, 0].spines.values(), visible=False)
    # ax[1, 0].tick_params(left=False, labelleft=False)
    # ax[1, 0].patch.set_visible(False)
    # #
    # ax[2, 0].axis("on")
    # ax[2, 0].set_ylabel("Uncertainty", fontsize=10)
    #
    # ax[2, 0].xaxis.set_visible(False)
    # plt.setp(ax[2, 0].spines.values(), visible=False)
    # ax[2, 0].tick_params(left=False, labelleft=False)
    # ax[2, 0].patch.set_visible(False)

    # plt.subplots_adjust(left=0.05,
    #                     bottom=0.025,
    #                     right=0.95,
    #                     top=0.95,
    #                     wspace=0.15,
    #                     hspace=0.2)

    # only plots
    plt.subplots_adjust(left=0.02,
                        bottom=0.15,
                        right=0.99,
                        top=0.875,
                        wspace=0.15,
                        hspace=0.30)
    # plt.subplot_tool()
    plt.savefig("/home/local/USHERBROOKE/juda2901/dev/3d_ece_w_hist.png")

    # plt.show()


