from itertools import cycle

import pandas as pd
import h5py
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    xl = pd.ExcelFile('results_TTO_analysis.xlsx')
    sheet_names_list = xl.sheet_names

    h5 = h5py.File("/data/rl4seg3d_stuff/3d_anatomical_reward_LM+ANAT_BEST_NARVAL_TTA_clean.h5", "r")
    reward_mins = {}
    min_ = 1.0
    for dicom in h5.keys():
        item = h5[dicom]
        reward = item["reward_map"]

        worst = np.asarray(reward).mean(axis=(0, 1, 2)).min()
        reward_mins.update({dicom: worst})
        min_ = worst if worst < min_ else min_
    print(reward_mins)


    tto_thresh = np.linspace(min_, 1.0, 10)
    # populations = np.linspace(0, 1.0, 20)
    # tto_thresh = [
    #     np.quantile(list(reward_mins.values()), 1 - p)
    #     for p in populations
    # ]
    # tto_thresh.reverse()
    print(tto_thresh)

    metrics = {'endo_epi-HD': {'ylim': [3.5, 6.0], "title": "Hausdorff Avg.", "ylabel":"mm"},
               'test-LM-mistake_per_cycle_7.5mm': {'ylim': [0, 2.0], "title":"MVC MpC 7.5mm" , "ylabel": "Mistake Count per Cycle"},
               'endo_epi-Dice': {'ylim': [90, 100], "title": "Dice Avg.", "ylabel":"%"},
               "test-anat_valid": {'ylim': [96.0, 100], "title": "Anatomical Validity", "ylabel":"% Subjects"},
               'test-temporal_valid': {'ylim': [83, 95], "title": "Temporal Validity", "ylabel":"% Subjects"}
    }

    colors = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    for metric, cfg in metrics.items():
        df = pd.read_excel("results_TTO_analysis.xlsx", sheet_name=metric)
        vals = []
        for t in tto_thresh:
            val = 0
            for dicom, rew in reward_mins.items():
                if rew > t:
                    val += df.loc[df['dicom_uuid']==dicom, 'RL4Seg3D'].item()
                else:
                    val += df.loc[df['dicom_uuid']==dicom, 'ALL_TTO'].item()

            vals += [val/len(reward_mins)]

        fig, ax = plt.subplots(figsize=(6, 4))

        vals = np.asarray(vals)
        if vals.max() < 1.0:
            vals = vals*100

        ax.plot(
            tto_thresh,
            vals,
            linewidth=2.5,
            marker="o",
            markersize=4
        )

        ax.set_xlabel("Uncertainty Trigger Threshold")
        ax.set_ylabel(cfg['ylabel'])
        ax.set_ylim(*cfg["ylim"])
        # lo, hi = vals.min(), vals.max()
        # pad = 1 * (hi - lo)
        # ax.set_ylim(lo - pad, hi + pad)

        # vmin, vmax = vals.min(), vals.max()
        # vrange = vmax - vmin if vmax > vmin else abs(vmax) * 0.1 + 1e-6
        # pad = 1.0 * vrange
        # lo = vmin - pad
        # hi = vmax + pad
        # # clamp inside semantic limits
        # lo = max(lo, cfg['ylim'][0])
        # hi = min(hi, cfg['ylim'][1])

        ax.set_title(cfg["title"])
        ax.grid(True, linestyle="--", alpha=0.4)

        plt.tight_layout()
        plt.show()


