import h5py
import numpy as np
import pandas as pd
import torch
from tqdm.contrib.concurrent import process_map

from rl4echo.utils.test_metrics import full_test_metrics
from collections import defaultdict
import matplotlib.pyplot as plt


def do(img, pred, gt, rew):
    b_img = img.transpose((2, 0, 1))
    b_pred = pred.transpose((2, 0, 1))
    b_gt = gt.transpose((2, 0, 1))

    b_reward = 1 - rew.transpose((2, 0, 1))

    worst_frame_reward = b_reward.mean(axis=(1, 2)).min()

    # plt.figure()
    # plt.imshow(b_reward[0].T)
    # plt.show()

    logs = full_test_metrics(b_pred, b_gt, np.asarray([[0.37, 0.37]]).repeat(repeats=len(b_pred), axis=0), device="cpu",
                             verbose=False)

    logs.update({"worst_frame_reward": worst_frame_reward})
    logs.update({'endo_epi-Dice': np.mean([logs["test/dice/LV"], logs["test/dice/epi"]]),
                 "endo_epi-HD": np.mean([logs["test/hd/LV"], logs["test/hd/epi"]])})
    return logs


if __name__ == "__main__":
    compute_metrics = False
    if compute_metrics:
        # with h5py.File("./../../3d_anatomical_reward_LM+ANAT_RNET1+LM_TEMPSCALED_BEST_POLICY.h5", "r") as h5:
        with h5py.File("./../../3d_anatomical_reward_LM+ANAT_RNET1_TEMPSCALED_NEW_NARVAL_POLICY_CARDINAL.h5", "r") as h5:
            # h5_ = h5py.File("./../../3dUNC_MCDropout_final3dunet.h5", "r")
            h5_ = h5py.File("./../../3dUNC_Ensemble_baseline3dunet.h5", "r")
            # h5_ = h5py.File("./../../3dUNC_MCDropout_baseline3dunet.h5", "r")
            print(len(h5.keys()))
            i = 0

            imgs = [np.array(h5[key]['img']) for key in h5.keys()]
            preds = [np.array(h5[key]['pred']) for key in h5.keys()]
            gts = [np.array(h5[key]['gt']) for key in h5.keys()]
            rews = [np.array(h5_[key]['unc_map']) for key in h5.keys()]

            all_logs = process_map(do, imgs, preds, gts, rews, max_workers=12, chunksize=1)

            all_logs = [
                {k: (float(v) if torch.is_tensor(v) else v) for k, v in subj.items()}
                for subj in all_logs
            ]

            print(all_logs)

            ranking_metric = "worst_frame_reward"
            df = pd.DataFrame(all_logs).sort_values(ranking_metric, ascending=False)

            # Parameters
            ranking_metric = "worst_frame_reward"
            oracle_metric = "test/Dice"
            percentages = np.linspace(1.0, 0.1, 10)


            # ---------- helper: compute percentile averages and add suffix to metric names ----------
            def compute_percentile_averages(subjects, sort_metric, percentages, suffix):
                df = pd.DataFrame(subjects).sort_values(sort_metric, ascending=False).reset_index(drop=True)
                n_total = len(df)
                rows = []
                for p in percentages:
                    k = max(1, int(round(p * n_total)))  # at least 1 subject
                    subset = df.head(k)
                    # drop the ranking column (we don't average it here)
                    avgs = subset.drop(columns=[sort_metric]).mean()
                    row = avgs.to_dict()
                    row["percent_retained"] = int(round(p * 100))
                    row["n_subjects"] = k
                    rows.append(row)
                results = pd.DataFrame(rows)
                # rename metric columns to attach suffix (keep percent_retained & n_subjects)
                rename_map = {col: f"{col}{suffix}" for col in results.columns if
                              col not in ["percent_retained", "n_subjects"]}
                return results.rename(columns=rename_map)


            # ---------- compute main and oracle tables ----------
            res_acc = compute_percentile_averages(all_logs, ranking_metric, percentages, suffix="_accRank")
            res_oracle = compute_percentile_averages(all_logs, oracle_metric, percentages, suffix="_oracleDiceRank")

            # ---------- merge side-by-side and inspect column names ----------
            merged = pd.merge(res_acc, res_oracle, on=["percent_retained", "n_subjects"], how="outer")
            merged = merged.sort_values("percent_retained", ascending=False).reset_index(drop=True)

            print("Columns in merged:", merged.columns.tolist())  # quick sanity check

            #merged.to_csv("reward+LM_from_ensemble_pred_metric_data2.csv")
            merged.to_csv("ensemble_metric_data3.csv")
            # merged.to_csv("reward+LM_from_MCD_StrongSeg_metric_data.csv")

    else:
        # merged = pd.read_csv("reward_cardiPolicy_metric_data.csv", index_col=0)
        merged = pd.read_csv("reward_TS+LM_metric_data.csv", index_col=0)
        # merged = pd.read_csv("reward+LM_from_ensemble_pred_metric_data.csv", index_col=0)
        # merged = pd.read_csv("ensemble_metric_data2.csv", index_col=0)
        # merged = pd.read_csv("reward+LM_from_MCD_StrongSeg_metric_data.csv", index_col=0)

    # merged2 = pd.read_csv("MCD_badunet_metric_data.csv", index_col=0)
    merged2 = pd.read_csv("MCD_metric_data.csv", index_col=0)
    # merged2 = pd.read_csv("ensemble_metric_data3.csv", index_col=0)

    # ---------- plotting Dice and Hausdorff (HD) for both methods ----------
    x = merged["percent_retained"]

    fig, ax1 = plt.subplots(figsize=(4, 3), tight_layout=True)
    ax1.set_xlabel("% Subjects Retained", fontsize=12)
    ax1.set_ylabel("Dice (%)", fontsize=12)
    l1 = ax1.plot(x, merged["endo_epi-Dice_accRank"], color='#1C84C4', label="RL4Seg3D Reward Net.")
    l1_ = ax1.plot(x, merged["endo_epi-Dice_oracleDiceRank"], color='black', linestyle='--', label="Dice Score Oracle")
    l1__ = ax1.plot(x, merged2["endo_epi-Dice_accRank"], color='orange', label="MCDropout")
    ax1.invert_xaxis()  # Optional: show decreasing % retained left to right
    ax1.grid(visible=True, linestyle=":", linewidth=0.7, alpha=0.7)
    lines = l1 + l1__ + l1_
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, frameon=False, fontsize=10, loc="upper left")
    plt.savefig("RNet+LM_vs_MCD_GoodSegm_Dice.png")
    plt.close()


    fig3, ax3 = plt.subplots(figsize=(4, 3), tight_layout=True)
    ax3.set_xlabel("% Subjects Retained", fontsize=12)
    ax3.set_ylabel("Anatomical Validity (%)", fontsize=12)
    l2 = ax3.plot(x, merged["test/anat_valid_accRank"], color='#1C84C4', label="RL4Seg3D Reward Net.")
    l2_ = ax3.plot(x, merged["test/anat_valid_oracleDiceRank"], color='black', linestyle='--', label="Dice Score Oracle")
    l2__ = ax3.plot(x, merged2["test/anat_valid_accRank"], color='orange', label="MCDropout")
    # l2___ = ax1.plot(x, merged2["test/anat_valid_oracleDiceRank"], marker="d", linestyle="--", color='gray',
    #                label="Anatomical validation")

    # ax1.tick_params(axis='y', labelcolor=color_dice)
    # ax1.set_ylim(0.92, 1.00)  # Dice and anat_val are usually between 0 and 1
    ax3.invert_xaxis()  # Optional: show decreasing % retained left to right
    ax3.grid(visible=True, linestyle=":", linewidth=0.7, alpha=0.7)
    plt.savefig("RNet+LM_vs_MCD_GoodSegm_AV.png")
    plt.close()

    # Secondary y-axis (right) for Hausdorff distance
    fig2, ax2 = plt.subplots(figsize=(4, 3), tight_layout=True)
    color_hd = "#ff7f0e"
    ax2.set_xlabel("% Subjects Retained", fontsize=12)
    ax2.set_ylabel("Hausdorff Distance (mm)", fontsize=12)
    l3 = ax2.plot(x, merged["endo_epi-HD_accRank"], color='#1C84C4', label="RL4Seg3D Reward Net.")
    l3_ = ax2.plot(x, merged["endo_epi-HD_oracleDiceRank"], color='black', linestyle='--', label="Dice Score Oracle")
    l3__ = ax2.plot(x, merged2["endo_epi-HD_accRank"], color='orange', label="Ensemble")
    # l3___ = ax2.plot(x, merged2["endo_epi-HD_oracleDiceRank"], marker="s", linestyle="-.", color="gray",
    #                label="Hausdorff distance")
    # ax2.tick_params(axis='y', labelcolor=color_hd)
    ax2.invert_xaxis()  # Optional: show decreasing % retained left to right
    ax2.grid(visible=True, linestyle=":", linewidth=0.7, alpha=0.7)
    # Optional: set limits if you want to zoom or unify scales
    # ax2.set_ylim(bottom=0)

    # Combine legends from both axes
    # ax2.legend(l2, ["Reward Network"], frameon=False, fontsize=12, loc="best")
    # ax3.legend(l3, ["Reward Network"], frameon=False, fontsize=12, loc="best")

    # plt.title("Segmentation Metrics vs. Subject Retention", fontsize=14)
    # plt.tight_layout()
    plt.savefig("RNet+LM_vs_MCD_GoodSegm_HD.png")
    plt.close()

    plt.show()