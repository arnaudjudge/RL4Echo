import h5py
import numpy as np
import torch
from rl4echo.utils.test_metrics import full_test_metrics
from collections import defaultdict
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Define thresholds you want to test
    thresholds = [0.0, 0.975, 0.98, 0.985, 0.99]

    # Store results per threshold
    metrics_by_threshold = {t: defaultdict(list) for t in thresholds}
    retention_by_threshold = {t: 0 for t in thresholds}
    total_cases = 0

    with h5py.File("./../../3d_anatomical_reward_LM+ANAT_BEST_NARVAL_TTA_ANATTEST_clean.h5", "r") as h5:
        print(len(h5.keys()))
        i = 0
        for key in h5.keys():
            b_img = np.array(h5[key]['img']).transpose((2, 0, 1))
            b_pred = np.array(h5[key]['pred']).transpose((2, 0, 1))
            b_gt = np.array(h5[key]['gt']).transpose((2, 0, 1))

            b_reward = np.array(h5[key]['reward_map']).transpose((2, 0, 1))

            worst_frame_reward = b_reward.mean(axis=(1, 2)).min()

            # for i in range(len(b_pred)):
            #     plt.figure
            #     plt.imshow(b_pred[i])
            #     plt.show()

            logs = full_test_metrics(b_pred, b_gt, np.asarray([[0.37, 0.37]]).repeat(repeats=len(b_pred), axis=0), device="cpu",
                                     verbose=False)

            for t in thresholds:
                        if worst_frame_reward >= t:
                            for k, v in logs.items():
                                if isinstance(v, torch.Tensor):
                                    v = v.item()
                                metrics_by_threshold[t][k].append(v)
                            retention_by_threshold[t] += 1
            i += 1
            print(i)
            # if i > 10:
            #     break

    import matplotlib.pyplot as plt

    # Plot Dice mean vs threshold
    dice_means = [np.mean(metrics_by_threshold[t]['test/Dice']) for t in thresholds]
    dice_stds = [np.std(metrics_by_threshold[t]['test/Dice']) for t in thresholds]
    hd_means = [np.mean(metrics_by_threshold[t]['test/Hausdorff']) for t in thresholds]

    # plt.figure(figsize=(10, 5))
    #
    # # Dice plot
    # plt.subplot(1, 2, 1)
    # plt.errorbar(thresholds, dice_means, yerr=dice_stds, fmt='-o', label='Dice')
    # plt.ylabel('Mean Dice')
    # plt.xlabel('Worst Frame Reward Threshold')
    # plt.title('Segmentation Dice vs. Confidence Threshold')
    # plt.grid(True)
    #
    # # Retention plot
    # plt.subplot(1, 2, 2)
    # retention_percents = [100 * retention_by_threshold[t] / (total_cases + 1e-5 )  for t in thresholds]
    # plt.plot(thresholds, retention_percents, '-o', color='green')
    # plt.ylabel('% of Sequences Retained')
    # plt.xlabel('Worst Frame Reward Threshold')
    # plt.title('Retention Rate vs. Threshold')
    # plt.grid(True)
    #
    # plt.tight_layout()
    # plt.show()

    # print("\nSummary:")
    # print(f"{'Threshold':>10} | {'Retained (%)':>12} | {'Mean Dice':>10} | {'Mean HD':>8}")
    # print("-" * 50)
    # for t in thresholds:
    #     r = 100 * retention_by_threshold[t] / (total_cases + 1e-5 )
    #     dice = np.mean(metrics_by_threshold[t]['test/Dice'])
    #     hd = np.mean(metrics_by_threshold[t]['test/Hausdorff'])
    #     print(f"{t:>10.3f} | {r:12.1f} | {dice:10.3f} | {hd:8.3f}")

    # Dynamically get all metric names from any one non-empty threshold
    metric_names = set()
    for th_dict in metrics_by_threshold.values():
        for name in th_dict.keys():
            if "_L" not in name and "_R" not in name:
                metric_names.add(name)
    metric_names = sorted(metric_names)

    max_len = max(len(name) for name in metric_names)
    col_width = max(max_len, 12)

    # Print header
    header = f"{'Threshold':>10} | {'Retained (%)':>12} | " + " | ".join(f"{m:>{col_width}}" for m in metric_names)
    print("\nSummary:")
    print(header)
    print("-" * len(header))

    # Print each row
    for t in thresholds:
        retained = retention_by_threshold[t]
        r = retained
        row = f"{t:10.3f} | {r:12.1f} | "
        for m in metric_names:
            vals = metrics_by_threshold[t].get(m, [])
            if len(vals) == 0:
                row += f"{'N/A':>{col_width}} | "
            else:
                mean_val = np.mean(vals)
                row += f"{mean_val:{col_width}.3f} | "
        print(row)