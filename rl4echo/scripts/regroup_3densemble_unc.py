import h5py
import numpy as np
import scipy
import torch
from tqdm import tqdm

if __name__ == "__main__":
    # f1 = h5py.File(f"../../3dUNC_Ensemble_shortened_baseline3dunet.h5", 'a')
    with h5py.File(f"../../3d_anatomical_reward_LM+ANAT_BEST_NARVAL_POLICY_CARDINAL.h5", 'a') as f:
        for case_identifier in tqdm(f.keys()):
            # k = [k for k in f[case_identifier].keys() if k == 'pred']
            # f_ = f[case_identifier]
            # for k_ in k:
            #     del f_[k_]
            # continue
            # if "unc_map" in f[case_identifier].keys():
            #     continue
            # pred_list = [np.asarray(f[case_identifier][k]) for k in f[case_identifier].keys() if "pred_SM" in k]
            gt = np.asarray(f[case_identifier]["gt"]).astype(np.uint8)
            img = np.asarray(f[case_identifier]["img"]).astype(np.uint8)

            # pred_list = np.concatenate(pred_list).astype(float) # S C H W T
            #
            # y_pred = pred_list.mean(axis=0).argmax(axis=0)
            y_pred = np.asarray(f[case_identifier]['pred']).astype(np.uint8)
            # probs = [pred_list[i] for i in range(pred_list.shape[0])]
            # probs = np.stack(probs, axis=0)
            # y_hat = probs.mean(0)
            # base = pred_list.shape[1]
            # uncertainty_map = scipy.stats.entropy(y_hat, axis=0, base=base)
            # uncertainty_map[~np.isfinite(uncertainty_map)] = 0
            uncertainty_map = np.asarray(f[case_identifier]["reward_map"])

            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(3, 6)
            for idx, i in enumerate(np.linspace(0, gt.shape[-1]-1, 6).astype(int)):
                ax[0, idx].imshow(y_pred[..., i].T)
                ax[0, idx].set_axis_off()
            for idx, i in enumerate(np.linspace(0, gt.shape[-1]-1, 6).astype(int)):
                ax[1, idx].imshow((y_pred[..., i] == gt[..., i]).T)
                ax[1, idx].set_axis_off()
            for idx, i in enumerate(np.linspace(0, gt.shape[-1]-1, 6).astype(int)):
                ax[2, idx].imshow(uncertainty_map[..., i].T)
                ax[2, idx].set_axis_off()
            plt.show()


            # f[case_identifier]['img'] = img
            # f[case_identifier]['gt'] = gt
            # f[case_identifier]['pred'] = y_pred
            # f[case_identifier]['accuracy_map'] = (y_pred != gt)
            # f[case_identifier]['unc_map'] = uncertainty_map
