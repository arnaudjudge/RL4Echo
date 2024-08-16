import numpy as np
import matplotlib.pyplot as plt
import h5py


if __name__ == "__main__":

    with h5py.File("LM.h5", "r") as h5:
        dicoms = list(set([k.split("_", 1)[1] for k in h5.keys()]))
        count = 0
        for d in dicoms:
            initial = h5["INITIAL_" + d]
            lm = h5["Combined_" + d]
            if (np.array(initial["mae"]) > 10).any() or(np.array(lm["mae"]) > 10).any():
                count += 1
                fig, ax = plt.subplots(1, 2)
                ax[0].imshow(np.array(initial["img"]).T, cmap='gray')
                ax[0].imshow(np.array(initial["pred"]).T, alpha=0.35, cmap='jet')
                ax[0].scatter(np.array(initial["p_points"])[0, 1], np.array(initial["p_points"])[0, 0], marker='x', c='r')
                ax[0].scatter(np.array(initial["p_points"])[1, 1], np.array(initial["p_points"])[1, 0], marker='x', c='r')
                ax[0].scatter(np.array(initial["lv_points"])[0, 1], np.array(initial["lv_points"])[0, 0], marker='x', c='g')
                ax[0].scatter(np.array(initial["lv_points"])[1, 1], np.array(initial["lv_points"])[1, 0], marker='x', c='g')
                ax[0].set_title(f"INITIAL {np.array(initial['mae'])}")

                ax[1].imshow(np.array(lm["img"]).T, cmap='gray')
                ax[1].imshow(np.array(lm["pred"]).T, alpha=0.35, cmap='jet')
                ax[1].scatter(np.array(lm["p_points"])[0, 1], np.array(lm["p_points"])[0, 0], marker='x', c='r')
                ax[1].scatter(np.array(lm["p_points"])[1, 1], np.array(lm["p_points"])[1, 0], marker='x', c='r')
                ax[1].scatter(np.array(lm["lv_points"])[0, 1], np.array(lm["lv_points"])[0, 0], marker='x', c='g')
                ax[1].scatter(np.array(lm["lv_points"])[1, 1], np.array(lm["lv_points"])[1, 0], marker='x', c='g')
                ax[1].set_title(f"LM {np.array(lm['mae'])}")
                plt.show()
        print(count)