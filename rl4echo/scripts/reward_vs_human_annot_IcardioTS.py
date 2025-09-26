import numpy as np
import h5py
import matplotlib.pyplot as plt

dicom_list = [
            "di-0CA9-8FC9-25BB", "di-1F8E-37E5-57ED", "di-1FD4-CB18-6EFC", "di-5A75-9C41-46D7", "di-5BB1-44DE-60BA",
            "di-62CF-F093-A156", "di-63D5-7095-7FB9", "di-A540-DBDD-7C1F", "di-AE68-A41B-5185", "di-C910-B188-A16F",
            "di-EB49-9AE0-F0D0", "di-6CF6-853C-CB97", "di-16B9-402D-037F", "di-46D0-7328-762F", "di-47EB-1516-2456",
            "di-7041-D238-665F", "di-8254-6AD3-C7FC", "di-A984-8D28-57F4", "di-B55F-84D9-6833", "di-C9E0-1668-D365",
            "di-E42E-19EA-16E7"
        ]

if __name__ == "__main__":

    h5 = h5py.File("/home/local/USHERBROOKE/juda2901/dev/RL4Echo/3d_anatomical_reward_LM+ANAT_BEST_NARVAL_TTA_clean.h5", "r")

    reward_mins_pass = []
    reward_mins_fail = []

    for d in h5.keys():
        rew = np.array(h5[d]['reward_map'])
        reward_frame_min = rew.mean(axis=(0, 1, 2)).min()
        if d in dicom_list:
            reward_mins_fail += [reward_frame_min]
        else:
            reward_mins_pass += [reward_frame_min]

    reward_mins_pass = np.asarray(reward_mins_pass)
    reward_mins_fail = np.asarray(reward_mins_fail)

    print(len(reward_mins_pass), reward_mins_pass.mean(), reward_mins_pass.min(), reward_mins_pass.max())
    print(len(reward_mins_fail), reward_mins_fail.mean(), reward_mins_fail.min(), reward_mins_fail.max())

    plt.figure()
    plt.scatter(np.ones_like(reward_mins_pass), reward_mins_pass, c='g')
    plt.scatter(np.zeros_like(reward_mins_fail), reward_mins_fail, c='r')

    plt.show()
