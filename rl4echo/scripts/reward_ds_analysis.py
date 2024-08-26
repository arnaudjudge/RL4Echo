from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt


if __name__ == "__main__":
    data_path = '/data/rl_logs/run_2_batchsize_increase/'
    img_list = []

    for im_file in Path(f"{data_path}").rglob("**/images/*.nii.gz"):
        img_list += [im_file.as_posix()]

    # dices = []
    # for i, path in enumerate(tqdm(img_list, total=len(img_list))):
    #     img = nib.load(path).get_fdata()
    #     gt = nib.load(path.replace("images", "gt")).get_fdata()
    #     pred = nib.load(path.replace("images", "pred")).get_fdata()
    #
    #     dices.append(dice(pred, gt, labels=(0, 1, 2), exclude_bg=True, all_classes=False))
    #
    # dices = np.asarray(dices)
    # np.save("dices.npy", dices)
    dices = np.load('dices.npy')

    print(dices.mean())
    plt.figure()
    plt.hist(dices, bins=100)
    plt.show()




    dices = dices[dices.nonzero()]
    print(dices.mean())


    plt.figure()
    plt.hist(dices, bins=100)
    plt.xlabel("Dice")
    plt.ylabel("Number of examples")
    plt.title("Dice histogram between valid/invalid masks")
    plt.show()
