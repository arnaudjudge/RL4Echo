import os
from multiprocessing import Pool
from pathlib import Path

import nibabel as nib
import numpy as np
from matplotlib import pyplot as plt, animation
from numpy import repeat
from scipy import ndimage
from tqdm import tqdm

from rl4echo.utils.Metrics import is_anatomically_valid, is_anatomically_valid_multiproc
from rl4echo.utils.test_metrics import dice
from vital.data.camus.config import Label
from vital.metrics.camus.anatomical.lv_metrics import LeftVentricleMetrics
from vital.metrics.camus.anatomical.myo_metrics import MyocardiumMetrics
from vital.metrics.evaluate.segmentation import Segmentation2DMetrics


def as_batch(action):
    y_pred_np_as_batch = action.transpose((2, 0, 1))
    return y_pred_np_as_batch


def clean_blobs(action):
    for i in range(action.shape[-1]):
        lbl, num = ndimage.label(action[..., i] != 0)
        # Count the number of elements per label
        count = np.bincount(lbl.flat)
        # Select the largest blob
        maxi = np.argmax(count[1:]) + 1
        # Remove the other blobs
        action[..., i][lbl != maxi] = 0
    return action


if __name__ == "__main__":
    GT_PATH = '/data/icardio/processed/segmentation/'
    SET_PATH = '../../testing_raw_10k16gpu2ndtry/'


    GIF_PATH = './gifs/'
    Path(GIF_PATH).mkdir(exist_ok=True)

    paths = [p for p in Path(SET_PATH).rglob('*.nii.gz')]
    print(len(paths))
    inval = 0
    total = 0
    for p in paths:  #tqdm(paths):
        #
        # LOAD IMAGES
        #
        # print(p)
        pred1 = clean_blobs(nib.load(p).get_fdata())

        gt_p = GT_PATH + p.relative_to(SET_PATH).as_posix()
        if not Path(gt_p).exists():
            continue
        gt = nib.load(gt_p).get_fdata()

        img_p = gt_p.replace(".nii.gz", "_0000.nii.gz").replace('segmentation', 'img')
        img = nib.load(img_p).get_fdata()

        #
        # METRICS
        #
        pred1_b = as_batch(pred1)
        gt_b = as_batch(gt)

        d = dice(pred1, gt, labels=[Label.LV, Label.MYO])
        print(d)
        if d < 0.9:
            continue

        # av = is_anatomically_valid(pred1_b, nib.load(img_p).header['pixdim'][1:3])
        # print(av)
        voxel_spacing = nib.load(img_p).header['pixdim'][1:3]

        def check_lv_metrics(segmentation, voxelspacing):
            segmentation_metrics = Segmentation2DMetrics(
                segmentation,
                [Label.BG, Label.LV, Label.MYO, (Label.LV, Label.MYO), Label.ATRIUM],
                voxelspacing=voxelspacing,
            )
            lv_metrics = LeftVentricleMetrics(segmentation_metrics)
            myo_metrics = MyocardiumMetrics(segmentation_metrics)

            # holes = lv_metrics.count_holes()
            # discon_lv = lv_metrics.count_disconnectivity()
            discon_myo = myo_metrics.count_disconnectivity()

            return discon_myo == 0


        segmentations = [pred1_b[i].T for i in range(len(pred1_b))]
        with Pool(processes=os.cpu_count()) as pool:
            out = list(
                pool.starmap(
                    check_lv_metrics,
                    zip(
                        segmentations,
                        repeat(voxel_spacing, repeats=len(segmentations))
                    )
                )
            )
        print(out)
        # av = is_anatomically_valid_multiproc(pred1_b, )
        # av = np.array(av)
        # total += len(av)
        # inval += (av != 1).sum()

        # for idx, val in enumerate(av):
        #     if val != 1:
        #         plt.figure()
        #         plt.imshow(pred1_b[idx, ...].T)
        # plt.show()
        if not all(out):
            for i in range(len(out)):
                if not out[i]:
                    plt.figure()
                    plt.imshow(pred1_b[i].T)
                    # plt.show()

            print(pred1_b.shape)
            fig, ax = plt.subplots()
            im = ax.imshow(pred1_b[0].T, animated=True)

            def update(i):
                im.set_array(pred1_b[i].T)
                # fig.title(f"{i}")
                ax.set_title(i)
                return im,


            animation_fig = animation.FuncAnimation(fig, update, frames=len(pred1_b), interval=300, blit=False,
                                                    repeat_delay=1000, )

            # animation_fig.save(f"gifs_RNet_anat_metric/{dicom}_{human_aq[-1]}.gif")
            plt.show()
            # plt.close()
    print(inval)
    print(total)