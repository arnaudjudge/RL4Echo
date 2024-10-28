from pathlib import Path
import nibabel as nib
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.ndimage
from matplotlib import pyplot as plt, animation
from matplotlib.colors import LinearSegmentedColormap
from scipy import ndimage
from torchio import Resize, ScalarImage
from skimage import draw

from rl4echo.utils.Metrics import is_anatomically_valid
from vital.metrics.evaluate.attribute import check_temporal_consistency_errors, compute_temporal_consistency_metric
from vital.utils.image.us.measure import EchoMeasure
from vital.data.camus.config import Label


if __name__ == "__main__":
    GT_PATH = '/data/landmarks_cardinal-icardio/gt/'

    GIF_PATH = './valve_gifs/'
    Path(GIF_PATH).mkdir(exist_ok=True)

    count = 0
    for p in Path(GT_PATH).rglob('*di-*.nii.gz'): #di-B77D-C54E-F11F
        #
        # LOAD IMAGES
        #
        print(p)

        gt = nib.load(p).get_fdata()
        img_nifti = nib.load(p.as_posix().replace('gt', 'img'))
        img = img_nifti.get_fdata()

        # landmarks
        lv_points = []
        for i in range(gt.shape[-1]):
            lv_points += [np.asarray(
                EchoMeasure._endo_base(gt[..., i].T, lv_labels=Label.LV, myo_labels=Label.MYO))]
        lv_points = np.asarray(lv_points)

        measures = [1.5, 2.5, 5, 7.5, 10]  # mm
        circles = []
        print(img_nifti.header['pixdim'][1:3])
        for m in measures:
            num_pixels = m / img_nifti.header['pixdim'][1]

            c = np.zeros_like(gt)
            for i in range(len(lv_points)):
                rr, cc = draw.disk(lv_points[i, 0][::-1], num_pixels, shape=None)
                c[rr, cc, i] = 1
                rr, cc = draw.disk(lv_points[i, 1][::-1], num_pixels, shape=None)
                c[rr, cc, i] = 1
            circles += [c]

        #
        # FIGURES
        #
        custom_cmap = LinearSegmentedColormap.from_list("custom", [(0, 0, 0), (0, 1, 0), (1, 0, 0)], N=3)
        fig, axes = plt.subplots(1, len(circles), figsize=(2*len(measures), 2))
        fig.suptitle("Valve distance thresholds")
        bk = []
        seg = []
        scat1 = []
        scat2 = []
        for idx, ax in enumerate(axes):
            bk += [ax.imshow(img[..., 0].T, animated=True, cmap='gray')]
            seg += [ax.imshow(circles[idx][..., 0].T, animated=True,
                                   cmap=custom_cmap,
                                   alpha=0.35,
                                   interpolation='none')]
            scat1 += [ax.scatter(lv_points[0, 0, 1], lv_points[0, 0, 0], marker='x', c='g', s=3)]
            scat2 += [ax.scatter(lv_points[0, 1, 1], lv_points[0, 1, 0], marker='x', c='g', s=3)]
            ax.set_title(f"{measures[idx]} mm")
            ax.axis("off")

        def update(i):
            for idx, s in enumerate(seg):
                s.set_array(circles[idx][..., i].T)
            for sc1 in scat1:
                sc1.set_offsets(lv_points[i, 0][::-1])
            for sc2 in scat2:
                sc2.set_offsets(lv_points[i, 1][::-1])
            for b in bk:
                b.set_array(img[..., i].T)
            return bk + seg + scat1 + scat2
        animation_fig = animation.FuncAnimation(fig, update, frames=img.shape[-1], interval=100, blit=True,
                                                repeat_delay=10, )
        animation_fig.save(f"{GIF_PATH}/{p.stem.split('.')[0]}.gif")
        # plt.show()
        plt.close()

        count += 1
        if count > 10000:
            break