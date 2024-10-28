from pathlib import Path
from typing import Union, Tuple

import nibabel as nib
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.ndimage
from matplotlib import pyplot as plt, animation
from matplotlib.colors import LinearSegmentedColormap
from scipy import ndimage
from scipy.signal import find_peaks
from torchio import Resize, ScalarImage
from skimage import draw

from rl4echo.utils.Metrics import is_anatomically_valid
from rl4echo.utils.cardiac_cycle_utils import estimate_num_cycles
from vital.metrics.evaluate.attribute import check_temporal_consistency_errors, compute_temporal_consistency_metric
from vital.utils.image.us.measure import EchoMeasure
from vital.data.camus.config import Label


if __name__ == "__main__":
    GT_PATH = '/data/landmarks_cardinal-icardio/gt/'

    GIF_PATH = './cycle_gifs/'
    Path(GIF_PATH).mkdir(exist_ok=True)

    count = 0
    for p in Path(GT_PATH).rglob('*di-*.gz'):
        print(p)

        gt = nib.load(p).get_fdata()
        img_nifti = nib.load(p.as_posix().replace('gt', 'img'))
        img = img_nifti.get_fdata()

        lv_area = EchoMeasure.structure_area(gt.transpose((2, 0, 1)), labels=1)

        n = estimate_num_cycles(lv_area, plot=True)

        count += 1
        if count > 15:
            break

