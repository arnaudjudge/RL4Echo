import json
import random

from torch.utils.data import IterableDataset

import pandas as pd
from replaybuffer import ReplayBuffer
from typing import Tuple

from typing import Optional, Tuple

import nibabel as nib

import numpy as np
from PIL import Image
from PIL.Image import Resampling


def resize_image(image: np.ndarray, size: Tuple[int, int], resample: Resampling = Resampling.NEAREST) -> np.ndarray:
    """Resizes the image to the specified dimensions.

    Args:
        image: ([N], H, W), Input image to resize. Must be in a format supported by PIL.
        size: Width (W') and height (H') dimensions of the resized image to output.
        resample: Resampling filter to use.

    Returns:
        ([N], H', W'), Input image resized to the specified dimensions.
    """
    resized_image = np.array(Image.fromarray(image).resize(size, resample=resample))
    return resized_image


class RLDataset(IterableDataset):
    """
    Iterable Dataset containing the ReplayBuffer
    which will be updated with new experiences during training
    Args:
        buffer: replay buffer
        sample_size: number of experiences to sample at a time
    """

    def __init__(self, buffer: ReplayBuffer, data_path, csv_file, img_size=(256, 256), sample_size: int = 1) -> None:
        self.buffer = buffer
        self.sample_size = sample_size

        self.data_path = data_path
        self.img_size = img_size

        self.df = pd.read_csv(csv_file, index_col=0, low_memory=False)
        self.df = self.df[(self.df['passed'] == True) & (self.df['relative_path'].notna())]

    def __iter__(self) -> Tuple:
        imgs, masks, rewards, log_probs, gt_masks = self.buffer.sample(self.sample_size)
        for i in range(len(rewards)):
            # from matplotlib import pyplot as plt
            # plt.figure()
            # plt.title(f'Reward {rewards[i]}')
            # plt.imshow(masks[i, 0, :, :].T)
            # plt.show()

            yield imgs[i], masks[i], rewards[i], log_probs[i], gt_masks[i]

    def get_new_image(self):
        idx = random.randint(0, len(self.df)-1)

        path_dict = json.loads(self.df.iloc[idx]['relative_path'].replace("\'", "\""))

        img = nib.load(self.data_path + '/raw/' + path_dict['raw']).get_fdata().sum(axis=2)
        img = np.expand_dims(resize_image(img, size=self.img_size), 0)
        img = (img - img.min()) / (img.max() - img.min())

        mask = nib.load(self.data_path + '/mask/' + path_dict['mask']).get_fdata()[:, :, 0]
        mask = resize_image(mask, size=self.img_size)

        return img, mask