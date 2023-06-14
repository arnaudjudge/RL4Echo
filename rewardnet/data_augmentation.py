import json
import random

import cv2
import pandas as pd
import pytorch_lightning as pl
import skimage
import scipy
import torch
from numpy.random import default_rng
from typing import Optional, Tuple

import nibabel as nib
import math
import numpy as np
import json
from PIL import Image
from PIL.Image import Resampling

from tqdm import tqdm
from matplotlib import pyplot as plt


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


def create_random_blobs(size=(256,256), seed=None):
    # https://stackoverflow.com/questions/71865493/is-it-possible-to-create-a-random-shape-on-an-image-in-python
    # seedval = 0
    rng = default_rng(seed=seed)

    # create random noise image
    noise = rng.integers(0, 255, size, np.uint8, True)

    # blur the noise image to control the size
    blur = cv2.GaussianBlur(noise, (0, 0), sigmaX=15, sigmaY=15, borderType=cv2.BORDER_DEFAULT)

    # stretch the blurred image to full dynamic range
    stretch = skimage.exposure.rescale_intensity(blur, in_range='image', out_range=(0, 255)).astype(np.uint8)

    # threshold stretched image to control the size
    thresh = cv2.threshold(stretch, 175, 255, cv2.THRESH_BINARY)[1]

    # apply morphology open and close to smooth out shapes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    result = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)

    # other option
    # https://github.com/scikit-image/scikit-image/blob/main/skimage/data/_binary_blobs.py
    # from skimage import data
    # b = data.binary_blobs(length=256, blob_size_fraction=0.8, volume_fraction=0.2)

    return result


def squish_image(image):
    choice = random.choice([0, 1])

    # only have one axis at a time
    if choice == 0:
        fx = 1
        fy = random.uniform(0.8, 0.9)
    elif choice == 1:
        fx = random.uniform(0.8, 0.9)
        fy = 1

    # https://stackoverflow.com/questions/44401085/how-to-scale-the-image-along-x-and-y-axis-and-crop-to-a-specific-height-and-widt
    # scaled_img = cv2.resize(image, None, fx=fx, fy=fy)
    h, w = image.shape[:2]  # get h, w of image

    scaled_img = cv2.resize(image, None, fx=fx, fy=fy)  # scale image

    sh, sw = scaled_img.shape[:2]  # get h, w of cropped, scaled image
    center_y = int(h / 2 - sh / 2)
    center_x = int(w / 2 - sw / 2)
    padded_scaled = np.zeros(image.shape, dtype=np.uint8)
    padded_scaled[center_y:center_y + sh, center_x:center_x + sw] = scaled_img

    return padded_scaled


def stretch_image(image):
    choice = random.choice([0, 1])

    # only have one axis at a time
    if choice == 0:
        fx = 1
        fy = random.uniform(1.2, 1.5)
    elif choice == 1:
        fx = random.uniform(1.2, 1.5)
        fy = 1

    # https://stackoverflow.com/questions/44401085/how-to-scale-the-image-along-x-and-y-axis-and-crop-to-a-specific-height-and-widt
    # scaled_img = cv2.resize(image, None, fx=fx, fy=fy)
    h, w = image.shape[:2]  # get h, w of image

    scaled_img = cv2.resize(image, None, fx=fx, fy=fy)  # scale image
    sh, sw = scaled_img.shape[:2]  # get h, w of scaled image
    center_y = int(sh / 2 - h / 2)
    center_x = int(sw / 2 - w / 2)
    cropped = scaled_img[center_y:center_y + h, center_x:center_x + w]

    return cropped


def rotate_image(image):
    # get random angle
    angle = random.randint(15, 30)
    neg = random.choice([-1, 1])
    angle = angle * neg
    # rotate the image
    return scipy.ndimage.rotate(image, angle, reshape=False, order=0)


def warp(img):
    img_output = np.zeros(img.shape, dtype=img.dtype)
    xfactor = random.randint(4, 8)
    yfactor = random.randint(4, 8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            offset_x = int(xfactor * math.sin(2 * 3.14 * i / 150))
            offset_y = int(yfactor * math.sin(2 * 3.14 * j / 150))
            if i + offset_y < img.shape[0]:
                img_output[i, j] = img[(i + offset_y) % img.shape[0], (j+offset_x)%img.shape[1]]
            else:
                img_output[i, j] = 0

    return img_output


if __name__ == "__main__":
    csv_path = '/home/local/USHERBROOKE/juda2901/dev/data/icardio/processed/processed.csv'
    data_path = '/home/local/USHERBROOKE/juda2901/dev/data/icardio/processed/'
    img_size = (256, 256)

    idx = 210

    df = pd.read_csv(csv_path, index_col=0, low_memory=False)
    df = df[(df['passed'] == True) & (df['relative_path'].notna())]

    path_dict = json.loads(df.iloc[idx]['relative_path'].replace("\'", "\""))

    # img = nib.load(data_path + '/raw/' + path_dict['raw']).get_fdata().sum(axis=2)
    # img = np.expand_dims(resize_image(img, size=img_size), 0)
    # img = (img - img.min()) / (img.max() - img.min())
    #
    # mask = nib.load(data_path + '/mask/' + path_dict['mask']).get_fdata()[:, :, 0]
    # mask = resize_image(mask, size=img_size)
    #
    # plt.figure()
    # plt.imshow(mask.astype(np.uint8))
    # plt.show()
    #
    # plt.figure()
    # plt.imshow(warp(mask.astype(np.uint8)))
    # plt.show()

    # result = create_random_blobs(size=img_size)
    # plt.figure()
    # plt.imshow(result)
    # plt.show()
    #
    # # and with holes
    # a = mask.astype(np.uint8) & ~result
    # plt.figure()
    # plt.imshow(a)
    # plt.show()
    #
    # #or with extra blobs
    # b = mask.astype(bool) | result.astype(bool)
    # plt.figure()
    # plt.imshow(b.astype(np.uint8))
    # plt.show()
    #
    # rot = rotate_image(mask)
    # plt.figure()
    # plt.imshow(rot.astype(np.uint8))
    # plt.show()
    #
    # squish = squish_image(stretch_image(mask))
    # plt.figure()
    # plt.imshow(squish)
    # plt.show()


    label_dict = {}
    total_mod = 0
    out_dir = './dataset_augmented_reward'

    import pathlib, os
    for f in pathlib.Path(f"{out_dir}/").rglob("*.nii.gz"):
        os.remove(f)

    for i in tqdm(range(25000)):
        path_dict = json.loads(df.iloc[i]['relative_path'].replace("\'", "\""))

        img = nib.load(data_path + '/raw/' + path_dict['raw']).get_fdata().sum(axis=2)
        img = np.expand_dims(resize_image(img, size=img_size), 0)
        img = (img - img.min()) / (img.max() - img.min())

        mask = nib.load(data_path + '/mask/' + path_dict['mask']).get_fdata()[:, :, 0]
        mask = resize_image(mask, size=img_size)
        modified = 0

        if random.random() > 0.975:
            mask = rotate_image(mask)
            modified = 1
        if random.random() > 0.975:
            mask = squish_image(mask)
            modified = 1
        if random.random() > 0.975:
            mask = stretch_image(mask)
            modified = 1
        if random.random() > 0.85:
            mask = warp(mask)
            modified = 1
        if random.random() > 0.7:
            blobs = create_random_blobs(img_size)
            if random.random() > 0.5:
                mask = mask.astype(np.uint8) & ~blobs
            else:
                mask = mask.astype(bool) | blobs.astype(bool)
                mask = mask.astype(np.uint8)
            modified = 1

        filename = f"{i}_{modified}.nii.gz"
        affine = np.diag(np.asarray([1, 1, 1, 0]))
        hdr = nib.Nifti1Header()

        mask_img = nib.Nifti1Image(np.expand_dims(mask, 0), affine, hdr)
        mask_img.to_filename(f"./{out_dir}/mask/{i}_{modified}.nii.gz")

        nifti_img = nib.Nifti1Image(img, affine, hdr)
        nifti_img.to_filename(f"./{out_dir}/images/{i}_{modified}.nii.gz")

        label_dict[filename] = modified
        total_mod += modified

    with open(f"{out_dir}/labels.json", 'w') as fp:
        json.dump(label_dict, fp)

    print(label_dict)
    print(total_mod)
