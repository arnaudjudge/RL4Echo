import numpy as np
import skimage.morphology
import torch
from bdicardio.utils.ransac_utils import ransac_sector_extraction
from matplotlib import pyplot as plt
from scipy import ndimage
from scipy.ndimage import binary_fill_holes
from torchmetrics.functional import dice

"""
Reward functions must each have pred, img, gt as input parameters
"""


@torch.no_grad()
def accuracy(pred, imgs, gt):
    actions = torch.round(pred)
    assert actions.shape == gt.shape
    simple = (actions == gt).float()
    return simple.mean(dim=(1, 2, 3), keepdim=True)


@torch.no_grad()
def accuracy_map(sampled_pred, deterministic_pred, gt):
    actions = torch.round(sampled_pred)
    assert actions.shape == gt.shape
    simple = (actions == gt).float()
    mean = simple.mean(dim=(1, 2, 3))
    mean_matrix = torch.zeros_like(sampled_pred)
    for i in range(len(sampled_pred)):

        if mean_matrix.mean() > 0.9:
            mask = deterministic_pred[i, 0, ...].cpu().numpy()
            # plt.figure()
            # plt.imshow(mask)

            # Find each blob in the image
            lbl, num = ndimage.label(mask)
            # plt.figure()
            # plt.imshow(lbl)
            # Count the number of elements per label
            count = np.bincount(lbl.flat)
            if not np.any(count[1:]):
                print("No blobs?")
            # Select the largest blob
            maxi = np.argmax(count[1:]) + 1
            # print(maxi)
            # Keep only the other blobs
            lbl[(lbl == maxi)] = 0
            lbl[(lbl != 0)] = 1

            bad_blob_mask = torch.from_numpy(lbl).to(bool).to(mean_matrix.device)
            # plt.figure()
            # plt.imshow((~bad_blob_mask).cpu().numpy())

            out = torch.where(~bad_blob_mask, mean[i], 0)
            #
            # plt.figure()
            # plt.imshow(out.cpu().numpy())
            # plt.show()
            mean_matrix[i, 0, ...] = out
        else:
            mean_matrix[i, ...] = mean[i]

    return mean_matrix


@torch.no_grad()
def morphological(pred, imgs, gt=None):
    rew = torch.zeros_like(pred)
    for i in range(len(rew)):
        mask = pred[i, 0, ...].cpu().numpy()

        # Find each blob in the image
        lbl, num = ndimage.label(mask)
        # Count the number of elements per label
        count = np.bincount(lbl.flat)
        if not np.any(count[1:]):
            rew[i, ...] = 0
        else:
            # Select the largest blob
            maxi = np.argmax(count[1:]) + 1
            # Keep only the other blobs
            lbl[lbl != maxi] = 0

            dil = skimage.morphology.binary_closing(lbl)
            map = (dil == mask)

            # image region of interest (non-black pixels) in the main blob
            im = imgs[i, 0, ...].cpu().numpy()
            im_roi = (im != 0.0)

            # is this better?
            im_roi, num = ndimage.label(im_roi)
            # Count the number of elements per label
            count = np.bincount(im_roi.flat)
            if not np.any(count[1:]):
                print("???")
            # Select the largest blob
            maxi = np.argmax(count[1:]) + 1
            # Keep only the other blobs
            im_roi[im_roi != maxi] = 0
            im_roi = skimage.morphology.binary_closing(im_roi)
            im_roi = binary_fill_holes(im_roi)

            mask_in_roi = (im_roi == mask)

            # ransac
            ransac = np.ones_like(map)
            try:
                ransac, *_ = ransac_sector_extraction(lbl, slim_factor=0.01, circle_center_tol=0.5, plot=False)
                ransac = (ransac == mask)
            except:
                pass
            # plt.figure()
            # plt.imshow(imgs[i, 0, ...].cpu().numpy())
            # plt.imshow(mask, alpha=0.5)
            #
            # plt.figure()
            # plt.imshow(map)
            #
            # plt.figure()
            # plt.imshow(mask_in_roi)
            #
            # plt.figure()
            # plt.imshow(ransac)
            #
            # print(map.mean())
            # print(mask_in_roi.mean())
            # print(ransac.mean())
            #
            # plt.figure()
            # plt.imshow((map & ransac) | mask_in_roi)
            # plt.title(((map & ransac) | mask_in_roi).mean())
            #
            # plt.show()

            #better than just all & ?
            rew[i, ...] = torch.from_numpy((map & ransac) | mask_in_roi)

    return rew


@torch.no_grad()
def dice_reward(pred, img, gt):
    dice_score = torch.zeros((len(pred), 1, 1, 1)).to(pred.device)
    for i in range(len(dice_score)):
        dice_score[i, ...] = dice(pred[i, ...], gt[i, ...].to(int))
    return dice_score

