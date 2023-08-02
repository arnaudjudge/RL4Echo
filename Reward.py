import numpy as np
import skimage.morphology
import torch
from matplotlib import pyplot as plt
from scipy import ndimage
from torchmetrics.functional import dice


@torch.no_grad()
def accuracy_reward(pred, gt):
    actions = torch.round(pred)
    assert actions.shape == gt.shape
    simple = (actions == gt).float()
    return simple.mean(dim=(1, 2, 3), keepdim=True)


@torch.no_grad()
def accuracy_reward_map(sampled_pred, deterministic_pred, gt):
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
def morphological(pred, imgs):
    rew = torch.zeros_like(pred)
    for i in range(len(rew)):
        mask = pred[i, 0, ...].cpu().numpy()
        # plt.figure()
        # plt.imshow(mask)

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
            # plt.figure()
            # plt.imshow(dil)
            # #
            # print(dil.shape)
            # print(gt.shape)

            map = (dil == mask)
            # plt.figure()
            # plt.imshow(map)

            im = imgs[i, 0, 0, ...].cpu().numpy()
            im_roi = (im != 0.0)
            # plt.figure()
            # plt.imshow(im_roi)

            mask_in_roi = (im_roi == mask)


            # plt.figure()
            # plt.imshow(map & mask_in_roi)
            # plt.title((map & mask_in_roi).mean())
            #
            # plt.show()
            rew[i, ...] = torch.from_numpy(map & mask_in_roi)

    return rew


@torch.no_grad()
def dice_reward(pred, gt):
    dice_score = torch.zeros((len(pred), 1, 1, 1)).to(pred.device)
    for i in range(len(dice_score)):
        dice_score[i, ...] = dice(pred[i, ...], gt[i, ...].to(int))
    return dice_score

