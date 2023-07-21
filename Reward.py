import numpy as np
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

        if mean_matrix.mean() > 0.7:
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
def dice_reward(pred, gt):
    dice_score = torch.zeros((len(pred), 1, 1, 1)).to(pred.device)
    for i in range(len(dice_score)):
        dice_score[i, ...] = dice(pred[i, ...], gt[i, ...].to(int))
    return dice_score

