import numpy as np
import skimage.morphology
import torch
import torch.nn.functional as F
#from bdicardio.utils.ransac_utils import ransac_sector_extraction
from scipy import ndimage
from scipy.ndimage import binary_fill_holes
from torch import distributions
from torchmetrics.functional import dice
from vital.models.segmentation.unet import UNet

from rewardnet.unet_heads import UNet_multihead

"""
Reward functions must each have pred, img, gt as input parameters
"""


class Reward:
    @torch.no_grad()
    def __call__(self, pred, imgs, gt, *args, **kwargs):
        raise NotImplementedError


class RewardUnet(Reward):
    def __init__(self, state_dict_path, temp_factor=1):
        self.net = UNet(input_shape=(2, 256, 256), output_shape=(1, 256, 256))
        self.net.load_state_dict(torch.load(state_dict_path))
        self.temp_factor = temp_factor
        if torch.cuda.is_available():
            self.net.cuda()

    @torch.no_grad()
    def __call__(self, pred, imgs, gt):
        stack = torch.stack((imgs.squeeze(1), pred), dim=1)
        return torch.sigmoid(self.net(stack)/self.temp_factor).squeeze(1)


class RewardUnetSigma(Reward):
    def __init__(self, state_dict_path):
        self.net = UNet_multihead(input_shape=(2, 256, 256), output_shape=(1, 256, 256),  sigma_out=True)
        self.net.load_state_dict(torch.load(state_dict_path))
        if torch.cuda.is_available():
            self.net.cuda()

    @torch.no_grad()
    def __call__(self, pred, imgs, gt):
        stack = torch.stack((imgs.squeeze(1), pred), dim=1)
        logits, sigma = self.net(stack)  # (N, C, H, W), (N, C, H, W)
        sigma = F.softplus(sigma)

        distribution = distributions.Normal(logits, torch.exp(sigma))

        x_hat = distribution.rsample((10,))

        mc_expectation = torch.sigmoid(x_hat).mean(dim=0)
        return mc_expectation.squeeze(1)


class AccuracyMap(Reward):
    @torch.no_grad()
    def __call__(self, pred, imgs, gt):
        actions = torch.round(pred)
        assert actions.shape == gt.shape, \
            print(f"Actions shape {actions.shape} vs GT shape {gt.shape}")
        simple = (actions == gt).float()
        simple = simple.mean(dim=(1, 2))
        mean_matrix = torch.zeros_like(pred, dtype=torch.float)
        for i in range(len(pred)):
            mean_matrix[i, ...] = simple[i]
        return mean_matrix


class Accuracy(Reward):
    @torch.no_grad()
    def __call__(self, pred, imgs, gt):
        actions = torch.round(pred)
        assert actions.shape == gt.shape, \
            print(f"Actions shape {actions.shape} vs GT shape {gt.shape}")
        simple = (actions == gt).float()
        return simple.mean(dim=(1, 2, 3), keepdim=True)


class PixelWiseAccuracy(Reward):
    @torch.no_grad()
    def __call__(self, pred, imgs, gt):
        actions = torch.round(pred)
        assert actions.shape == gt.shape, \
            print(f"Actions shape {actions.shape} vs GT shape {gt.shape}")
        return (actions == gt).float()


class Morphological(Reward):
    @torch.no_grad()
    def __call__(self, pred, imgs, gt=None):
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
                blob = (dil == mask)

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
                ransac = np.ones_like(blob)
                try:
                    ransac, *_ = ransac_sector_extraction(lbl, slim_factor=0.01, circle_center_tol=0.5, plot=False)
                    ransac = (ransac == mask)
                except:
                    pass
                # plt.figure()
                # plt.imshow(imgs[i, 0, ...].cpu().numpy())
                # plt.imshow(mask, alpha=0.5)
                # plt.title("Image and predicted mask")
                #
                # plt.figure()
                # plt.imshow(blob)
                # plt.title("Blob")
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
                # plt.imshow((blob & ransac) | mask_in_roi)
                # plt.title(((blob & ransac) | mask_in_roi).mean())
                #
                # plt.show()

                # better than just all & ?
                rew[i, ...] = torch.from_numpy((blob & ransac) | mask_in_roi)

        return rew


class DiceReward(Reward):
    @torch.no_grad()
    def __call__(self, pred, img, gt):
        dice_score = torch.zeros((len(pred), 1, 1, 1)).to(pred.device)
        for i in range(len(dice_score)):
            dice_score[i, ...] = dice(pred[i, ...], gt[i, ...].to(int))
        return dice_score
