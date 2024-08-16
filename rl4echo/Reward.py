import random

import numpy as np
import skimage.morphology
import torch
import torch.nn.functional as F
#from bdicardio.utils.ransac_utils import ransac_sector_extraction
from scipy import ndimage
from scipy.ndimage import binary_fill_holes, gaussian_filter
from skimage.draw import polygon
from skimage.morphology import convex_hull_image
from torch import distributions
from torchmetrics.functional import dice
from vital.models.segmentation.unet import UNet

from rewardnet.unet_heads import UNet_multihead
from vital.data.camus.config import Label
from vital.utils.image.us.measure import EchoMeasure

from matplotlib import pyplot as plt
from PIL import Image, ImageFilter

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
    def __call__(self, pred, imgs, gt): #, pred_unsampled=None):
        stack = torch.stack((imgs.squeeze(1), pred), dim=1)
        r = torch.sigmoid(self.net(stack) / self.temp_factor).squeeze(1)
        return r


class LandmarkWGTReward(Reward):
    @torch.no_grad()
    def __call__(self, pred, imgs, gt, pred_unsampled=None):
        # GT must be available for this version of the reward,
        # another version with predictions for GT landmarks will exist
        r = torch.ones_like(pred).float() #.type(torch.float32)

        for i in range(len(gt)):
            lv_points = np.asarray(
                EchoMeasure._endo_base(gt[i].cpu().numpy().T, lv_labels=Label.LV, myo_labels=Label.MYO))
            # if pred_unsampled is not None:
            #     p = pred_unsampled[i].cpu().numpy()
            # else:
            #     p = pred[i].cpu().numpy()
            p = pred[i].cpu().numpy()
            try:
                lbl, num = ndimage.measurements.label(p != 0)
                # Count the number of elements per label
                count = np.bincount(lbl.flat)
                # Select the largest blob
                maxi = np.argmax(count[1:]) + 1
                # Remove the other blobs
                p[lbl != maxi] = 0

                # plt.figure()
                # # plt.imshow(gt[i].cpu().numpy().T)
                # plt.imshow(p.T) #, alpha=0.35)

                p1 = ndimage.median_filter(p, size=5)
                #
                # plt.figure()
                # plt.imshow(p1.T)
                # plt.show()
                # remove border artifacts
                lbl, num = ndimage.measurements.label(p1 == 1)
                # Count the number of elements per label
                count = np.bincount(lbl.flat)
                # Select the largest blob
                maxi = np.argmax(count[1:]) + 1
                # Remove the other blobs
                p1_ = p1.copy()
                p1[lbl != maxi] = 0
                p1[p1_ == 2] = 2

                # plt.figure()
                # plt.imshow(p1.T)
                # plt.show()

                p_points = np.asarray(
                    EchoMeasure._endo_base(p1.T, lv_labels=Label.LV, myo_labels=Label.MYO))
                a = np.ones_like(r[i].cpu().numpy())
                b = np.ones_like(r[i].cpu().numpy())

                lv_points = lv_points[np.argsort(lv_points[:, 1])]
                p_points = p_points[np.argsort(p_points[:, 1])]

                d0 = (np.linalg.norm(lv_points[0] - p_points[0]) / a.shape[0]*75)
                d1 = (np.linalg.norm(lv_points[1] - p_points[1]) / b.shape[0]*75)
                # d0 = min(max(1, np.exp(np.linalg.norm(lv_points[0] - p_points[0]) / b.shape[0] * 50)), 15)
                # print(d0)
                # d1 = min(max(1, np.exp(np.linalg.norm(lv_points[1] - p_points[1]) / b.shape[0] * 50)), 15)
                # print(d1)
                if d0 > 0:
                    rr, cc, val = skimage.draw.line_aa(p_points[0, 1], p_points[0, 0], lv_points[0, 1], lv_points[0, 0])
                    a[rr, cc] = 1 - val
                    a = gaussian_filter(a, sigma=d0)
                    a = (a - np.min(a)) / (np.max(a) - np.min(a))
                if d1 > 0:
                    rr, cc, val = skimage.draw.line_aa(p_points[1, 1], p_points[1, 0], lv_points[1, 1], lv_points[1, 0])
                    b[rr, cc] = 1 - val
                    b = gaussian_filter(b, sigma=d1)
                    b = (b - np.min(b)) / (np.max(b) - np.min(b))

                c = np.minimum(a, b)
                c = np.minimum(c, (p1 == pred[i].cpu().numpy()).astype(np.uint8))

                # plt.figure()
                # plt.imshow(np.minimum(c, (p1 == p).astype(np.uint8)).T)
                # plt.show()

                # plt.figure()
                # plt.imshow(gt[i].cpu().numpy().T)
                # plt.imshow(p.T, alpha=0.35)
                # plt.imshow(c.T, alpha=0.35)
                # plt.scatter(p_points[0, 1], p_points[0, 0])
                # plt.scatter(p_points[1, 1], p_points[1, 0])
                # plt.show()

                r[i] = torch.tensor(c).to(r.device)
            except Exception as e:
                print(e)
                # plt.figure()
                # plt.imshow(gt[i].cpu().numpy().T)
                # plt.imshow(p.T, alpha=0.35)
                # # plt.imshow(c.T, alpha=0.35)
                # # plt.scatter(p_points[0, 1], p_points[0, 0])
                # # plt.scatter(p_points[1, 1], p_points[1, 0])
                # plt.show()

                r[i] = torch.zeros_like(r[i])
                # r[i] = torch.ones_like(r[i])*0.5
        return r


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
