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
        for i in range(len(gt)):
            # print(gt[i].sum())
            # plt.figure()
            # plt.imshow(gt[i].cpu().numpy().T)
            # plt.show()
            if random.random() < 0.5:
                r[i] = torch.ones_like(r[i])
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

                    p_points = np.asarray(
                        EchoMeasure._endo_base(p.T, lv_labels=Label.LV, myo_labels=Label.MYO))
                    a = np.ones_like(r[i].cpu().numpy())
                    b = np.ones_like(r[i].cpu().numpy())

                    lv_points = lv_points[np.argsort(lv_points[:, 1])]
                    p_points = p_points[np.argsort(p_points[:, 1])]

                    d0 = max(1, (np.linalg.norm(lv_points[0] - p_points[0]) / a.shape[0]*75))
                    d1 = max(1, (np.linalg.norm(lv_points[1] - p_points[1]) / b.shape[0]*75))
                    # d0 = min(max(1, np.exp(np.linalg.norm(lv_points[0] - p_points[0]) / b.shape[0] * 50)), 15)
                    # print(d0)
                    # d1 = min(max(1, np.exp(np.linalg.norm(lv_points[1] - p_points[1]) / b.shape[0] * 50)), 15)
                    # print(d1)

                    # if d0 > 1:
                    rr, cc, val = skimage.draw.line_aa(p_points[0, 1], p_points[0, 0], lv_points[0, 1], lv_points[0, 0])
                    a[rr, cc] = 1 - val
                    a = gaussian_filter(a, sigma=d0)
                    a = (a - np.min(a)) / (np.max(a) - np.min(a))
                    # if d1 > 1:
                    rr, cc, val = skimage.draw.line_aa(p_points[1, 1], p_points[1, 0], lv_points[1, 1], lv_points[1, 0])
                    b[rr, cc] = 1 - val
                    b = gaussian_filter(b, sigma=d1)
                    b = (b - np.min(b)) / (np.max(b) - np.min(b))

                    c = np.minimum(a, b)

                    # plt.figure()
                    # plt.imshow(gt[i].cpu().numpy().T)
                    # plt.imshow(p.T, alpha=0.3, cmap='jet')
                    # #
                    # plt.figure()
                    # plt.imshow(gt[i].cpu().numpy().T)
                    # plt.imshow(pred[i].cpu().numpy().T)
                    # plt.scatter(lv_points[0, 1], lv_points[0, 0], c='r')
                    # plt.scatter(lv_points[1, 1], lv_points[1, 0], c='r')
                    # plt.scatter(p_points[0, 1], p_points[0, 0], c='g')
                    # plt.scatter(p_points[1, 1], p_points[1, 0], c='g')

                    # r[i] = torch.clamp(r[i] - (1 - torch.tensor(c).to(r.device)), 0)
                    r[i] = torch.tensor(c).to(r.device) #  * max(0.75, (10 - (1-c.mean())*1000) / 10)
                    # r[i] = r[i] + torch.tensor(c).to(r.device)
                    # r[i] = (r[i] - torch.min(r[i])) / (torch.max(r[i]) - torch.min(r[i]))
                    # plt.figure()
                    # plt.imshow(r[i].cpu().numpy().T)
                    # plt.show()
                except Exception as e:
                    print(e)
                    r[i] = torch.zeros_like(r[i])
        return r

    # @torch.no_grad()
    # def __call__(self, pred, imgs, gt):
    #     stack = torch.stack((imgs.squeeze(1), pred), dim=1)
    #     r = torch.sigmoid(self.net(stack) / self.temp_factor).squeeze(1)
    #     try:
    #         for i in range(len(gt)):
    #             lv_points = np.asarray(EchoMeasure._endo_base(gt[i].cpu().numpy(), lv_labels=Label.LV, myo_labels=Label.MYO))
    #             p_points = np.asarray(EchoMeasure._endo_base(pred[i].cpu().numpy(), lv_labels=Label.LV, myo_labels=Label.MYO))
    #             a = np.ones_like(r[i].cpu().numpy())
    #             # polygon
    #             vertices = np.concatenate([[lv_points[0], p_points[0], p_points[1], lv_points[1]]])
    #             vertices = sorted(vertices, key=lambda row: row[1])
    #             topRow = sorted(vertices[2:], key=lambda row: row[0])
    #             bottomRow = sorted(vertices[0:2], key=lambda row: row[0])
    #             vertices = np.array([topRow[0], topRow[1], bottomRow[1], bottomRow[0]])
    #
    #             rr, cc = polygon(vertices[:, 1], vertices[:, 0], a.shape)
    #             a[rr, cc] = 0.0
    #             a = gaussian_filter(a, sigma=10)
    #
    #             gt_line = self.draw_line_across_array(lv_points[0], lv_points[1], np.zeros_like(r[i].cpu().numpy()))
    #             gt_line[(np.cumsum(gt_line, axis=0) > 0)] = 1
    #             gt_line = torch.tensor(gt_line.T).to(r.device)
    #             # r[i][(r[i] != 1) & (gt_line == 1)] = 1
    #
    #             p_line = self.draw_line_across_array(p_points[0], p_points[1], np.zeros_like(r[i].cpu().numpy()))
    #             p_line[(np.cumsum(p_line, axis=0) > 0)] = 1
    #             p_line = torch.tensor(p_line.T).to(r.device)
    #             # r[i][(pred[i] != 0) & (p_line == 1)] = 0
    #
    #             diff = (gt_line != p_line)
    #             diff_per = diff.sum() / (gt_line.shape[0] * gt_line.shape[1])
    #
    #             # if lines are close enough
    #             if diff_per < 0.1:
    #                 # below gt, remove all not null
    #                 r[i][(r[i] != 1) & (gt_line == 1)] = 1
    #                 r[i][(pred[i] != 0) & (gt_line == 1)] = 0
    #
    #             r[i] = r[i] * torch.tensor(a.T).to(r.device)
    #
    #             plt.figure()
    #             plt.imshow((diff * p_line).cpu().numpy().T)
    #             plt.title(f"{diff_per}")
    #
    #             plt.figure()
    #             plt.imshow(pred[i].cpu().numpy().T)
    #             # plt.imshow(p_line.cpu().numpy().T, alpha=0.3)
    #
    #             plt.figure()
    #             plt.imshow(r[i].cpu().numpy().T)
    #
    #             plt.figure()
    #             plt.imshow(a)
    #             # plt.imshow(p_line.cpu().numpy().T, alpha=0.3)
    #             # plt.imshow(gt_line.cpu().numpy().T, alpha=0.3)
    #             plt.show()
    #
    #     except Exception as e:
    #         print(e)
    #
    #         # plt.figure()
    #         # plt.imshow(pred[i].cpu().numpy().T)
    #         # plt.figure()
    #         # plt.imshow(r[i].cpu().numpy().T)
    #         # plt.scatter(lv_points_seq[i, 0, 0], lv_points_seq[i, 0, 1], c='g')
    #         # plt.scatter(lv_points_seq[i, 1, 0], lv_points_seq[i, 1, 1], c='g')
    #         #
    #         # plt.scatter(r_points[0, 0], r_points[0, 1], c='r')
    #         # plt.scatter(r_points[1, 0], r_points[1, 1], c='r')
    #         # plt.show()
    #     return r

    @staticmethod
    def draw_line_across_array(p1, p2, a):
        m = (p2[1] - p1[1]) / (p2[0] - p1[0] + 1e-6)
        b = p2[1] - m * p2[0]
        rr, cc = skimage.draw.line(int(b), 0, int(m * a.shape[1] + b), a.shape[1]-1)
        rr[rr <= 0] = 0
        rr[rr >= a.shape[1]-1] = a.shape[1]-1
        a[rr, cc] = 1
        return a


class LandmarkWGTReward(Reward):
    @torch.no_grad()
    def __call__(self, pred, imgs, gt): #, pred_unsampled=None):
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

                p_points = np.asarray(
                    EchoMeasure._endo_base(p.T, lv_labels=Label.LV, myo_labels=Label.MYO))
                a = np.ones_like(r[i].cpu().numpy())
                b = np.ones_like(r[i].cpu().numpy())

                lv_points = lv_points[np.argsort(lv_points[:, 1])]
                p_points = p_points[np.argsort(p_points[:, 1])]

                d0 = max(1, (np.linalg.norm(lv_points[0] - p_points[0]) / a.shape[0]*75))
                d1 = max(1, (np.linalg.norm(lv_points[1] - p_points[1]) / b.shape[0]*75))
                # d0 = min(max(1, np.exp(np.linalg.norm(lv_points[0] - p_points[0]) / b.shape[0] * 50)), 15)
                # print(d0)
                # d1 = min(max(1, np.exp(np.linalg.norm(lv_points[1] - p_points[1]) / b.shape[0] * 50)), 15)
                # print(d1)

                rr, cc, val = skimage.draw.line_aa(p_points[0, 1], p_points[0, 0], lv_points[0, 1], lv_points[0, 0])
                a[rr, cc] = 1 - val
                a = gaussian_filter(a, sigma=d0)
                a = (a - np.min(a)) / (np.max(a) - np.min(a))
                rr, cc, val = skimage.draw.line_aa(p_points[1, 1], p_points[1, 0], lv_points[1, 1], lv_points[1, 0])
                b[rr, cc] = 1 - val
                b = gaussian_filter(b, sigma=d1)
                b = (b - np.min(b)) / (np.max(b) - np.min(b))

                c = np.minimum(a, b)

                r[i] = torch.tensor(c).to(r.device)
            except Exception as e:
                print(e)
                r[i] = torch.zeros_like(r[i])
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
