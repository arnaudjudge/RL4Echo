from copy import copy, deepcopy
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import skimage.morphology
import torch
import torch.nn.functional as F
from monai.data import MetaTensor
#from bdicardio.utils.ransac_utils import ransac_sector_extraction
from scipy import ndimage
from scipy.ndimage import binary_fill_holes
from torch import distributions, Tensor
from torchmetrics.functional import dice

from patchless_nnunet.utils.inferers import SlidingWindowInferer
from patchless_nnunet.utils.softmax import softmax_helper
from vital.models.segmentation.unet import UNet

from rl4echo.rewardnet.unet_heads import UNet_multihead

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


class RewardUnet3D(Reward):
    def __init__(self, net, state_dict_path, temp_factor=1):
        self.net = net
        self.net.load_state_dict(torch.load(state_dict_path))
        self.temp_factor = temp_factor

    @torch.no_grad()
    def __call__(self, pred, imgs, gt):
        stack = torch.stack((imgs.squeeze(1), pred), dim=1)
        # return as list for code suiting multireward
        return [torch.sigmoid(self.net(stack)/self.temp_factor).squeeze(1)]

    @torch.no_grad()
    def predict_full_sequence(self, pred, imgs, gt):
        stack = torch.stack((imgs.squeeze(1), pred), dim=1)

        self.patch_size = list([stack.shape[-3], stack.shape[-2], 4])
        self.inferer.roi_size = self.patch_size
        return [torch.sigmoid(self.predict(stack, apply_softmax=False)).squeeze(1)]

    def prepare_for_full_sequence(self, batch_size=1) -> None:  # noqa: D102
        sw_batch_size = batch_size

        self.inferer = SlidingWindowInferer(
            roi_size=self.net.patch_size,
            sw_batch_size=sw_batch_size,
            overlap=0.5,
            mode='gaussian',
            cache_roi_weight_map=True,
        )

    def predict(
        self, image: Union[Tensor, MetaTensor], apply_softmax: bool = True
    ) -> Union[Tensor, MetaTensor]:
        """Predict 2D/3D images with sliding window inference.

        Args:
            image: Image to predict.
            apply_softmax: Whether to apply softmax to prediction.

        Returns:
            Aggregated prediction over all sliding windows.

        Raises:
            NotImplementedError: If the patch shape is not 2D nor 3D.
            ValueError: If 3D patch is requested to predict 2D images.
        """
        if len(image.shape) == 5:
            if len(self.net.patch_size) == 3:
                # Pad the last dimension to avoid 3D segmentation border artifacts
                pad_len = 6 if image.shape[-1] > 6 else image.shape[-1] - 1
                image = F.pad(image, (pad_len, pad_len, 0, 0, 0, 0), mode="reflect")
                pred = self.predict_3D_3Dconv_tiled(image, apply_softmax)
                # Inverse the padding after prediction
                return pred[..., pad_len:-pad_len]
            else:
                raise ValueError("Check your patch size. You dummy.")
        if len(image.shape) == 4:
            raise ValueError("No 2D images here. You dummy.")

    def predict_3D_3Dconv_tiled(
        self, image: Union[Tensor, MetaTensor], apply_softmax: bool = True
    ) -> Union[Tensor, MetaTensor]:
        """Predict 3D image with 3D model.

        Args:
            image: Image to predict.
            apply_softmax: Whether to apply softmax to prediction.

        Returns:
            Aggregated prediction over all sliding windows.

        Raises:
            ValueError: If image is not 3D.
        """
        if not len(image.shape) == 5:
            raise ValueError("image must be (b, c, w, h, d)")

        if apply_softmax:
            return softmax_helper(self.sliding_window_inference(image))
        else:
            return self.sliding_window_inference(image)

    def sliding_window_inference(
        self, image: Union[Tensor, MetaTensor],
    ) -> Union[Tensor, MetaTensor]:
        """Inference using sliding window.

        Args:
            image: Image to predict.

        Returns:
            Predicted logits.
        """
        return self.inferer(
            inputs=image,
            network=self.net,
        )


class MultiRewardUnet3D(Reward):
    def __init__(self, net, state_dict_paths, temp_factor=1):
        self.nets = []
        for path in state_dict_paths:
            n = deepcopy(net)
            n.load_state_dict(torch.load(path))
            self.nets += [n]
        # self.nets = net
        # self.net2 = deepcopy(net)
        # self.net.load_state_dict(torch.load(state_dict_path))
        self.temp_factor = temp_factor
        # self.net2.load_state_dict(torch.load("/home/local/USHERBROOKE/juda2901/dev/RL4Echo/reward_net_3d_landmarks.ckpt"))
        # self.net2.cuda()

    @torch.no_grad()
    def __call__(self, pred, imgs, gt):
        stack = torch.stack((imgs.squeeze(1), pred), dim=1)
        r = []
        for net in self.nets:
            r += [torch.sigmoid(net(stack)/self.temp_factor).squeeze(1)]
        return r #torch.minimum(r1, r2)


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
