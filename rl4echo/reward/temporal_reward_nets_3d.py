from copy import deepcopy
from typing import Union, List
from pathlib import Path

import numpy as np
import scipy.ndimage
import torch
import torch.nn.functional as F
from monai.data import MetaTensor
from torch import Tensor

from patchless_nnunet.utils.inferers import SlidingWindowInferer
from rl4echo.reward.generic_reward import Reward
from rl4echo.utils.temporal_metrics import get_temporal_consistencies

"""
Reward functions must each have pred, img, gt as input parameters
"""


class TemporalRewardUnets3D(Reward):
    def __init__(self, net, state_dict_paths, temp_factor=1):
        self.nets = {}
        for name, path in state_dict_paths.items():
            n = deepcopy(net)
            if path and Path(path).exists():
                n.load_state_dict(torch.load(path))
            else:
                print(f"BEWARE! You don't have a valid path for this reward net: {name}"
                      "Ignore if using full checkpoint file")
            n.eval()
            self.nets.update({name: n})
        self.temp_factor = temp_factor

    @torch.no_grad()
    def __call__(self, pred, imgs, gt):
        stack = torch.stack((imgs.squeeze(1), pred), dim=1)
        r = []
        for net in self.get_nets():
            with torch.cuda.amp.autocast(enabled=False):  # force float32
                out = net(stack)
            rew = torch.sigmoid(out / self.temp_factor).squeeze(1)
            for i in range(rew.shape[0]):
                for j in range(rew.shape[-1]):
                    rew[i, ..., j] = rew[i, ..., j] - rew[i, ..., j].min()
                    rew[i, ..., j] = rew[i, ..., j] / rew[i, ..., j].max()
            # r += [torch.sigmoid(net(stack)/self.temp_factor).squeeze(1)]
            r += [rew]
        # r[1][r[1] < 0.9] = 0
        r_backp = r.copy()
        r = [torch.minimum(r[0], r[1])]

        for i in range(len(pred)):
            pred_as_b = pred[i].cpu().numpy().transpose((2, 1, 0))
            temp_constistencies, measures_1d = get_temporal_consistencies(pred_as_b, skip_measurement_metrics=True)

            # print(temp_constistencies)
            temp_constistencies = scipy.ndimage.gaussian_filter1d(np.array(list(temp_constistencies.values())).astype(np.float), 1.1, axis=1)

            # print(temp_constistencies)

            temp_constistencies = torch.tensor(temp_constistencies, device=r[0].device).sum(dim=0)
            # print(temp_constistencies)
            tc_penalty = torch.ones(len(temp_constistencies), device=r[0].device) + temp_constistencies
            print(tc_penalty)



            # r[0][i] = r[0][i] * tc_penalty

            # print(temp_constistencies)
            # print(measures_1d)
            import matplotlib.pyplot as plt
            from matplotlib.colors import LinearSegmentedColormap
            f1, ax1 = plt.subplots(2, 2, figsize=(8, 8))
            custom_cmap = LinearSegmentedColormap.from_list("custom", [(0, 0, 0), (0, 1, 0), (1, 0, 0)], N=3)
            ax1[0, 0].imshow(imgs[i, 0, ..., 0].cpu().numpy().T, cmap='gray')
            ax1[0, 0].imshow(pred_as_b[0, ...], alpha=0.3, cmap=custom_cmap, interpolation="none")
            # ax1[1].imshow(imgs[i, 0, ..., 1].cpu().numpy().T, cmap='gray')
            # ax1[1].imshow(pred_as_b[1, ...], alpha=0.3)
            # ax1[2].imshow(imgs[i, 0, ..., 2].cpu().numpy().T, cmap='gray')
            # ax1[2].imshow(pred_as_b[2, ...], alpha=0.3)
            # ax1[3].imshow(imgs[i, 0, ..., 3].cpu().numpy().T, cmap='gray')
            # ax1[3].imshow(pred_as_b[3, ...], alpha=0.3)

            # f, ax = plt.subplots(1, 4)
            rew = r[0]
            # ax[0].imshow(rew[i, ..., 0].cpu().numpy().T, cmap='gray', vmin=0, vmax=1)
            ax1[0, 1].imshow(r_backp[0][i, ..., 0].cpu().numpy().T, cmap='gray', vmin=0, vmax=1)
            ax1[1, 0].imshow(r_backp[1][i, ..., 0].cpu().numpy().T, cmap='gray', vmin=0, vmax=1)
            # ax[2].imshow(rew[i, ..., 2].cpu().numpy().T, cmap='gray', vmin=0, vmax=1)
            # ax[3].imshow(rew[i, ..., 3].cpu().numpy().T, cmap='gray', vmin=0, vmax=1)

            rew = 1 - r[0].cpu().numpy()

            for j in range(rew.shape[-1]):
                frame_penalty = tc_penalty.cpu().numpy()[i]
                # frame_penalty = 1.05 # force for fig
                if frame_penalty != 1:
                    rew[i, ..., j] = scipy.ndimage.gaussian_filter(rew[i, ..., j], sigma=frame_penalty*10)
                    rew[i, ..., j] = rew[i, ..., j] - rew[i, ..., j].min()
                    rew[i, ..., j] = rew[i, ..., j] / rew[i, ..., j].max()

            rew = np.minimum(r[0].cpu().numpy(), 1 - rew)
            # f, ax2 = plt.subplots(1, 4)
            # ax2[0].imshow(rew[i, ..., 0].T, cmap='gray', vmin=0, vmax=1)
            # ax2[1].imshow(rew[i, ..., 1].T, cmap='gray', vmin=0, vmax=1)
            ax1[1, 1].imshow(rew[i, ..., 0].T, cmap='gray', vmin=0, vmax=1)
            # ax2[3].imshow(rew[i, ..., 3].T, cmap='gray', vmin=0, vmax=1)

            ax1[0, 0].get_xaxis().set_visible(False)
            ax1[1, 0].get_xaxis().set_visible(False)
            ax1[0, 1].get_xaxis().set_visible(False)
            ax1[1, 1].get_xaxis().set_visible(False)
            ax1[0, 0].get_yaxis().set_visible(False)
            ax1[1, 0].get_yaxis().set_visible(False)
            ax1[0, 1].get_yaxis().set_visible(False)
            ax1[1, 1].get_yaxis().set_visible(False)

            plt.savefig("reward_fig.png")
            plt.show()

            r[0] = torch.tensor(rew, device=r[0].device)

        return r

    def get_nets(self):
        return list(self.nets.values())

    def get_reward_index(self, reward_name):
        return list(self.nets.keys()).index(reward_name)

    @torch.no_grad()
    def predict_full_sequence(self, pred, imgs, gt):
        stack = torch.stack((imgs.squeeze(1), pred), dim=1)

        self.patch_size = list([stack.shape[-3], stack.shape[-2], 4])
        self.inferer.roi_size = self.patch_size
        with torch.cuda.amp.autocast(enabled=False):  # force float32
            return [torch.sigmoid(p).squeeze(1) for p in self.predict(stack)]

    def prepare_for_full_sequence(self, batch_size=1) -> None:  # noqa: D102
        sw_batch_size = batch_size

        self.inferer = SlidingWindowInferer(
            roi_size=self.get_nets()[0].patch_size,
            sw_batch_size=sw_batch_size,
            overlap=0.5,
            mode='gaussian',
            cache_roi_weight_map=True,
        )

    def predict(
        self, image: Union[Tensor, MetaTensor],
    ) -> List[Union[Tensor, MetaTensor]]:
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
            if np.asarray([len(self.get_nets()[i].patch_size) == 3 for i in range(len(self.get_nets()))]).all():
                # Pad the last dimension to avoid 3D segmentation border artifacts
                pad_len = 6 if image.shape[-1] > 6 else image.shape[-1] - 1
                image = F.pad(image, (pad_len, pad_len, 0, 0, 0, 0), mode="reflect")
                pred = self.predict_3D_3Dconv_tiled(image)
                # Inverse the padding after prediction
                return [p[..., pad_len:-pad_len] for p in pred]
            else:
                raise ValueError("Check your patch size. You dummy.")
        if len(image.shape) == 4:
            raise ValueError("No 2D images here. You dummy.")

    def predict_3D_3Dconv_tiled(
        self, image: Union[Tensor, MetaTensor],
    ) -> List[Union[Tensor, MetaTensor]]:
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

        return self.sliding_window_inference(image)

    def sliding_window_inference(
        self, image: Union[Tensor, MetaTensor],
    ) -> List[Union[Tensor, MetaTensor]]:
        """Inference using sliding window.

        Args:
            image: Image to predict.

        Returns:
            Predicted logits.
        """
        return [self.inferer(
            inputs=image,
            network=n,
        ) for n in self.get_nets()]
