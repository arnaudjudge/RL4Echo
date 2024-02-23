from typing import Dict, Tuple

import hydra
import numpy as np
import scipy
import torch
import torch.distributions as distributions
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from torchmetrics.utilities.data import to_onehot
from vital.data.config import Tags
from vital.metrics.train.metric import DifferentiableDiceCoefficient
from vital.data.camus.config import Label
import h5py

from uncertainty.mcdropout_utils import patch_module
from uncertainty.uncertainty import SegmentationUncertainty
from uncertainty.unet import UNet


class AleatoricUncertainty(SegmentationUncertainty):
    """Aleatoric uncertainty system.

    Args:
        iterations: number of mc dropout iterations.
    """

    def __init__(self, iterations: int = 10, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.iterations = iterations
        self.model = patch_module(self.model)


    def configure_model(self):
        return UNet(input_shape=self.input_shape, output_shape=self.output_shape, dropout=0.25)

    def test_step(self, batch, batch_idx):
        x, y = batch[Tags.img], batch[Tags.gt]

        print(batch['id'])

        # Forward
        logits, sigma = self(x)  # (N, C, H, W), (N, C, H, W)
        sigma = F.softplus(sigma)

        if self.is_log_sigma:
            distribution = distributions.Normal(logits, torch.exp(sigma))
        else:
            distribution = distributions.Normal(logits, sigma + 1e-8)

        samples = distribution.rsample((25,))

        print(samples.shape)

        if logits.shape[1] == 1:
            y_hat = torch.sigmoid(logits)
            # mc_expectation = torch.sigmoid(samples).mean(dim=0)
            samples = torch.sigmoid(samples)
            sigma = sigma.squeeze(1)
            pred = y_hat.round()
        else:
            y_hat = F.softmax(logits, dim=1)
            prediction_onehot = to_onehot(y_hat.argmax(1), num_classes=samples.shape[2]).type(torch.bool)
            sigma = torch.where(prediction_onehot, sigma, sigma * 0).sum(dim=1)
            # mc_expectation = F.softmax(samples, dim=2).mean(dim=0)
            samples = F.softmax(samples, dim=2)
            pred = y_hat.argmax(1)

        entropy = self.sample_entropy(samples, apply_activation=False)

        print(y_hat.shape)
        print(entropy.shape)

        print(samples.shape)

        # from matplotlib import pyplot as plt
        #
        # f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
        # ax1.imshow(x[0].cpu().squeeze())
        # ax2.imshow(y_hat[0].argmax(0).cpu().squeeze())
        # ax3.imshow(samples[0, 0].argmax(0).cpu().squeeze())
        # ax4.imshow(entropy[0].squeeze())
        #
        # plt.show()

        # with h5py.File('jsrt_contour.h5', "w") as dataset:
        #
        #
        #     for i in range(x.shape[0]):
        #         group = dataset.create_group(batch['id'][i])
        #
        #         group.create_dataset(name=Tags.img, data=x[i].cpu(), **img_save_options)
        #         group.create_dataset(name=Tags.gt, data=y[i].cpu(), **seg_save_options)
        #         group.create_dataset(name=ContourTags.contour, data=landmarks)

        with h5py.File(self.output_file, 'a') as f:
            for i in range(x.shape[0]):
                dicom = batch['id'][i].replace('/', '_')
                print(dicom)
                f.create_group(dicom)
                f[dicom]['img'] = x[i].cpu().numpy()
                f[dicom]['gt'] = y[i].cpu().numpy()
                f[dicom]['pred'] = pred[i].cpu().numpy()
                f[dicom]['reward_map'] = entropy[i]
                f[dicom]['accuracy_map'] = (pred[i].cpu().numpy() != y[i].cpu().numpy()).astype(np.uint8)