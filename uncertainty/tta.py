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
from uncertainty.augmentations.affine import RandomRotation, RandomTranslation
from uncertainty.augmentations.augmentation import Compose
from uncertainty.augmentations.brightnesscontrast import RandomBrightnessContrast
from uncertainty.augmentations.gamma import RandomGamma


class TTAUncertainty(SegmentationUncertainty):
    """Aleatoric uncertainty system.

    Args:
        iterations: number of mc dropout iterations.
    """

    def __init__(self, iterations: int = 10, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.iterations = iterations
        self.model = self.configure_model()
        self.model = patch_module(self.model)

        self.tta_transforms = Compose([RandomRotation(3),
                                       RandomBrightnessContrast(0.2, 0.2),
                                       RandomGamma((0.8, 1.2)),
                                       RandomTranslation(5, 5)])

    def configure_model(self):
        net = UNet(input_shape=self.input_shape, output_shape=self.output_shape, dropout=0.25)
        net.load_state_dict(torch.load("/data/rl_logs/run_1/0/actor.ckpt"))
        return net

    def test_step(self, batch, batch_idx):
        x, y = batch[Tags.img], batch[Tags.gt]

        print(batch['id'])
        samples = []

        for _ in range(self.iterations):
            params = self.tta_transforms.get_params()
            items = self.tta_transforms.apply({'image': x}, params=params)

            pred = self.model(items['image'])
            items = self.tta_transforms.un_apply({'mask': pred}, params=params)

            samples.append(items['mask'])


        samples = np.asarray([s.cpu().numpy() for s in samples])
        print(samples.shape)

        entropy = self.sample_entropy(samples, apply_activation=True)
        pred = samples.mean(axis=0).argmax(axis=1)
        print(pred.shape)
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

        pad = 10
        entropy[:, 0:pad] = 0
        entropy[:, -pad:] = 0
        entropy[:, :, 0:pad] = 0
        entropy[:, :, -pad:] = 0

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
                dicom = batch['id'][i].replace('/', '_')+ "_" + batch['instant'][i]
                print(dicom)
                f.create_group(dicom)
                f[dicom]['img'] = x[i].cpu().numpy()
                f[dicom]['gt'] = y[i].cpu().numpy()
                f[dicom]['pred'] = pred[i]#.cpu().numpy()
                f[dicom]['reward_map'] = entropy[i]#.cpu().numpy()
                f[dicom]['accuracy_map'] = (pred[i] != y[i].cpu().numpy()).astype(np.uint8)
        print(self.output_file)