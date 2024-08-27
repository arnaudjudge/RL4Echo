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
from uncertainty.uncertainty import SegmentationUncertainty
from uncertainty.unet import UNet


class AleatoricUncertainty(SegmentationUncertainty):
    """Aleatoric uncertainty system.

    Args:
        iterations: number of mc dropout iterations.
    """

    def __init__(self, iterations: int = 10, is_log_sigma: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters("iterations", "is_log_sigma")
        self.iterations = iterations
        self.is_log_sigma = is_log_sigma
        self._dice = DifferentiableDiceCoefficient(include_background=False, reduction="none")#.env, apply_activation=False)

        assert not (is_log_sigma and len(self.hparams.data_params.labels) > 1), "Does not work with >1 labels"

    def configure_model(self):
        return UNet(input_shape=self.input_shape, output_shape=self.output_shape, sigma_out=True)

    def _shared_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Dict[str, Tensor]:  # noqa: D102
        x, y = batch[Tags.img], batch[Tags.gt]

        # Forward
        logits, sigma = self(x)  # (N, C, H, W), (N, C, H, W)
        binary = logits.shape[1] == 1

        sigma = F.softplus(sigma)

        if self.is_log_sigma:
            distribution = distributions.Normal(logits, torch.exp(sigma))
        else:
            distribution = distributions.Normal(logits, sigma + 1e-8)

        x_hat = distribution.rsample((self.iterations,))

        if binary:
            mc_expectation = torch.sigmoid(x_hat).mean(dim=0)
            ce = F.binary_cross_entropy(mc_expectation.squeeze(), y.float())
        else:
            mc_expectation = F.softmax(x_hat, dim=2).mean(dim=0)
            log_probs = mc_expectation.log()
            ce = F.nll_loss(log_probs, y)

        # y_hat_prime = np.concatenate([mc_expectation.detach().cpu(), 1 - mc_expectation.detach().cpu()], axis=1)
        # uncertainty_map = scipy.stats.entropy(y_hat_prime, axis=1)
        # from matplotlib import pyplot as plt
        # fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
        # ax1.imshow(logits[0].detach().cpu().squeeze())
        # ax2.imshow(sigma[0].detach().cpu().squeeze())
        # ax3.imshow(uncertainty_map[0].squeeze())
        # ax4.imshow(mc_expectation[0].detach().cpu().squeeze())
        # plt.show()

        dice_values = self._dice(mc_expectation, y)
        dices = {f"dice/{label}": dice for label, dice in zip([Label.LV, Label.MYO], dice_values)}
        mean_dice = dice_values.mean()

        loss = (self.hparams.ce_weight * ce) + (self.hparams.dice_weight * (1 - mean_dice))

        # if self.is_val_step and batch_idx == 0 and self.hparams.log_figures:
        #     mc = mc_expectation.detach().cpu()
        #     if binary:
        #         y_hat = torch.sigmoid(logits).round()
        #         sigma_pred = sigma
        #         uncertainty_map = scipy.stats.entropy(np.concatenate([mc, 1 - mc], axis=1), axis=1)
        #     else:
        #         y_hat = logits.argmax(dim=1)
        #         prediction_onehot = to_onehot(y_hat, num_classes=len(self.hparams.data_params.labels)).type(torch.bool)
        #         sigma_pred = torch.where(prediction_onehot, sigma, sigma * 0).sum(dim=1)
        #         uncertainty_map = scipy.stats.entropy(mc, axis=1)
        #
        #     # self.log_images(
        #     #     title="MC expectation",
        #     #     num_images=5,
        #     #     axes_content={
        #     #         "MC": mc_expectation.cpu().squeeze().numpy(),
        #     #     },
        #     # )
        #
        #     self.log_images(
        #         title="Sample",
        #         num_images=5,
        #         axes_content={
        #             "Image": x.cpu().squeeze().numpy(),
        #             "Gt": y.squeeze().cpu().numpy(),
        #             "Pred": y_hat.detach().cpu().squeeze().numpy(),
        #             "Sigma": sigma_pred.detach().cpu().squeeze().numpy(),
        #             "Entropy": uncertainty_map.squeeze(),
        #         },
        #     )

        # Format output
        return {"loss": loss, "ce": ce, "dice": mean_dice, **dices}

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

        # dice_values = self._dice(pred, y)
        # dices = {f"dice/{label}": dice for label, dice in zip([Label.LV, Label.MYO], dice_values)}
        # mean_dice = dice_values.mean()

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
                dicom = batch['id'][i].replace('/', '_') + "_" + batch['instant'][i]
                print(dicom)
                f.create_group(dicom)
                f[dicom]['img'] = x[i].cpu().numpy()
                f[dicom]['gt'] = y[i].cpu().numpy()
                f[dicom]['pred'] = pred[i].cpu().numpy()
                f[dicom]['reward_map'] = entropy[i]
                f[dicom]['accuracy_map'] = (pred[i].cpu().numpy() != y[i].cpu().numpy()).astype(np.uint8)

        # return {"dice": mean_dice, **dices}