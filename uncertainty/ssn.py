import math
from typing import Dict, Tuple

import SimpleITK as sitk
import hydra
import matplotlib.pyplot as plt
import scipy
import torch
import torch.distributions as td
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchmetrics.utilities.data import to_onehot
from vital.data.config import Tags
from vital.metrics.train.metric import DifferentiableDiceCoefficient
import numpy as np
from vital.data.camus.config import Label
from uncertainty.uncertainty import SegmentationUncertainty
from uncertainty.unet import UNet


class Sampler(object):
    def __init__(self, seed=None):
        self.seed = seed
        if seed is not None:
            torch.manual_seed(seed)

    def __call__(self, num_samples):
        raise NotImplementedError


class LowRankMultivariateNormalRandomSampler(Sampler):
    def __init__(self, logit_mean, cov_diag, cov_factor, seed=None):
        super().__init__(seed)
        self.dist, self.shape, self.rank = self.build_distribution(logit_mean, cov_diag, cov_factor)

    @staticmethod
    def build_distribution(logit_mean, cov_diag, cov_factor):
        # logit_mean = cast_to_tensor(logit_mean, device)
        # cov_diag = cast_to_tensor(cov_diag, device)
        # cov_factor = cast_to_tensor(cov_factor, device)

        shape = logit_mean.shape
        num_classes = shape[0]
        rank = int(cov_factor.shape[0] / num_classes)
        logit_mean = logit_mean.view(-1)
        cov_diag = cov_diag.view(-1)
        cov_factor = cov_factor.view((rank, -1)).transpose(1, 0)
        epsilon = 1e-3
        dist = td.LowRankMultivariateNormal(loc=logit_mean, cov_factor=cov_factor, cov_diag=cov_diag + epsilon)
        return dist, shape, rank

    def __call__(self, num_samples):
        logit_samples = self.dist.sample([num_samples])
        logit_samples = logit_samples.view((num_samples,) + self.shape)

        if logit_samples.shape[1] == 1:
            prob_maps = torch.sigmoid(logit_samples)
            samples = torch.round(prob_maps)
        else:
            prob_maps = torch.softmax(logit_samples, dim=1)
            samples = torch.argmax(logit_samples, dim=1)
        return samples, prob_maps


class ReshapedDistribution(td.Distribution):
    def __init__(self, base_distribution: td.Distribution, new_event_shape: Tuple[int, ...]):
        self.base_distribution = base_distribution
        super().__init__(batch_shape=base_distribution.batch_shape, event_shape=new_event_shape)
        self.base_distribution = base_distribution
        self.new_shape = base_distribution.batch_shape + new_event_shape

    @property
    def support(self):
        return self.base_distribution.support

    @property
    def arg_constraints(self):
        return {}

    @property
    def mean(self):
        return self.base_distribution.mean.view(self.new_shape)

    @property
    def variance(self):
        return self.base_distribution.variance.view(self.new_shape)

    def rsample(self, sample_shape=torch.Size()):
        return self.base_distribution.rsample(sample_shape).view(sample_shape + self.new_shape)

    def log_prob(self, value):
        return self.base_distribution.log_prob(value.view(self.batch_shape + (-1,)))

    def entropy(self):
        return self.base_distribution.entropy()


class StochasticSegmentationNetworkLossMCIntegral(nn.Module):
    def __init__(self, num_mc_samples: int = 20):
        super().__init__()
        self.num_mc_samples = num_mc_samples

    @staticmethod
    def fixed_re_parametrization_trick(dist, num_samples):
        assert num_samples % 2 == 0
        samples = dist.rsample((num_samples // 2,))
        mean = dist.mean.unsqueeze(0)
        samples = samples - mean
        return torch.cat([samples, -samples]) + mean

    def forward(self, logits, target, distribution, **kwargs):
        batch_size = logits.shape[0]
        num_classes = logits.shape[1]
        # assert num_classes >= 2  # not implemented for binary case with implied background
        # logit_sample = distribution.rsample((self.num_mc_samples,))
        logit_sample = self.fixed_re_parametrization_trick(distribution, self.num_mc_samples)
        target = target.unsqueeze(1)
        target = target.expand((self.num_mc_samples,) + target.shape)

        flat_size = self.num_mc_samples * batch_size
        logit_sample = logit_sample.view((flat_size, num_classes, -1))
        target = target.reshape((flat_size, -1))

        if num_classes == 1:
            log_prob = -F.binary_cross_entropy_with_logits(logit_sample.squeeze(), target.float(),
                                                           reduction='none').view(
                (self.num_mc_samples, batch_size, -1))
        else:
            log_prob = -F.cross_entropy(logit_sample, target, reduction='none').view(
                (self.num_mc_samples, batch_size, -1))
        loglikelihood = torch.mean(torch.logsumexp(torch.sum(log_prob, dim=-1), dim=0) - math.log(self.num_mc_samples))
        loss = -loglikelihood
        return loss


class StochasticSegmentationNetwork(SegmentationUncertainty):
    def __init__(self,
                 rank: int = 5,
                 mc_samples: int = 20,
                 epsilon=1e-5,
                 diagonal=False,
                 *args, **kwargs):
        self.rank = rank
        self.save_hyperparameters()
        super().__init__(*args, **kwargs)
        self.mc_samples = mc_samples

        self._dice = DifferentiableDiceCoefficient(include_background=False, reduction="none", apply_activation=True)

        self.loss = StochasticSegmentationNetworkLossMCIntegral(mc_samples)

        self.epsilon = epsilon
        self.diagonal = diagonal  # whether to use only the diagonal (independent normals)


    def configure_model(self):
        return UNet(input_shape=self.input_shape, output_shape=self.output_shape, ssn_rank=self.rank)

    def forward(self, x: torch.Tensor, *args, **kwargs):

        mean, cov_diag, cov_factor = self.net(x)

        batch_size = x.shape[0]
        event_shape = self.output_shape
        num_classes = self.output_shape[0]

        mean = mean.view((batch_size, -1))

        cov_diag = cov_diag.exp() + self.epsilon
        cov_diag = cov_diag.view((batch_size, -1))

        cov_factor = cov_factor.view((batch_size, self.rank, num_classes, -1))
        cov_factor = cov_factor.flatten(2, 3)
        cov_factor = cov_factor.transpose(1, 2)

        if self.diagonal:
            base_distribution = td.Independent(td.Normal(loc=mean, scale=torch.sqrt(cov_diag)), 1)
        else:
            try:
                base_distribution = td.LowRankMultivariateNormal(loc=mean, cov_factor=cov_factor, cov_diag=cov_diag)
            except:
                print('Covariance became not invertible using independent normals for this batch!')
                base_distribution = td.Independent(td.Normal(loc=mean, scale=torch.sqrt(cov_diag)), 1)

        distribution = ReshapedDistribution(base_distribution, event_shape)

        shape = (batch_size,) + event_shape
        logit_mean = mean.view(shape)
        cov_diag_view = cov_diag.view(shape).detach()
        cov_factor_view = cov_factor.transpose(2, 1).view(
            (batch_size, num_classes * self.rank) + event_shape[1:]).detach()

        output_dict = {'logit_mean': logit_mean.detach(),
                       'cov_diag': cov_diag_view,
                       'cov_factor': cov_factor_view,
                       'distribution': distribution}

        return logit_mean, output_dict

    def _shared_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Dict[str, Tensor]:  # noqa: D102
        x, y = batch[Tags.img], batch[Tags.gt]

        # plt.figure()
        # plt.imshow(x[0].cpu().squeeze().numpy())
        # plt.show()

        logit_mean, output_dict = self.forward(x)

        loss = self.loss(logit_mean, y, **output_dict)

        dice_values = self._dice(logit_mean, y)
        dices = {f"dice/{label}": dice for label, dice in zip([Label.LV, Label.MYO], dice_values)}
        mean_dice = dice_values.mean()

        # if batch_idx == 0:
        #     print(logit_mean.shape)
        #     if logit_mean.shape[1] == 1:
        #         y_hat = torch.sigmoid(logit_mean).round()
        #     else:
        #         y_hat = logit_mean.argmax(dim=1)
        #
        #     self.log_images(
        #         title="VAL Sample" if self.is_val_step else 'TRAIN Sample',
        #         num_images=5,
        #         axes_content={
        #             "Image": x.cpu().squeeze().numpy(),
        #             "Gt": y.squeeze().cpu().numpy(),
        #             "Pred": y_hat.detach().cpu().squeeze().numpy(),
        #         },
        #     )

        return {"loss": loss, 'dice': mean_dice, **dices}

    def configure_optimizers(self):
        # add weight decay so predictions are less certain, more randomness?
        return torch.optim.RMSprop(self.parameters(), lr=0.001, weight_decay=1e-4)

