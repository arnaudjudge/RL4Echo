import matplotlib.pyplot as plt
import numpy
import torch
import torch.nn as nn
from scipy import ndimage
from torch.distributions import OneHotCategorical
from torch.distributions.categorical import Categorical
from torch.distributions.binomial import Binomial
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.normal import Normal
from vital.vital.metrics.train.functional import differentiable_dice_score
from torchmetrics import Dice

import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt
from reward_scaling import scale_reward


def dice_coefficient(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection) / (np.sum(y_true) + np.sum(y_pred))


def torch_dice_coefficient(y_true, y_pred):
    intersection = torch.sum(y_true * y_pred)
    return (2. * intersection) / (torch.sum(y_true) + torch.sum(y_pred))


class Agent:
    """
    Base Agent class handeling the interaction with the environment
    Args:
        env: training environment
        replay_buffer: replay buffer storing experiences
    """

    def get_action(self, img: np.array, prev_actions: np.array, net: nn.Module, epsilon: float, device: str, sample: bool = True) -> Tuple:
        """
        Using the given network, decide what action to carry out
        using an epsilon-greedy policy
        Args:
            img: new image (state) from which to get action (segmentation)
            net: neural net used for segmentatio (unet)
            epsilon: value to determine likelihood of taking a random action
            device: current device
        Returns:
            action
        """
        if type(img) == numpy.ndarray:
            image = torch.from_numpy(img)
        else:
            image = img
        if len(image.shape) == 3:
            image = image.unsqueeze(0)

        if prev_actions is not None:
            if type(prev_actions) == numpy.ndarray:
                prev_actions = torch.from_numpy(prev_actions).cuda(device)
            if len(prev_actions.shape) == 3:
                prev_actions = prev_actions.unsqueeze(1)

        if device not in ['cpu']:
            image = image.cuda(device)

        logits = torch.sigmoid(net(image))

        distribution = Bernoulli(probs=logits)
        if prev_actions is None:
            prev_actions = distribution.sample()

        if sample:
            actions = distribution.sample()
        else:
            actions = torch.round(logits)

        # if prev_actions is None:
        #     prev_actions = actions.clone()

        log_probs = distribution.log_prob(prev_actions)

        return actions.squeeze(1), log_probs, logits.squeeze(1)

    @torch.no_grad()
    def get_reward(self, img, segmentation, rewardnet, gt, device):

        if type(img) == numpy.ndarray:
            img = torch.from_numpy(img)
        if len(img.shape) == 3:
            img = img.unsqueeze(0)
        img = img.float().to(device)

        if type(gt) == numpy.ndarray:
            gt = torch.from_numpy(gt)
        if len(gt.shape) == 2:
            gt = gt.unsqueeze(0)
        gt = gt.float().to(device)

        # SIMPLE COMPARISON WITH GT
        actions = torch.round(segmentation)
        simple = (actions == gt).float()

        #
        # # REWARD NETWORK
        # stack = torch.stack((gt, actions), dim=1)
        #
        # if device not in ['cpu']:
        #     stack = stack.cuda(device)
        #
        # r = torch.sigmoid(rewardnet(stack))
        #
        # #reward = torch.argmax(r, dim=-1)
        # reward = r.squeeze(1).repeat(256, 256, 1).permute(2, 0, 1)
        # reward = torch.where(reward == 0, torch.tensor(-1).to(device), reward)

        # x = torch.tensor(np.arange(0, 1, 0.01))
        # y_np = scale_reward(x).detach().cpu().numpy()

        #reward = scale_reward(simple.mean(dim=(1,2), keepdim=True))

        # if len(reward) > 1:
        #
        #     plt.figure()
        #     plt.plot(x, y_np)
        #     plt.scatter(simple.mean(dim=(1,2), keepdim=True).detach().cpu().numpy(), reward.detach().cpu().numpy())

        #self.baseline = self.baseline / 2 + simple.mean() / 2

        return simple.mean(dim=(1,2), keepdim=True)
