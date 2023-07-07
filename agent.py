import matplotlib.pyplot as plt
import numpy
import torch
import torch.nn as nn
from scipy import ndimage
from torch.distributions import OneHotCategorical
from torch.distributions.categorical import Categorical
from torch.distributions.binomial import Binomial
from torch.distributions.bernoulli import Bernoulli
from vital.vital.metrics.train.functional import differentiable_dice_score
from torchmetrics import Dice

from replaybuffer import ReplayBuffer, Experience
from RLDataset import RLDataset

import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt
from reward_func import scale_reward


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

    def __init__(self, replay_buffer: ReplayBuffer, dataset: RLDataset) -> None:
        self.replay_buffer = replay_buffer
        self.dataset = dataset

    def get_action(self, img: np.array, prev_actions: np.array, net: nn.Module, epsilon: float, device: str) -> Tuple:
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

        if type(prev_actions) == numpy.ndarray:
            prev_actions = torch.from_numpy(prev_actions).cuda(device)

        if device not in ['cpu']:
            image = image.cuda(device)

        logits = torch.sigmoid(net(image))
        # actions = torch.round(logits)

        distribution = Bernoulli(probs=logits)
        if prev_actions is None:
            prev_actions = distribution.sample()
        actions = distribution.sample()
        # random = torch.rand(actions.shape).to(device)
        # explored_actions = torch.where(random >= epsilon, actions, sample)

        log_probs = distribution.log_prob(prev_actions).mean(dim=(2,3))

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

        return simple.mean(dim=(1,2), keepdim=True)

    def play_step(self, buffer: ReplayBuffer, net: nn.Module, rewardnet: nn.Module, epsilon: float = 0.0, device: str = 'cuda:0') -> Tuple[float, bool]:
        """
        Carries out a single interaction step between the agent and the environment
        Args:
            net: neural net used for segmentation (unet)
            rewardnet: neural net used for reward calculation
            epsilon: value to determine likelihood of taking a random action
            device: current device
        Returns:
            reward, done
        """

        img, gt_mask = self.dataset.get_new_image()

        action, log_prob, seg = self.get_action(img, gt_mask, net, epsilon, device)

        # get reward
        reward = self.get_reward(img, seg, rewardnet, gt_mask, device)
        exp = Experience(img,
                         action.cpu().detach().numpy(), #np.expand_dims(gt_mask, axis=0),
                         reward.cpu().detach().numpy(), #torch.ones_like(reward).cpu().detach().numpy(),
                         log_prob.squeeze(0).cpu().detach().numpy(),
                         gt_mask)

        buffer.append(exp)

        return reward, log_prob

