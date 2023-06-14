import matplotlib.pyplot as plt
import numpy
import torch
import torch.nn as nn
from torch.distributions import OneHotCategorical
from torch.distributions.categorical import Categorical
from vital.vital.metrics.train.functional import differentiable_dice_score

from replaybuffer import ReplayBuffer, Experience
from RLDataset import RLDataset

import numpy as np
from typing import Tuple


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

    def get_action(self, img: np.array, net: nn.Module, epsilon: float, device: str) -> Tuple:
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
        image = torch.tensor(img)
        if len(image.shape) == 3:
            image = image.unsqueeze(0)

        if device not in ['cpu']:
            image = image.cuda(device)

        segmentation = net(image)
        _, action = torch.max(segmentation, dim=1)

        distribution = Categorical(torch.softmax(segmentation, dim=1).permute(0, 2, 3, 1))  # Permute to sample on last dimension
        # ???
        sample = distribution.sample()
        log_probs = distribution.log_prob(sample)#.reshape(action.shape[0], -1).sum(-1)

        return action, log_probs

    @torch.no_grad()
    def get_reward(self, img, segmentation, rewardnet, gt, device):

        if type(img) == numpy.ndarray:
            img = torch.from_numpy(img)
        if len(img.shape) == 3:
            img = img.unsqueeze(0)
        img = img.float().to(device)

        # DICE
        # dice = torch.ones((len(segmentation)), device=device)
        # # p = torch.argmax(nn.functional.softmax(segmentation, dim=1), dim=1).unsqueeze(1)
        # p = nn.functional.softmax(segmentation.float(), dim=1)
        # for i in range(len(dice)):
        #     dice[i] = differentiable_dice_score(p[i, ...].unsqueeze(0), torch.tensor(gt)[i, ...].unsqueeze(0))
        #     plt.figure()
        #     plt.title(dice[i])
        #     plt.imshow(segmentation.cpu().numpy()[0])
        #
        #     plt.figure()
        #     plt.imshow(gt)
        #
        #     plt.show()

        # REWARD NETWORK
        stack = torch.stack((img, segmentation.unsqueeze(1)), dim=1).squeeze(2)

        if device not in ['cpu']:
            stack = stack.cuda(device)

        r = rewardnet(stack)

        reward = torch.argmax(r, dim=-1)
        # reward = torch.where(reward == 0, torch.tensor(-1).to(device), reward)
        return reward #dice / 100

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

        action, log_prob = self.get_action(img, net, epsilon, device)

        # get reward
        reward = self.get_reward(img, action, rewardnet, gt_mask, device)

        exp = Experience(img,
                         action.cpu().detach().numpy(),
                         reward.cpu().detach().numpy(),
                         log_prob.cpu().detach().numpy(),
                         gt_mask)

        buffer.append(exp)

        return reward, log_prob

