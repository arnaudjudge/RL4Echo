import torch
import torch.nn as nn
from torch.distributions import Bernoulli
from vital.vital.models.segmentation.unet import UNet
from rewardnet.reward_net import get_resnet


class Actor(nn.Module):

    def __init__(self):
        super().__init__()
        self.net = UNet(input_shape=(1, 256, 256), output_shape=(1, 256, 256))

    def forward(self, x):
        logits = torch.sigmoid(self.net(x))
        dist = Bernoulli(probs=logits)
        return logits, dist


class Critic(nn.Module):

    def __init__(self):
        super().__init__()
        self.net = get_resnet(input_channels=1, num_classes=1)

    def forward(self, x):
        return torch.sigmoid(self.net(x))


class ActorCritic(nn.Module):

    def __init__(self, eps_greedy_term=0.0):
        super().__init__()

        self.actor = Actor()
        self.critic = Critic()

        self.eps_greedy_term = eps_greedy_term

    def get_optimizers(self):
        return torch.optim.Adam(self.actor.net.parameters(), lr=1e-3), \
               torch.optim.Adam(self.critic.net.parameters(), lr=1e-3)

    def act(self, imgs, sample=True):
        logits, distribution = self.actor(imgs)

        if sample:
            actions = distribution.sample()

            random = torch.rand(logits.shape).to(actions.device)
            actions = torch.where(random >= self.eps_greedy_term, actions, torch.round(logits))
        else:
            actions = torch.round(logits)

        return actions

    def evaluate(self, imgs, actions):
        """

        Args:
            imgs: (state) images to evaluate
            actions: segmentation taken over images

        Returns:

        """
        logits, distribution = self.actor(imgs)
        log_probs = distribution.log_prob(actions)

        actions = distribution.sample()

        v = self.critic(imgs).unsqueeze(-1).unsqueeze(-1)

        return actions, logits, log_probs, v


