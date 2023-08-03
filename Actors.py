import torch
import torch.nn as nn
from torch.distributions import Bernoulli
from vital.vital.models.segmentation.unet import UNet
from rewardnet.reward_net import get_resnet


class UnetActor(nn.Module):

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


class UnetCritic(nn.Module):

    def __init__(self):
        super().__init__()
        self.net = UNet(input_shape=(1, 256, 256), output_shape=(1, 256, 256))

    def forward(self, x):
        return torch.sigmoid(self.net(x))


class PGActor(nn.Module):
    """
        Simple policy gradient actor
    """
    def __init__(self, eps_greedy_term=0.0):
        super().__init__()

        self.actor = UnetActor()

        self.eps_greedy_term = eps_greedy_term

    def get_optimizers(self):
        return torch.optim.Adam(self.actor.net.parameters(), lr=1e-3)

    def act(self, imgs, sample=True):
        """
            Get actions from actor based on batch of images
        Args:
            imgs: batch of images
            sample: bool, use sample from distribution or deterministic method

        Returns:
            Actions
        """
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
            Evaluate images with both actor and critic
        Args:
            imgs: (state) images to evaluate
            actions: segmentation taken over images

        Returns:
            actions (sampled), logits from actor predictions, log_probs, placeholder value function estimate from critic
        """
        logits, distribution = self.actor(imgs)
        log_probs = distribution.log_prob(actions)

        actions = distribution.sample()

        return actions, logits, log_probs, torch.zeros(len(actions)), torch.zeros(len(actions))


class ActorCritic(nn.Module):
    """
        ActorCritic actor class, evaluates actor and value function approximate
    """
    def __init__(self, eps_greedy_term=0.0):
        super().__init__()

        self.actor = UnetActor()
        self.critic = Critic()

        self.eps_greedy_term = eps_greedy_term

    def get_optimizers(self):
        return torch.optim.Adam(self.actor.net.parameters(), lr=1e-3), \
               torch.optim.Adam(self.critic.net.parameters(), lr=1e-3)

    def act(self, imgs, sample=True):
        """
            Get actions from actor based on batch of images
        Args:
            imgs: batch of images
            sample: bool, use sample from distribution or deterministic method

        Returns:
            Actions
        """
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
            Evaluate images with both actor and critic
        Args:
            imgs: (state) images to evaluate
            actions: segmentation taken over images

        Returns:
            actions (sampled), logits from actor predictions, log_probs, value function estimate from critic
        """
        logits, distribution = self.actor(imgs)
        log_probs = distribution.log_prob(actions)

        actions = distribution.sample()
        entropy = distribution.entropy()

        # unsqueeze to match shape to multiply with logprobs
        v = self.critic(imgs).unsqueeze(-1).unsqueeze(-1)

        return actions, logits, log_probs, entropy, v


class ActorCritic_UnetCritic(nn.Module):
    """
        ActorCritic actor class, evaluates actor and value function approximate
        Value function is represented as a grid/matrix, unet is value function approximator
    """
    def __init__(self, eps_greedy_term=0.0):
        super().__init__()

        self.actor = UnetActor()
        self.critic = UnetCritic()

        self.eps_greedy_term = eps_greedy_term

    def get_optimizers(self):
        return torch.optim.Adam(self.actor.net.parameters(), lr=1e-3), \
               torch.optim.Adam(self.critic.net.parameters(), lr=1e-3)

    def act(self, imgs, sample=True):
        """
            Get actions from actor based on batch of images
        Args:
            imgs: batch of images
            sample: bool, use sample from distribution or deterministic method

        Returns:
            Actions
        """
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
            Evaluate images with both actor and critic
        Args:
            imgs: (state) images to evaluate
            actions: segmentation taken over images

        Returns:
            actions (sampled), logits from actor predictions, log_probs, value function estimate from critic
        """
        logits, distribution = self.actor(imgs)
        log_probs = distribution.log_prob(actions)

        actions = distribution.sample()
        entropy = distribution.entropy()

        v = self.critic(imgs)

        return actions, logits, log_probs, entropy, v


