import copy

import torch
import torch.nn as nn
from torch.distributions import Bernoulli
from vital.vital.models.segmentation.unet import UNet
from rewardnet.reward_net import get_resnet


class UnetActor(nn.Module):

    def __init__(self, actor_pretrain_ckpt):
        super().__init__()
        self.net = UNet(input_shape=(1, 256, 256), output_shape=(1, 256, 256))

        if actor_pretrain_ckpt:
            # if starting from pretrained model, keep version of
            self.net.load_state_dict(torch.load(actor_pretrain_ckpt))

            # copy to have version of initial pretrained net
            self.old_net = copy.deepcopy(self.net)
            # will never be updated
            self.old_net.requires_grad_(False)

    def forward(self, x):
        logits = torch.sigmoid(self.net(x))
        dist = Bernoulli(probs=logits)

        if hasattr(self, "old_net"):
            old_logits = torch.sigmoid(self.old_net(x))
            old_dist = Bernoulli(probs=old_logits)
        else:
            old_dist = None
        return logits, dist, old_dist


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


class Actor(nn.Module):
    """
        Simple policy gradient actor
    """
    def __init__(self, eps_greedy_term=0.0, actor_pretrain_ckpt=None, actor_lr=1e-3, critic_lr=1e-3):
        super().__init__()

        self.actor = UnetActor(actor_pretrain_ckpt)
        self.critic = None

        self.actor_lr = actor_lr
        self.critic_lr = critic_lr

        self.eps_greedy_term = eps_greedy_term

    def get_optimizers(self):
        if self.critic is None:
            return torch.optim.Adam(self.actor.net.parameters(), lr=self.actor_lr)
        else:
            return torch.optim.Adam(self.actor.net.parameters(), lr=self.actor_lr), \
                   torch.optim.Adam(self.critic.net.parameters(), lr=self.critic_lr)

    def act(self, imgs, sample=True):
        """
            Get actions from actor based on batch of images
        Args:
            imgs: batch of images
            sample: bool, use sample from distribution or deterministic method

        Returns:
            Actions
        """
        logits, distribution, _ = self.actor(imgs)

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
            In this default case, the critic is null, therefore is not considered
        Args:
            imgs: (state) images to evaluate
            actions: segmentation taken over images

        Returns:
            actions (sampled), logits from actor predictions, log_probs,
            entropy placeholder, placeholder value function estimate from critic
        """
        logits, distribution, old_distribution = self.actor(imgs)
        log_probs = distribution.log_prob(actions)

        if old_distribution:
            old_log_probs = old_distribution.log_prob(actions).detach()
        else:
            old_log_probs = log_probs.detach()

        actions = distribution.sample()

        return actions, logits, log_probs, torch.zeros(len(actions)), torch.zeros(len(actions)), old_log_probs
