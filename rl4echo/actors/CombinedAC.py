import copy

import torch
import torch.nn as nn
from torch.distributions import Categorical

from rl4echo.actors.Actors import Actor


class Unet3DActorCriticCategorical(nn.Module):
    def __init__(self, net, pretrain_ckpt=None, ref_ckpt=None):
        super().__init__()
        self.net = net
        self.old_net = copy.deepcopy(self.net)

        if pretrain_ckpt:
            # if starting from pretrained model, keep version of
            self.net.load_state_dict(torch.load(pretrain_ckpt))

            if ref_ckpt:
                # copy to have version of initial pretrained net
                self.old_net.load_state_dict(torch.load(ref_ckpt))
                # will never be updated
                self.old_net.requires_grad_(False)

    def forward(self, x):
        logits, v = self.net(x, with_critic=True)
        logits = torch.softmax(logits, dim=1)
        dist = Categorical(probs=logits.permute(0, 2, 3, 4, 1))

        v = torch.sigmoid(v)

        if hasattr(self, "old_net"):
            old_logits = torch.softmax(self.old_net(x), dim=1)
            old_dist = Categorical(probs=old_logits.permute(0, 2, 3, 4, 1))
        else:
            old_dist = None
        return logits, dist, old_dist, v


class CombinedActorCriticUnetCritic(nn.Module):
    """
        ActorCritic actor class, evaluates actor and value function approximate
        Value function is represented as a grid/matrix, from second head of actor network
    """
    def __init__(self,
                 actor,
                 eps_greedy_term=0.0,
                 actor_lr=1e-3,
                 critic_lr=1e-3):
        super().__init__()

        self.actor = actor

        self.actor_lr = actor_lr
        self.critic_lr = critic_lr

        self.eps_greedy_term = eps_greedy_term

    def get_optimizers(self):
        return torch.optim.Adam(self.actor.net.params.actor.parameters(), lr=self.actor_lr), \
                torch.optim.Adam(self.actor.net.params.critic.parameters(), lr=self.critic_lr)

    def act(self, imgs, sample=True):
        """
            Get actions from actor based on batch of images
        Args:
            imgs: batch of images
            sample: bool, use sample from distribution or deterministic method

        Returns:
            Actions
        """
        logits, distribution, _, _ = self.actor(imgs)

        if sample:
            actions = distribution.sample()

            if logits.shape != actions.shape:
                logits = torch.argmax(logits, dim=1)
            random = torch.rand(logits.shape).to(actions.device)
            actions = torch.where(random >= self.eps_greedy_term, actions, torch.round(logits))
        else:
            if len(logits.shape) > 3:
                # categorical, softmax output
                actions = torch.argmax(logits, dim=1)
            else:
                # bernoulli, sigmoid output
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
        logits, distribution, old_distribution, v = self.actor(imgs)
        log_probs = distribution.log_prob(actions)

        if old_distribution:
            old_log_probs = old_distribution.log_prob(actions).detach()
        else:
            old_log_probs = log_probs.detach()

        sampled_actions = distribution.sample()
        entropy = distribution.entropy()

        v = v.squeeze(1)

        return sampled_actions, logits, log_probs, entropy, v, old_log_probs

