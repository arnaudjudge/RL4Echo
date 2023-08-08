from typing import Any
from typing import Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.loggers import TensorBoardLogger

from Actors import ActorCritic, ActorCriticUnetCritic
from RLmodule import RLmodule
from SectorDataModule import SectorDataModule


class PPO(RLmodule):

    def __init__(self,
                 clip_value: float = 0.2,
                 k_steps_per_batch: int = 5,
                 entropy_coeff: float = 0.0,
                 *args: Any,
                 **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.clip_value = clip_value
        self.k_steps = k_steps_per_batch
        self.entropy_coeff = entropy_coeff

        # since optimization is done manually, this flag needs to be set
        self.automatic_optimization = False

    def get_actor(self):
        return ActorCriticUnetCritic()

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], nb_batch):
        """
            Defines PPO training steo
            Get actions, log_probs and rewards from current policy
            Calculate and backprop losses for actor and critic K times in loop over same batch
        Args:
            batch: current mini batch of replay data
            nb_batch: batch number

        Returns:
            Training loss and log metrics or None
        """
        opt_net, opt_critic = self.optimizers()

        b_img, b_gt, b_use_gt = batch

        # get actions, log_probs, rewards, etc from pi (stays constant for all steps k)
        prev_actions, prev_log_probs, prev_rewards = self.rollout(b_img, b_gt, b_use_gt)

        # iterate with pi prime k times
        for k in range(self.k_steps):
            # calculates training loss
            loss, critic_loss, metrics_dict = self.compute_policy_loss((b_img, prev_actions, prev_rewards,
                                                                        prev_log_probs, b_gt))

            opt_net.zero_grad()
            self.manual_backward(loss, retain_graph=True)
            opt_net.step()

            opt_critic.zero_grad()
            self.manual_backward(critic_loss)
            opt_critic.step()

            logs = {**metrics_dict,
                    **{'loss': loss,
                       'critic_loss': critic_loss,
                       }
                    }

            self.log_dict(logs, prog_bar=True)

    def compute_policy_loss(self, batch, **kwargs):
        """
            Compute unsupervised loss to maximise reward using PPO method.
        Args:
            batch: batch of images, actions, log_probs, rewards and groudn truth
            sample: whether to sample from distribution or deterministic approach (mainly for val, test steps)

        Returns:
            mean loss(es) for the batch, metrics dictionary
        """
        b_img, b_actions, b_rewards, b_log_probs, b_gt = batch

        _, logits, log_probs, entropy, v = self.actor.evaluate(b_img, b_actions)

        assert b_rewards.shape == v.shape
        adv = b_rewards - v

        # PPO loss
        # importance ratio
        assert b_log_probs.shape == log_probs.shape
        ratio = (log_probs - b_log_probs).exp()

        # clamp with epsilon value
        clipped = ratio.clamp(1 - self.clip_value, 1 + self.clip_value)

        # min trick
        loss = -torch.min(adv * ratio, adv * clipped).mean() + (-self.entropy_coeff * entropy.mean())

        # Critic loss
        critic_loss = nn.MSELoss()(v, b_rewards.mean(dim=(1, 2, 3), keepdim=True))

        # metrics dict
        metrics = {
                'v': v.mean(),
                'advantage': adv.mean(),
                'reward': b_rewards.mean(),
                'log_probs': log_probs.mean(),
                'ratio': ratio.mean(),
                'approx_kl_div': torch.mean((torch.exp(ratio) - 1) - ratio),
        }

        return loss, critic_loss, metrics

