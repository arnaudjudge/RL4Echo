from typing import Any
from typing import Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from Actors import ActorCritic
from RLmodule import RLmodule
from Reward import accuracy_reward
from SectorDataModule import SectorDataModule


class PPO(RLmodule):

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.clip_value = 0.2
        self.k_steps = 5

        self.automatic_optimization = False

    def get_actor(self):
        return ActorCritic()

    def get_reward_func(self):
        return accuracy_reward

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], nb_batch):
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

        b_img, b_gt = batch

        # get actions, log_probs, rewards, etc from pi (stays constant for all steps k)
        prev_actions, prev_log_probs, prev_rewards = self.rollout(b_img, b_gt, inject_gt=self.train_gt_inject_frac)

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

        _, logits, log_probs, v = self.actor.evaluate(b_img, b_actions)
        reward = self.reward_func(torch.round(logits), b_gt.unsqueeze(1))

        adv = b_rewards - v

        # PPO loss
        # importance ratio
        assert b_log_probs.shape == log_probs.shape
        ratio = (log_probs - b_log_probs).exp()

        # clamp with epsilon value
        clipped = ratio.clamp(1 - self.clip_value, 1 + self.clip_value)

        # min trick
        loss = -torch.min(adv * ratio, adv * clipped).mean()

        # Critic loss
        critic_loss = nn.MSELoss()(v, b_rewards)

        # metrics dict
        metrics = {
                'v': v.mean(),
                'advantage': adv.mean(),
                'prev_reward': b_rewards.mean(),
                'reward': reward.mean(),
                'log_probs': log_probs.mean(),
                'ratio': ratio.mean(),
                'approx_kl_div': torch.mean((torch.exp(ratio) - 1) - ratio),
        }

        return loss, critic_loss, metrics


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    logger = TensorBoardLogger('logs', name='PPO')

    model = PPO()
    dl = SectorDataModule('/home/local/USHERBROOKE/juda2901/dev/data/icardio/train_subset/',
                          '/home/local/USHERBROOKE/juda2901/dev/data/icardio/train_subset/subset.csv')

    checkpoint_callback = ModelCheckpoint(monitor="val_reward", mode='max')
    trainer = pl.Trainer(max_epochs=10, logger=logger, log_every_n_steps=1, gpus=1, callbacks=[checkpoint_callback])

    trainer.fit(train_dataloaders=dl, model=model)

    trainer.test(model=model, dataloaders=dl, ckpt_path="best")

