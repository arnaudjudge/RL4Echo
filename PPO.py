from collections import OrderedDict
from typing import Any
import cv2
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

import random
import numpy as np
from typing import Tuple

from ActorCritic import ActorCritic
from RLmodule import RLmodule
from Reward import accuracy_reward
from simpledatamodule import SectorDataModule


class PPO(RLmodule):

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.actor = self.get_actor()

        self.clip_value = 0.2
        self.k_steps = 5
        self.train_gt_inject_frac = 0.0

        self.automatic_optimization = False

    def get_actor(self):
        return ActorCritic()

    def get_reward_func(self):
        return accuracy_reward

    @torch.no_grad()
    def rollout(self, imgs, gt, inject_gt=0.0, sample=True):
        # use no grad since it can come from RB, must be like this to reuse gradient in for loop
        actions = self.actor.act(imgs, sample=sample)

        # find random indexes and replace actions with gt with probability 'inject_gt'
        idx = np.random.choice(len(actions), size=(np.random.random(size=len(actions)) < inject_gt).sum(), replace=False)
        if len(idx) > 0:
            actions[idx, ...] = gt.unsqueeze(1)[idx, ...]

        sampled_actions, _, log_probs, _ = self.actor.evaluate(imgs, actions)
        rewards = self.reward_func(actions, gt.unsqueeze(1))
        return actions, sampled_actions, log_probs, rewards

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], nb_batch) -> OrderedDict:
        """
        Then calculates loss based on the minibatch recieved
        Args:
            batch: current mini batch of replay data
            nb_batch: batch number
        Returns:
            Training loss and log metrics
        """
        device = self.get_device(batch)
        opt_net, opt_critic = self.optimizers()

        b_img, b_gt = batch

        # get actions, lp, reward, etc from pi (stays constant for all steps k)
        prev_actions, prev_sampled_actions, prev_log_probs, prev_rewards = self.rollout(b_img, b_gt,
                                                                                        inject_gt=self.train_gt_inject_frac)

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
            Compute unsupervised loss to maximise reward using policy gradient method.
        Args:
            prediction (Tensor): (N, H, W, K) segmentation map predicted by network
            y (Tensor): (N, H, W, K) reference segmentation map for reward calculation
        Returns:
            Tensor, mean loss for the batch
        """
        b_img, b_actions, b_rewards, b_log_probs, b_gt = batch

        sampled_actions, logits, log_probs, v = self.actor.evaluate(b_img, b_actions)
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

    def validation_step(self, batch, batch_idx: int):
        device = self.get_device(batch)
        b_img, b_gt = batch

        prev_actions, prev_sampled_actions, prev_log_probs, prev_rewards = self.rollout(b_img, b_gt)

        loss, critic_loss, metrics_dict = self.compute_policy_loss((b_img, prev_actions, prev_rewards,
                                                                    prev_log_probs, b_gt))

        logs = {'val_loss': loss,
                "val_reward": torch.mean(prev_rewards.type(torch.float)),
                }

        self.log_tb_images((b_img[0, ...].unsqueeze(0),
                            prev_actions[0, ...].unsqueeze(0),
                            prev_rewards[0, ...].unsqueeze(0),
                            b_gt[0, ...].unsqueeze(0),
                            batch_idx))

        self.log_dict(logs)
        return logs

    def log_tb_images(self, viz_batch, prefix="") -> None:

        # Get tensorboard logger
        tb_logger = None
        for logger in self.trainer.loggers:
            if isinstance(logger, TensorBoardLogger):
                tb_logger = logger.experiment
                break
        if tb_logger is None:
            raise ValueError('TensorBoard Logger not found')

        idx = random.randint(0, len(viz_batch[0])-1)

        tb_logger.add_image(f"{prefix}Image", viz_batch[0][idx], viz_batch[4])
        tb_logger.add_image(f"{prefix}GroundTruth", viz_batch[3][idx].unsqueeze(0), viz_batch[4])

        def put_text(img, text):
            img = img.copy().astype(np.uint8)*255
            return cv2.putText(img.squeeze(0), "{:.3f}".format(text), (0, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (125), 2)

        tb_logger.add_image(f"{prefix}Prediction", torch.tensor(
            put_text(viz_batch[1][idx].cpu().detach().numpy(), viz_batch[2][idx].float().mean().item())).unsqueeze(0),
                            viz_batch[4])

    def test_step(self, batch, batch_idx: int):
        b_img, b_gt = batch

        prev_actions, prev_sampled_actions, prev_log_probs, prev_rewards = self.rollout(b_img, b_gt, sample=False)
        loss, critic_loss, metrics_dict = self.compute_policy_loss((b_img, prev_actions, prev_rewards,
                                                                    prev_log_probs, b_gt))

        logs = {'test_loss': loss,
                "test_reward": torch.mean(prev_rewards.type(torch.float)),
                }
        self.log_tb_images((b_img[0, ...].unsqueeze(0),
                            prev_actions[0, ...].unsqueeze(0),
                            prev_rewards[0, ...].unsqueeze(0),
                            b_gt[0, ...].unsqueeze(0),
                            batch_idx), prefix='test_')

        self.log_dict(logs)
        return logs


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

