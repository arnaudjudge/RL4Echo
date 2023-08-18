import random
from typing import Any
from typing import Tuple

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger

from utils.Metrics import accuracy
from utils.logging_helper import log_image


class RLmodule(pl.LightningModule):

    def __init__(self, actor, reward, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.actor = actor
        self.reward_func = reward

    def configure_optimizers(self):
        return self.actor.get_optimizers()

    @torch.no_grad()  # no grad since tensors are reused in PPO's for loop
    def rollout(self, imgs: torch.tensor, gt: torch.tensor, use_gt: torch.tensor = None, sample: bool = True):
        """
            Rollout the policy over a batch of images and ground truth pairs
        Args:
            imgs: batch of images
            gt: batch of ground truth segmentation maps
            inject_gt: probability of replacing policy result with ground truth
            sample: whether to sample actions from distribution or determinist

        Returns:
            Actions (used for rewards, log_pobs, etc), sampled_actions (mainly for display), log_probs, rewards
        """
        actions = self.actor.act(imgs, sample=sample)
        rewards = self.reward_func(actions, imgs, gt.unsqueeze(1))

        if use_gt is not None:
            actions[use_gt, ...] = gt.unsqueeze(1)[use_gt, ...]
            rewards[use_gt, ...] = torch.ones_like(rewards)[use_gt, ...]

        _, _, log_probs, _, _, _ = self.actor.evaluate(imgs, actions)
        return actions, log_probs, rewards

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], nb_batch):
        """
            Defines training steo, calculates loss based on the minibatch recieved.
            Optionally backprop through networks with self.automatic_optimization = False flag,
            otherwise return loss dict and Lighting does it automatically
        Args:
            batch: current mini batch of replay data
            nb_batch: batch number

        Returns:
            Training loss and log metrics or None
        """
        raise NotImplementedError

    def compute_policy_loss(self, batch, sample=True, **kwargs):
        """
            Compute unsupervised loss to maximise reward using policy gradient method.
        Args:
            batch: batch of images, actions, log_probs, rewards and groudn truth
            sample: whether to sample from distribution or deterministic approach (mainly for val, test steps)

        Returns:
            mean loss(es) for the batch, metrics dictionary
        """
        raise NotImplementedError

    def validation_step(self, batch, batch_idx: int):
        """
            Defines validation step (using sampling to show model confidence)
            Computes actions from current policy and calculates loss, rewards (and other metrics)
            Logs images and segmentations to logger
        Args:
            batch: batch of images and ground truth
            batch_idx: index of batch

        Returns:
            Dict of logs
        """
        b_img, b_gt, b_use_gt = batch

        prev_actions, prev_log_probs, prev_rewards = self.rollout(b_img, b_gt)

        loss, critic_loss, metrics_dict = self.compute_policy_loss((b_img, prev_actions, prev_rewards,
                                                                    prev_log_probs, b_gt, b_use_gt))

        acc = accuracy(prev_actions, b_img, b_gt.unsqueeze(1))

        logs = {'val_loss': loss,
                "val_reward": torch.mean(prev_rewards.type(torch.float)),
                "val_acc": acc.mean(),
                }

        # log images
        idx = random.randint(0, len(b_img) - 1)  # which image to log
        log_image(self.logger, img=b_img[idx], title='Image', number=batch_idx)
        log_image(self.logger, img=b_gt[idx].unsqueeze(0), title='GroundTruth', number=batch_idx)
        log_image(self.logger, img=prev_actions[idx], title='Prediction', number=batch_idx,
                  img_text=prev_rewards[idx].mean())
        if prev_rewards.shape == prev_actions.shape:
            log_image(self.logger, img=prev_rewards[idx], title='RewardMap', number=batch_idx)

        self.log_dict(logs)
        return logs

    def test_step(self, batch, batch_idx: int):
        """
            Defines test step (uses deterministic method to show real results)
            Computes actions from current policy and calculates loss, rewards (and other metrics)
            Logs images and segmentations to logger
        Args:
            batch: batch of images and ground truth
            batch_idx: index of batch

        Returns:
            Dict of logs
        """
        b_img, b_gt, b_use_gt = batch

        prev_actions, prev_log_probs, prev_rewards = self.rollout(b_img, b_gt, sample=False)
        loss, critic_loss, metrics_dict = self.compute_policy_loss((b_img, prev_actions, prev_rewards,
                                                                    prev_log_probs, b_gt, b_use_gt))
        acc = accuracy(prev_actions, b_img, b_gt.unsqueeze(1))

        logs = {'test_loss': loss,
                "test_reward": torch.mean(prev_rewards.type(torch.float)),
                'test_acc': acc.mean()
                }

        # for logging v
        _, _, _, _, v, _ = self.actor.evaluate(b_img, prev_actions)

        for i in range(len(b_img)):
            log_image(self.logger, img=b_img[i], title='test_Image', number=batch_idx * (i + 1))
            log_image(self.logger, img=b_gt[i].unsqueeze(0), title='test_GroundTruth', number=batch_idx * (i + 1))
            log_image(self.logger, img=prev_actions[i], title='test_Prediction', number=batch_idx * (i + 1),
                      img_text=prev_rewards[i].mean())
            if v.shape == prev_actions.shape:
                log_image(self.logger, img=v[i], title='test_v_function', number=batch_idx * (i + 1),
                          img_text=v[i].mean())
            if prev_rewards.shape == prev_actions.shape:
                log_image(self.logger, img=prev_rewards[i], title='test_RewardMap', number=batch_idx * (i + 1))


        self.log_dict(logs)
        return logs

