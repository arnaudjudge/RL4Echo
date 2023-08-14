import random
from typing import Any
from typing import Tuple

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger

from Reward import accuracy


class RLmodule(pl.LightningModule):

    def __init__(self, reward, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.actor = self.get_actor()
        self.reward_func = reward

    def get_actor(self):
        raise NotImplementedError

    def configure_optimizers(self):
        return self.actor.get_optimizers()

    @torch.no_grad()  # no grad since tensors are reused in PPO's for loop
    def rollout(self, imgs: torch.tensor, gt: torch.tensor, use_gt: torch.tensor = None, sample: bool =True):
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

        _, _, log_probs, _, _ = self.actor.evaluate(imgs, actions)
        return actions, log_probs, rewards

    def log_tb_images(self, viz_batch, prefix="", i=1) -> None:
        """
            Log images to tensor board (Could this be simply for any logger without change?)
        Args:
            viz_batch: batch of images and metrics to log
            prefix: prefix to add to image titles

        Returns:
            None
        """

        # Get tensorboard logger
        tb_logger = None
        for logger in self.trainer.loggers:
            if isinstance(logger, TensorBoardLogger):
                tb_logger = logger.experiment
                break
        if tb_logger is None:
            raise ValueError('TensorBoard Logger not found')

        idx = random.randint(0, len(viz_batch[0]) - 1)

        tb_logger.add_image(f"{prefix}Image", viz_batch[0][idx], viz_batch[4]*(i+1))
        tb_logger.add_image(f"{prefix}GroundTruth", viz_batch[3][idx].unsqueeze(0), viz_batch[4]*(i+1))

        def put_text(img, text):
            img = img.copy().astype(np.uint8) * 255
            return cv2.putText(img.squeeze(0), "{:.3f}".format(text), (0, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (125), 2)

        tb_logger.add_image(f"{prefix}Prediction", torch.tensor(
            put_text(viz_batch[1][idx].cpu().detach().numpy(), viz_batch[2][idx].float().mean().item())).unsqueeze(0),
                            viz_batch[4]*(i+1))

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

        self.log_tb_images((b_img[0, ...].unsqueeze(0),
                            prev_actions[0, ...].unsqueeze(0),
                            prev_rewards[0, ...].unsqueeze(0),
                            b_gt[0, ...].unsqueeze(0),
                            batch_idx))

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

        for i in range(len(b_img)):
            self.log_tb_images((b_img[i, ...].unsqueeze(0),
                                prev_actions[i, ...].unsqueeze(0),
                                acc[i, ...].unsqueeze(0),
                                b_gt[i, ...].unsqueeze(0),
                                batch_idx), prefix='test_', i=i)


        self.log_dict(logs)
        return logs

