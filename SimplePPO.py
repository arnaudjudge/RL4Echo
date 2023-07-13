from collections import OrderedDict
from typing import Any
import cv2
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from torchmetrics import Dice
from vital.vital.models.segmentation.unet import UNet
from vital.vital.metrics.train.functional import differentiable_dice_score
import random
import numpy as np
from typing import Tuple

from agent import Agent
from RLDataset import RLDataset
from replaybuffer import ReplayBuffer, Experience
from rewardnet.reward_net import get_resnet

from tqdm import tqdm

from simpledatamodule import SectorDataModule


class SimplePPO(pl.LightningModule):

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.net = UNet(input_shape=(1, 256, 256), output_shape=(1, 256, 256))
        self.net.to(device='cuda:0')

        self.critic = get_resnet(input_channels=1, num_classes=1)
        self.net.to(device='cuda:0')

        self.agent = Agent(None, None)

        self.clip_value = 0.1
        self.k_steps = 5

        self.automatic_optimization = False

    def forward(self, x):
        return self.net.forward(x)

    def get_device(self, batch) -> str:
        """Retrieve device currently being used by minibatch"""
        return batch[0].device.index if self.on_gpu else 'cpu'

    def configure_optimizers(self):
        return torch.optim.Adam(self.net.parameters(), lr=1e-3), \
               torch.optim.Adam(self.critic.parameters(), lr=1e-3)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], nb_batch) -> OrderedDict:
        """
        Carries out a single step through the environment to update the replay buffer.
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
        # use no grad since it can come from RB, must be like this to reuse gradient in for loop
        with torch.no_grad():
            _, prev_log_probs, prev_segmentations = self.agent.get_action(b_img, None, self.net, epsilon=0.0,
                                                                                     device=self.get_device(batch),
                                                                                     sample=True)
            prev_actions = torch.round(prev_segmentations)
            prev_rewards = self.agent.get_reward(b_img, prev_segmentations, None, b_gt, device)

        v = torch.sigmoid(self.critic(b_img).unsqueeze(-1))

        adv = prev_rewards - v.detach()
        # adv = (adv - adv.mean()) / (adv.std() + 1e-10)

        # iterate with pi prime k times
        for k in range(self.k_steps):
            # calculates training loss
            loss, rewards, actions, log_probs, ratio = self.compute_policy_loss((b_img, prev_actions, adv,
                                                                                 prev_log_probs, b_gt), sample=True)

            v = self.critic(b_img).unsqueeze(-1)
            critic_loss = nn.MSELoss()(v, prev_rewards)

            opt_net.zero_grad()
            self.manual_backward(loss, retain_graph=True)
            opt_net.step()

            opt_critic.zero_grad()
            self.manual_backward(critic_loss)
            opt_critic.step()

            logs = {
                'loss': loss,
                'critic_loss': critic_loss,
                'advantage': adv.mean(),
                'v': v.mean(),
                'prev_reward': torch.mean(prev_rewards.type(torch.float)),
                'reward': torch.mean(rewards.type(torch.float)),
                'log_probs': log_probs.mean(),
                'ratio': ratio.mean(),
            }
            self.log_dict(logs, prog_bar=True)

        # return OrderedDict({'loss': loss, 'log': logs, 'progress_bar': logs})

    def compute_policy_loss(self, batch, sample=True, **kwargs):
        """
            Compute unsupervised loss to maximise reward using policy gradient method.
        Args:
            prediction (Tensor): (N, H, W, K) segmentation map predicted by network
            y (Tensor): (N, H, W, K) reference segmentation map for reward calculation
        Returns:
            Tensor, mean loss for the batch
        """
        b_imgs, b_actions, b_rewards, b_log_probs, b_gt = batch

        actions, log_probs, seg = self.agent.get_action(b_imgs, b_actions, self.net, epsilon=0.0,
                                                        device=self.get_device(batch), sample=sample)

        reward = self.agent.get_reward(b_imgs, seg, None, b_gt, self.get_device(batch))

        # importance ratio
        assert b_log_probs.shape == log_probs.shape
        ratio = (log_probs - b_log_probs).exp()

        # clamp with epsilon value
        clipped = ratio.clamp(1 - self.clip_value, 1 + self.clip_value)

        # min trick
        loss = -torch.min(b_rewards * ratio, b_rewards * clipped).mean()

        return loss, reward, actions, log_probs, ratio

    def validation_step(self, batch, batch_idx: int):
        device = self.get_device(batch)

        b_img, b_gt = batch

        prev_actions, prev_log_probs, prev_segmentations = self.agent.get_action(b_img, None, self.net, epsilon=0.0,
                                                                                 device=self.get_device(batch),
                                                                                 sample=True)
        prev_rewards = self.agent.get_reward(b_img, prev_segmentations, None, b_gt, device)

        # calculates training loss
        loss, reward, actions, log_probs, _ = self.compute_policy_loss((b_img, prev_actions, prev_rewards, prev_log_probs, b_gt), sample=True)

        logs = {'val_loss': loss,
                "val_reward": torch.mean(reward.type(torch.float)),
                }
        self.log_tb_images((b_img[0, ...].unsqueeze(0), actions[0, ...].unsqueeze(0), reward[0, ...].unsqueeze(0),
                            prev_actions[0, ...].unsqueeze(0), prev_rewards[0, ...].unsqueeze(0), b_gt[0, ...].unsqueeze(0),
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

        tb_logger.add_image(f"{prefix}Image", viz_batch[0][idx], viz_batch[6])
        tb_logger.add_image(f"{prefix}GroundTruth", viz_batch[5][idx].unsqueeze(0), viz_batch[6])

        def put_text(img, text):
            img = img.copy().astype(np.uint8)*255
            return cv2.putText(img, "{:.3f}".format(text), (0, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (125), 2)

        tb_logger.add_image(f"{prefix}Prediction", torch.tensor(
            put_text(viz_batch[1][idx].cpu().detach().numpy(), viz_batch[2][idx].float().mean().item())).unsqueeze(0),
                            viz_batch[6])

        tb_logger.add_image(f"{prefix}Previous prediction", torch.tensor(
            put_text(viz_batch[3][idx].squeeze(0).cpu().detach().numpy(),
                     viz_batch[4][idx].float().mean().item())).unsqueeze(0), viz_batch[6])

    def test_step(self, batch, batch_idx: int):
        device = self.get_device(batch)

        b_img, b_gt = batch

        prev_actions, prev_log_probs, prev_segmentations = self.agent.get_action(b_img, None, self.net, epsilon=0.0,
                                                                                 device=self.get_device(batch),
                                                                                 sample=False)
        prev_rewards = self.agent.get_reward(b_img, prev_segmentations, None, b_gt, device)

        # calculates training loss
        loss, reward, actions, log_probs, _ = self.compute_policy_loss(
            (b_img, prev_actions, prev_rewards, prev_log_probs, b_gt), sample=False)

        logs = {'test_loss': loss,
                "test_reward": torch.mean(reward.type(torch.float)),
                }
        self.log_tb_images((b_img[0, ...].unsqueeze(0), actions[0, ...].unsqueeze(0), reward[0, ...].unsqueeze(0),
                            prev_actions[0, ...].unsqueeze(0), prev_rewards[0, ...].unsqueeze(0),
                            b_gt[0, ...].unsqueeze(0),
                            batch_idx), prefix='test_')

        self.log_dict(logs)
        return logs


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    logger = TensorBoardLogger('logs', name='simplePPO')

    model = SimplePPO()
    dl = SectorDataModule('/home/local/USHERBROOKE/juda2901/dev/data/icardio/train_subset/',
                          '/home/local/USHERBROOKE/juda2901/dev/data/icardio/train_subset/subset.csv')

    trainer = pl.Trainer(max_epochs=100, logger=logger, log_every_n_steps=1, gpus=1)

    trainer.fit(train_dataloaders=dl, model=model)

    trainer.test(model=model, dataloaders=dl)
