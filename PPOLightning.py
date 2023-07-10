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
from dicenet.dice_net import get_resnet_regression

from tqdm import tqdm


class PPOLightning(pl.LightningModule):

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        sd = torch.load('supervised/supervised_overfit.ckpt')
        self.net = UNet(input_shape=(1, 256, 256), output_shape=(1, 256, 256))
        #self.net.load_state_dict(sd)
        self.net.to(device='cuda:0')

        self.reward_net = get_resnet_regression(input_channels=2)
        sd = torch.load('./dicenet/dice_model_state_dict_10k.ckpt')
        # sd = torch.load('./equalnet/equal_state_dict.ckpt')
        self.reward_net.load_state_dict(sd)
        self.reward_net.to(device='cuda:0')

        self.buffer = ReplayBuffer(capacity=500)
        self.dataset = RLDataset(self.buffer,
                                 data_path='/home/local/USHERBROOKE/juda2901/dev/data/icardio/processed/',
                                 csv_file='/home/local/USHERBROOKE/juda2901/dev/data/icardio/processed/processed.csv',
                                 sample_size=500)
        self.agent = Agent(self.buffer, self.dataset)

        self.epsilon = 0.1

        self.populate(self.buffer, 500)

    def forward(self, x):
        return self.net.forward(x)

    def populate(self, buffer, steps: int = 1000) -> None:
        """
        Carries out several random steps through the environment to initially fill
        up the replay buffer with experiences
        Args:
            steps: number of random steps to populate the buffer with
        """
        print("Populating replay buffer...")
        for i in tqdm(range(steps)):
            self.agent.play_step(buffer, self.net, self.reward_net)

    def train_dataloader(self) -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving experiences"""
        dataloader = DataLoader(dataset=self.dataset,
                                batch_size=32,
                                )
        return dataloader

    def val_dataloader(self) -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving experiences"""
        #val_buffer = ReplayBuffer(100)
        val_dataset = RLDataset(self.buffer,
                                data_path='/home/local/USHERBROOKE/juda2901/dev/data/icardio/processed/',
                                csv_file='/home/local/USHERBROOKE/juda2901/dev/data/icardio/processed/processed.csv',
                                sample_size=16)
        #self.populate(val_buffer, 100)

        dataloader = DataLoader(dataset=val_dataset,
                                batch_size=8,
                                )
        return dataloader

    def test_dataloader(self) -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving experiences"""
        test_buffer = ReplayBuffer(100)
        test_dataset = RLDataset(test_buffer,
                                data_path='/home/local/USHERBROOKE/juda2901/dev/data/icardio/processed/',
                                csv_file='/home/local/USHERBROOKE/juda2901/dev/data/icardio/processed/processed.csv',
                                sample_size=100)
        self.populate(test_buffer, 100)

        dataloader = DataLoader(dataset=test_dataset,
                                batch_size=8,
                                )
        return dataloader
    def get_device(self, batch) -> str:
        """Retrieve device currently being used by minibatch"""
        return batch[0].device.index if self.on_gpu else 'cpu'

    def configure_optimizers(self):
        return torch.optim.Adam(self.net.parameters(), lr=1e-3)

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

        # # step through environment with agent
        # _, _ = self.agent.play_step(self.buffer, self.net, self.reward_net, device=device)

        # calculates training loss
        loss, reward, ratio, actions = self.compute_policy_loss(batch)

        approx_kl_div = torch.mean((torch.exp(ratio) - 1) - ratio)

        logs = {
            'loss': loss,
            'reward': torch.mean(reward.type(torch.float)),
            'ratio': ratio.mean(),
            'approx_kl_div': approx_kl_div,
            #'dice': differentiable_dice_score(actions.type(torch.float), batch[4])
        }

        self.log_dict(logs)
        return OrderedDict({'loss': loss, 'log': logs, 'progress_bar': logs})

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

        reward = self.agent.get_reward(b_imgs, seg, self.reward_net, b_gt, self.get_device(batch))

        # importance ratio
        assert b_log_probs.shape == log_probs.shape
        ratio = (log_probs - b_log_probs).exp()

        # clamp with epsilon value
        clipped = ratio.clamp(1 - self.epsilon, 1 + self.epsilon)

        # min trick
        loss = -torch.min(b_rewards * ratio, b_rewards * clipped).mean()

        for i in range(len(b_imgs)):
            exp = Experience(b_imgs[i, ...].cpu().detach().numpy(),
                             actions[i, ...].unsqueeze(0).cpu().detach().numpy(),
                             reward[i, ...].unsqueeze(0).cpu().detach().numpy(),
                             log_probs[i, ...].cpu().detach().numpy(),
                             b_gt[i, ...].cpu().detach().numpy())

            self.buffer.append(exp)

        return loss, reward, ratio, actions

    def validation_step(self, batch, batch_idx: int):
        device = self.get_device(batch)

        loss, reward, ratio, actions = self.compute_policy_loss(batch, sample=True)

        logs = {'val_loss': loss,
                "val_reward": torch.mean(reward.type(torch.float)),
                }
        #if batch_idx % 0: # Log every 100 batches
        self.log_tb_images((batch[0], actions, reward, batch[1], batch[2], batch[4], batch_idx))

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

        loss, reward, ratio, actions = self.compute_policy_loss(batch, sample=False)

        logs = {'test_loss': loss,
                "test_reward": torch.mean(reward.type(torch.float)),
                }
        self.log_tb_images((batch[0], actions, reward, batch[1], batch[2], batch[4], batch_idx), prefix='test_')

        self.log_dict(logs)
        return logs


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    logger = TensorBoardLogger('logs', name='ppo_reward_net')

    model = PPOLightning()

    trainer = pl.Trainer(max_epochs=100, logger=logger, log_every_n_steps=1, gpus=1)

    trainer.fit(model)

    trainer.test(model)
