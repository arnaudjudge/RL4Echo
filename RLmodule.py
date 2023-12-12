import copy
import random
from datetime import datetime
from pathlib import Path
from typing import Any
from typing import Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from bdicardio.utils.ransac_utils import ransac_sector_extraction
from pytorch_lightning.loggers import TensorBoardLogger
import nibabel as nib
from scipy import ndimage

from rewardnet.data_augmentation import create_random_blobs
from utils.Metrics import accuracy, dice_score
from vital.vital.metrics.train.functional import differentiable_dice_score

from utils.file_utils import save_batch_to_dataset_v2
from utils.logging_helper import log_image


class RLmodule(pl.LightningModule):

    def __init__(self, actor, reward, actor_save_path=None, critic_save_path=None, predict_save_dir=None, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.actor = actor
        self.reward_func = reward

        self.actor_save_path = actor_save_path
        self.critic_save_path = critic_save_path

        self.predict_save_dir = predict_save_dir

    def configure_optimizers(self):
        return self.actor.get_optimizers()

    @torch.no_grad()  # no grad since tensors are reused in PPO's for loop
    def rollout(self, imgs: torch.tensor, gt: torch.tensor, use_gt: torch.tensor = None, sample: bool = True):
        """
            Rollout the policy over a batch of images and ground truth pairs
        Args:
            imgs: batch of images
            gt: batch of ground truth segmentation maps
            use_gt: replace policy result with ground truth (bool mask of len of batch)
            sample: whether to sample actions from distribution or determinist

        Returns:
            Actions (used for rewards, log_pobs, etc), sampled_actions (mainly for display), log_probs, rewards
        """
        actions = self.actor.act(imgs, sample=sample)
        rewards = self.reward_func(actions, imgs, gt.unsqueeze(1))

        # plt.figure()
        # plt.imshow(rewards[0, 0, ...].cpu().numpy())
        # plt.show()

        if use_gt is not None:
            actions[use_gt, ...] = gt.unsqueeze(1)[use_gt, ...]

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
        dice = dice_score(prev_actions, b_gt.unsqueeze(1))

        logs = {'val_loss': loss,
                "val_reward": torch.mean(prev_rewards.type(torch.float)),
                "val_acc": acc.mean(),
                "val_dice": dice.mean()
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
        dice = dice_score(prev_actions, b_gt.unsqueeze(1))

        logs = {'test_loss': loss,
                "test_reward": torch.mean(prev_rewards.type(torch.float)),
                'test_acc': acc.mean(),
                "test_dice": dice.mean()
                }

        # for logging v
        _, _, _, _, v, _ = self.actor.evaluate(b_img, prev_actions)

        for i in range(len(b_img)):
            log_image(self.logger, img=b_img[i], title='test_Image', number=batch_idx * (i + 1))
            log_image(self.logger, img=b_gt[i].unsqueeze(0), title='test_GroundTruth', number=batch_idx * (i + 1))
            log_image(self.logger, img=prev_actions[i], title='test_Prediction', number=batch_idx * (i + 1),
                      img_text=acc[i].mean())
            if v.shape == prev_actions.shape:
                log_image(self.logger, img=v[i], title='test_v_function', number=batch_idx * (i + 1),
                          img_text=v[i].mean())
            if prev_rewards.shape == prev_actions.shape:
                log_image(self.logger, img=prev_rewards[i], title='test_RewardMap', number=batch_idx * (i + 1))

        # save_batch_to_dataset_v2(b_img, b_gt, prev_actions, batch_idx, './simple_reward_net/dataset_supervised/')

        self.log_dict(logs)
        return logs

    def on_test_end(self) -> None:
        if self.actor_save_path:
            torch.save(self.actor.actor.net.state_dict(), self.actor_save_path)
        if self.critic_save_path:
            torch.save(self.actor.critic.net.state_dict(), self.critic_save_path)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        b_img, b_gt, *_ = batch
        actions, _, _ = self.rollout(b_img, b_gt, sample=True)
        actions_unsampled, _, _ = self.rollout(b_img, b_gt, sample=False)
        acc = accuracy(actions_unsampled, b_img, b_gt.unsqueeze(1))
        print((acc > 0.99).sum())
        initial_params = copy.deepcopy(self.actor.actor.net.state_dict())
        #for itr in range(100):
        itr = 0
        for i in range(len(b_img)):
            if acc[i] > 0.99:
                for j, multiplier in enumerate([0.005, 0.01]): #, 0.025, 0.04]):
                    # get random seed based on time to maximise randomness of noise and subsequent predictions
                    # explore as much space around policy as possible
                    time_seed = int(round(datetime.now().timestamp())) + i
                    torch.manual_seed(time_seed)

                    # load initial params so noise is not compounded
                    self.actor.actor.net.load_state_dict(initial_params, strict=True)

                    # add noise to params
                    with torch.no_grad():
                        for param in self.actor.actor.net.parameters():
                            param.add_(torch.randn(param.size()).cuda() * multiplier)

                    # make prediction
                    action, *_ = self.actor.actor(b_img[i].unsqueeze(0))
                    action = torch.round(action)

                    # if np.random.rand() > 0.9:
                    #     blobs = create_random_blobs()
                    #     action = action.cpu().numpy().astype(np.uint8) & ~blobs
                    #     action = torch.tensor(action)

                    # f, (ax1, ax2) = plt.subplots(1, 2)
                    # ax1.set_title(f"Original")
                    # ax1.imshow(actions_unsampled[i, 0, ...].cpu().numpy())
                    #
                    # ax2.set_title(f"{multiplier}, seed: {time_seed}")
                    # ax2.imshow(action[0, 0, ...].cpu().numpy())
                    # plt.show()

                    Path(f"{self.predict_save_dir}/images").mkdir(parents=True, exist_ok=True)
                    Path(f"{self.predict_save_dir}/gt").mkdir(parents=True, exist_ok=True)
                    Path(f"{self.predict_save_dir}/pred").mkdir(parents=True, exist_ok=True)

                    filename = f"{batch_idx}_{itr}_{i}_{time_seed}.nii.gz"
                    affine = np.diag(np.asarray([1, 1, 1, 0]))
                    hdr = nib.Nifti1Header()

                    nifti_img = nib.Nifti1Image(b_img[i].cpu().numpy(), affine, hdr)
                    nifti_img.to_filename(f"./{self.predict_save_dir}/images/{filename}")

                    nifti_gt = nib.Nifti1Image(np.expand_dims(actions_unsampled[i, 0].cpu().numpy(), 0), affine, hdr)
                    nifti_gt.to_filename(f"./{self.predict_save_dir}/gt/{filename}")

                    nifti_pred = nib.Nifti1Image(action[0].cpu().numpy(), affine, hdr)
                    nifti_pred.to_filename(f"./{self.predict_save_dir}/pred/{filename}")
            else:
                try:
                    # Find each blob in the image
                    lbl, num = ndimage.label(actions[i, 0, ...].cpu().numpy())
                    # Count the number of elements per label
                    count = np.bincount(lbl.flat)
                    # Select the largest blob
                    maxi = np.argmax(count[1:]) + 1
                    # Keep only the other blobs
                    lbl[lbl != maxi] = 0
                    ransac, *_ = ransac_sector_extraction(lbl, slim_factor=0.01, circle_center_tol=0.45, plot=False)

                    if ransac.sum() < actions[i, 0, ...].cpu().numpy().sum() * 0.1:
                        continue
                    # f, (ax1, ax2) = plt.subplots(1, 2)
                    # ax1.set_title(f"Original")
                    # ax1.imshow(actions[i, 0, ...].cpu().numpy())
                    #
                    # ax2.set_title(f"ransac")
                    # ax2.imshow(ransac)
                    # plt.show()

                    Path(f"{self.predict_save_dir}/images").mkdir(parents=True, exist_ok=True)
                    Path(f"{self.predict_save_dir}/gt").mkdir(parents=True, exist_ok=True)
                    Path(f"{self.predict_save_dir}/pred").mkdir(parents=True, exist_ok=True)

                    filename = f"{batch_idx}_{itr}_{i}_{random.randint(1, 1000)}.nii.gz"
                    affine = np.diag(np.asarray([1, 1, 1, 0]))
                    hdr = nib.Nifti1Header()

                    nifti_img = nib.Nifti1Image(b_img[i].cpu().numpy(), affine, hdr)
                    nifti_img.to_filename(f"./{self.predict_save_dir}/images/{filename}")

                    nifti_gt = nib.Nifti1Image(np.expand_dims(ransac, 0), affine, hdr)
                    nifti_gt.to_filename(f"./{self.predict_save_dir}/gt/{filename}")

                    nifti_pred = nib.Nifti1Image(actions[i].cpu().numpy(), affine, hdr)
                    nifti_pred.to_filename(f"./{self.predict_save_dir}/pred/{filename}")
                except:
                    pass

        # make sure initial params are back at end of step
        self.actor.actor.net.load_state_dict(initial_params)


