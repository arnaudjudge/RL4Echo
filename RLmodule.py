import copy
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pytorch_lightning as pl
import torch
from torch import Tensor
from vital.metrics.camus.anatomical.utils import check_segmentation_validity

from utils.Metrics import accuracy, dice_score, is_anatomically_valid
from utils.file_utils import get_img_subpath, save_to_reward_dataset
from utils.logging_helper import log_image
from utils.tensor_utils import convert_to_numpy


class RLmodule(pl.LightningModule):

    def __init__(self, actor, reward, corrector=None, actor_save_path=None, critic_save_path=None, predict_save_dir=None, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.actor = actor
        self.reward_func = reward

        self.actor_save_path = actor_save_path
        self.critic_save_path = critic_save_path

        self.predict_save_dir = predict_save_dir
        self.pred_corrector = corrector

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
        rewards = self.reward_func(actions, imgs, gt)

        if use_gt is not None:
            actions[use_gt, ...] = gt[use_gt, ...]

        _, _, log_probs, _, _, _ = self.actor.evaluate(imgs, actions)
        return actions, log_probs, rewards

    def training_step(self, batch: dict[str, Tensor], nb_batch):
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

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int):
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
        b_img, b_gt, b_use_gt = batch['img'], batch['gt'], batch['use_gt']

        prev_actions, prev_log_probs, prev_rewards = self.rollout(b_img, b_gt)

        loss, critic_loss, metrics_dict = self.compute_policy_loss((b_img, prev_actions, prev_rewards,
                                                                    prev_log_probs, b_gt, b_use_gt))

        acc = accuracy(prev_actions, b_img, b_gt)
        dice = dice_score(prev_actions, b_gt)

        logs = {'val_loss': loss,
                "val_reward": torch.mean(prev_rewards.type(torch.float)),
                "val_acc": acc.mean(),
                "val_dice": dice.mean()
                }

        # log images
        idx = random.randint(0, len(b_img) - 1)  # which image to log
        log_image(self.logger, img=b_img[idx].permute((0, 2, 1)), title='Image', number=batch_idx)
        log_image(self.logger, img=b_gt[idx].unsqueeze(0).permute((0, 2, 1)), title='GroundTruth', number=batch_idx)
        log_image(self.logger, img=prev_actions[idx].unsqueeze(0).permute((0, 2, 1)), title='Prediction', number=batch_idx,
                  img_text=prev_rewards[idx].mean())
        if prev_rewards.shape == prev_actions.shape:
            log_image(self.logger, img=prev_rewards[idx].unsqueeze(0).permute((0, 2, 1)), title='RewardMap', number=batch_idx)

        self.log_dict(logs)
        return logs

    def test_step(self, batch: dict[str, Tensor], batch_idx: int):
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
        b_img, b_gt, b_use_gt = batch['img'], batch['gt'], batch['use_gt']

        prev_actions, prev_log_probs, prev_rewards = self.rollout(b_img, b_gt, sample=False)
        loss, critic_loss, metrics_dict = self.compute_policy_loss((b_img, prev_actions, prev_rewards,
                                                                    prev_log_probs, b_gt, b_use_gt))
        acc = accuracy(prev_actions, b_img, b_gt)
        dice = dice_score(prev_actions, b_gt)
        anat_errors = is_anatomically_valid(prev_actions)

        logs = {'test_loss': loss,
                "test_reward": torch.mean(prev_rewards.type(torch.float)),
                'test_acc': acc.mean(),
                "test_dice": dice.mean(),
                "test_anat_valid": anat_errors.mean()
                }

        # for logging v
        _, _, _, _, v, _ = self.actor.evaluate(b_img, prev_actions)

        for i in range(len(b_img)):
            log_image(self.logger, img=b_img[i].permute((0, 2, 1)), title='test_Image', number=batch_idx * (i + 1))
            log_image(self.logger, img=b_gt[i].unsqueeze(0).permute((0, 2, 1)), title='test_GroundTruth', number=batch_idx * (i + 1))
            log_image(self.logger, img=prev_actions[i].unsqueeze(0).permute((0, 2, 1)), title='test_Prediction', number=batch_idx * (i + 1),
                      img_text=dice[i].mean())
            if v.shape == prev_actions.shape:
                log_image(self.logger, img=v[i].unsqueeze(0).permute((0, 2, 1)), title='test_v_function', number=batch_idx * (i + 1),
                          img_text=v[i].mean())
            if prev_rewards.shape == prev_actions.shape:
                log_image(self.logger, img=prev_rewards[i].unsqueeze(0).permute((0, 2, 1)), title='test_RewardMap', number=batch_idx * (i + 1))

        self.log_dict(logs)
        return logs

    def on_test_end(self) -> None:
        if self.actor_save_path:
            torch.save(self.actor.actor.net.state_dict(), self.actor_save_path)
        if self.critic_save_path:
            torch.save(self.actor.critic.net.state_dict(), self.critic_save_path)

    def predict_step(self, batch: dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0) -> Any:
        b_img, ids, inst = batch['img'], batch['id'], batch.get('instant', None)

        actions, _, _ = self.rollout(b_img, torch.zeros_like(b_img).squeeze(1), sample=True)
        actions_unsampled, _, _ = self.rollout(b_img, torch.zeros_like(b_img).squeeze(1), sample=False)

        corrected, corrected_validity, ae_comp = self.pred_corrector.correct_batch(b_img, actions_unsampled)

        initial_params = copy.deepcopy(self.actor.actor.net.state_dict())
        itr = 0

        for i in range(len(b_img)):
            self.trainer.datamodule.add_to_train(ids[i], inst[i] if inst else None)
            if ae_comp[i] > 0.95 and check_segmentation_validity(actions_unsampled[i].cpu().numpy().T, (1.0, 1.0),
                                                                 [0, 1, 2]):
                self.trainer.datamodule.add_to_gt(ids[i], inst[i] if inst else None)

                path = self.trainer.datamodule.get_approx_gt_subpath(ids[i], inst[i] if inst else None)
                approx_gt_path = self.trainer.datamodule.hparams.approx_gt_dir + '/approx_gt/' + path
                Path(approx_gt_path).parent.mkdir(parents=True, exist_ok=True)
                hdr = nib.Nifti1Header()
                nifti = nib.Nifti1Image(torch.round(actions_unsampled[i]).cpu().numpy(), np.diag(np.asarray([-1, -1, 1, 0])), hdr)
                nifti.to_filename(approx_gt_path)

                # force actions to resemble image?
                filename = f"{batch_idx}_{itr}_{i}_wrong_s_{int(round(datetime.now().timestamp()))}.nii.gz"
                save_to_reward_dataset(self.predict_save_dir,
                                       filename,
                                       convert_to_numpy(b_img[i]),
                                       np.expand_dims(convert_to_numpy(actions_unsampled[i]), 0),
                                       np.expand_dims(convert_to_numpy(actions[random.randint(0, len(b_img)-1)]), 0))

                for j, multiplier in enumerate([0.005, 0.01, 0.015, 0.02]):
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
                    deformed_action, *_ = self.actor.actor(b_img[i].unsqueeze(0))
                    if len(deformed_action.shape) > 3:
                        deformed_action = deformed_action.argmax(dim=1)
                    else:
                        deformed_action = torch.round(deformed_action)

                    # f, (ax1, ax2) = plt.subplots(1, 2)
                    # ax1.set_title(f"Good initial action")
                    # ax1.imshow(actions_unsampled[i, ...].cpu().numpy().T)
                    #
                    # ax2.set_title(f"Deformed network's action")
                    # ax2.imshow(deformed_action[0, ...].cpu().numpy().T)
                    # plt.show()

                    filename = f"{batch_idx}_{itr}_{i}_{time_seed}.nii.gz"
                    save_to_reward_dataset(self.predict_save_dir,
                                           filename,
                                           convert_to_numpy(b_img[i]),
                                           np.expand_dims(convert_to_numpy(actions_unsampled[i]), 0),
                                           convert_to_numpy(deformed_action))
            else:
                if not corrected_validity[i]:
                    continue

                # f, (ax1, ax2) = plt.subplots(1, 2)
                # ax1.set_title(f"Bad initial action")
                # ax1.imshow(actions[i, ...].cpu().numpy().T)
                #
                # ax2.set_title(f"Corrected action")
                # ax2.imshow(corrected[i, ...].T)
                # plt.show()

                filename = f"{batch_idx}_{itr}_{i}_{int(round(datetime.now().timestamp()))}.nii.gz"
                save_to_reward_dataset(self.predict_save_dir,
                                       filename,
                                       convert_to_numpy(b_img[i]),
                                       convert_to_numpy(corrected[i]),
                                       np.expand_dims(convert_to_numpy(actions[i]), 0))

        self.trainer.datamodule.update_dataframe()
        # make sure initial params are back at end of step
        self.actor.actor.net.load_state_dict(initial_params)


