import copy
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Union

import SimpleITK as sitk
import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torchio as tio
from einops import rearrange
from lightning import LightningModule
from monai.data import MetaTensor
from scipy import ndimage
from torch import Tensor
from torch.nn.functional import pad
from torchvision.transforms.functional import adjust_contrast, rotate

from patchless_nnunet.utils.inferers import SlidingWindowInferer
from patchless_nnunet.utils.softmax import softmax_helper
from rl4echo.utils.Metrics import accuracy, dice_score
from rl4echo.utils.correctors import AEMorphoCorrector
from rl4echo.utils.file_utils import save_to_reward_dataset
from rl4echo.utils.logging_helper import log_sequence, log_video
from rl4echo.utils.tensor_utils import convert_to_numpy
from rl4echo.utils.test_metrics import full_test_metrics
from vital.metrics.camus.anatomical.utils import check_segmentation_validity

import matplotlib.pyplot as plt


def shrink_perturb(model, lamda=0.5, sigma=0.01):
    for (name, param) in model.named_parameters():
        if 'weight' in name:  # just weights
            # nc = param.shape[0]  # cols
            # nr = param.shape[1]  # rows
            # for i in range(nr):
            #     for j in range(nc):
            param.data = \
                (lamda * param.data) + \
                torch.normal(0.0, sigma, size=(param.data.shape), device=next(model.parameters()).device)
    return model


class RLmodule3D(LightningModule):

    def __init__(self, actor, reward,
                 corrector=None,
                 actor_save_path=None,
                 critic_save_path=None,
                 save_uncertainty_path=None,
                 predict_save_dir=None,
                 predict_do_model_perturb=True,
                 predict_do_img_perturb=True,
                 predict_do_corrections=True,
                 predict_do_temporal_glitches=True,
                 save_on_test=True,
                 vae_on_test=False,
                 worst_frame_thresholds=None, #{"anatomical": 0.985},
                 save_csv_after_predict=None,
                 val_batch_size=4,
                 tta=True,
                 temp_files_path='.',
                 *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.save_hyperparameters(logger=False, ignore=["actor", "reward", "corrector"])

        self.actor = actor
        self.reward_func = reward
        if hasattr(self.reward_func, 'net'):
            self.register_module('rewardnet', self.reward_func.net)
        elif hasattr(self.reward_func, 'nets'):
            if isinstance(self.reward_func.nets, list):  # backward compatibility
                for i, n in enumerate(self.reward_func.nets):
                    self.register_module(f'rewardnet_{i}', n)
            elif isinstance(self.reward_func.nets, dict):
                for i, n in enumerate(self.reward_func.nets.values()):
                    self.register_module(f'rewardnet_{i}', n)

        self.pred_corrector = corrector
        if isinstance(self.pred_corrector, AEMorphoCorrector):
            self.register_module('correctorAE', self.pred_corrector.ae_corrector.temporal_regularization.autoencoder)

        self.predicted_rows = []

        self.temp_files_path = Path(temp_files_path)
        if not self.temp_files_path.exists() and self.trainer.global_rank == 0:
            self.temp_files_path.mkdir(parents=True, exist_ok=True)

        # for test time overfitting
        self.initial_test_params = None

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

    def ttoverfit(self, batch_image, num_iter=10, **kwargs):
        """
            Run a few iterations of optimization to overfit on one test sequence in unsupervised
        Args:
            batch: batch of images
            num_iter: number of iterations of the optimization loop
        Returns:
            None, actor is modified, ready for inference on batch_image alone
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
        b_imgs, b_gts, b_use_gts = batch['img'].squeeze(0), batch['gt'].squeeze(0), batch['use_gt'].squeeze(0)

        logs = {'val/loss': [],
                "val/reward": [],
                "val/acc": [],
                "val/dice": []
                }
        for i in range(0, b_imgs.shape[0], self.hparams.val_batch_size):
            b_img = b_imgs[i:i+self.hparams.val_batch_size]
            b_gt = b_gts[i:i+self.hparams.val_batch_size]
            b_use_gt = b_use_gts[i:i+self.hparams.val_batch_size]

            prev_actions, prev_log_probs, prev_rewards = self.rollout(b_img, b_gt, sample=False)
            prev_rewards = torch.mean(torch.stack(prev_rewards, dim=0), dim=0)

            loss, critic_loss, metrics_dict = self.compute_policy_loss((b_img, prev_actions, prev_rewards,
                                                                        prev_log_probs, b_gt, b_use_gt))

            acc = accuracy(prev_actions, b_img, b_gt)
            dice = dice_score(prev_actions, b_gt)

            logs["val/loss"] += [loss]
            logs["val/reward"] += [torch.mean(prev_rewards.type(torch.float))]
            logs["val/acc"] += [acc.mean()]
            logs["val/dice"] += [dice.mean()]

            _, _, _, _, v, _ = self.actor.evaluate(b_img, prev_actions)
            # log images
            if self.trainer.global_rank == 0:
                idx = random.randint(0, len(b_img) - 1)  # which image to log
                log_sequence(self.logger, img=b_img[idx], title='Image', number=batch_idx, epoch=self.current_epoch)
                log_sequence(self.logger, img=b_gt[idx].unsqueeze(0), title='GroundTruth', number=batch_idx,
                             epoch=self.current_epoch)
                log_sequence(self.logger, img=prev_actions[idx].unsqueeze(0), title='Prediction', number=batch_idx,
                          img_text=prev_rewards[idx].mean(), epoch=self.current_epoch)
                log_sequence(self.logger, img=v[idx].unsqueeze(0), title='VFunction', number=batch_idx,
                             img_text=v[idx].mean(), epoch=self.current_epoch)
                if prev_rewards.shape == prev_actions.shape:
                    log_sequence(self.logger, img=prev_rewards[idx].unsqueeze(0), title='RewardMap', number=batch_idx,
                                 epoch=self.current_epoch)

        logs = {k: torch.tensor(v, device=self.device).mean() for k, v in logs.items()}
        self.log_dict(logs, on_epoch=True, sync_dist=True)
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
        b_img, b_gt, meta_dict = batch['img'], batch['gt'], batch['image_meta_dict']

        # b_img = adjust_contrast(b_img.permute((4, 0, 1, 2, 3)), 0.4).permute((1, 2, 3, 4, 0))
        # b_img += torch.randn(b_img.size()).to(next(self.actor.actor.net.parameters()).device) * 0.1
        # b_img /= b_img.max()

        self.patch_size = list([b_img.shape[-3], b_img.shape[-2], 4])
        self.inferer.roi_size = self.patch_size
        #
        # dicom_list = [
        #     "di-0CA9-8FC9-25BB", "di-1F8E-37E5-57ED",
        #     "di-1FD4-CB18-6EFC",
        #     "di-5A75-9C41-46D7",
        #     "di-5BB1-44DE-60BA",
        #     "di-62CF-F093-A156", "di-63D5-7095-7FB9", "di-A540-DBDD-7C1F", "di-AE68-A41B-5185", "di-C910-B188-A16F",
        #     "di-EB49-9AE0-F0D0", "di-6CF6-853C-CB97", "di-16B9-402D-037F", "di-46D0-7328-762F", "di-47EB-1516-2456",
        #     "di-7041-D238-665F", "di-8254-6AD3-C7FC", "di-A984-8D28-57F4", "di-B55F-84D9-6833", "di-C9E0-1668-D365",
        #     "di-E42E-19EA-16E7"
        # ]

        # filtered_dicom_list9655 =  ['di-62CF-F093-A156', 'di-0CA9-8FC9-25BB', 'di-AE68-A41B-5185', 'di-6781-303D-8227',
        # 'di-47EB-1516-2456', 'di-6C37-9F35-0D42', 'di-4220-E23D-CE04', 'di-63D5-7095-7FB9', 'di-16B9-402D-037F',
        # 'di-E42E-19EA-16E7', 'di-8254-6AD3-C7FC', 'di-46D0-7328-762F', 'di-1FD4-CB18-6EFC', 'di-BBFD-BFA0-F9FA',
        # 'di-1E9C-3E1D-87E3', 'di-1CE1-2EF7-3142', 'di-B55F-84D9-6833', 'di-1ECA-EAC3-8EAE', 'di-1599-6138-451C',
        #                             'di-5F58-9EFC-734D', 'di-3B2D-9F8D-8A3A']
        # filtered_dicom_list9655 = ['di-6C37-9F35-0D42']

        # filtered_dicom_list97 = ['di-62CF-F093-A156', 'di-0CA9-8FC9-25BB', 'di-AE68-A41B-5185', 'di-6781-303D-8227', 'di-47EB-1516-2456',
        #  'di-6C37-9F35-0D42', 'di-B17B-8983-52D9', 'di-3E0E-FE3B-8A32', 'di-4220-E23D-CE04', 'di-63D5-7095-7FB9',
        #  'di-16B9-402D-037F', 'di-E42E-19EA-16E7', 'di-8254-6AD3-C7FC', 'di-46D0-7328-762F', 'di-1FD4-CB18-6EFC',
        #  'di-BBFD-BFA0-F9FA', 'di-1E9C-3E1D-87E3', 'di-1CE1-2EF7-3142', 'di-B10E-9CBE-3867', 'di-1E5A-7083-58BC',
        #  'di-B55F-84D9-6833', 'di-1ECA-EAC3-8EAE', 'di-1599-6138-451C', 'di-7291-29A4-97F8', 'di-5F58-9EFC-734D',
        #  'di-3B2D-9F8D-8A3A']

        # dicom_list = ['di-5A75-9C41-46D7']

        dicom_list_AVTV = ['di-63D5-7095-7FB9']
        # ['di-C910-B188-A16F', 'di-1063-8366-5D88', 'di-0CA9-8FC9-25BB', 'di-AE68-A41B-5185',
        #                    'di-47EB-1516-2456', 'di-5A75-9C41-46D7', 'di-7798-9454-7280', 'di-8075-490B-DF12',
        #                    'di-C9E0-1668-D365', 'di-E42E-19EA-16E7', 'di-9BBB-E0E8-ABFE', 'di-46D0-7328-762F',
        #                    'di-0343-2269-9415', 'di-1FD4-CB18-6EFC', 'di-A540-DBDD-7C1F', 'di-056D-EBB0-1AF2',
        #                    'di-BBFD-BFA0-F9FA', 'di-1F8E-37E5-57ED', 'di-C5EF-3C1E-3B51', 'di-B55F-84D9-6833',
        #                    'di-5BB1-44DE-60BA', 'di-7291-29A4-97F8']

        if batch['image_meta_dict']['case_identifier'][0] in dicom_list_AVTV:
            start_time = time.time()
            first_prev_actions = self.tta_predict(b_img) if self.hparams.tta else self.predict(b_img).argmax(dim=1)
            print(f"\nFirst Prediction took {round(time.time() - start_time, 4)} (s).")

            prev_rewards = torch.stack(self.reward_func.predict_full_sequence(first_prev_actions, b_img, b_gt), dim=0)
            prev_rewards_mean = torch.mean(prev_rewards, dim=0)
            prev_rewards_min = torch.minimum(prev_rewards[0], prev_rewards[1])

            # reward_frame_min = prev_rewards_min.cpu().numpy().mean(axis=(0, 1, 2)).argmin()
        # reward_frame_min = prev_rewards_min.cpu().numpy().mean(axis=(0, 1, 2)).min()
        # print(batch['image_meta_dict']['case_identifier'][0], reward_frame_min, prev_rewards_mean.mean())
        # thresholds = 0.95
        # thresh_validated = reward_frame_min > thresholds

        # start_time = time.time()
        # y_pred_np_as_batch = prev_actions.cpu().numpy().squeeze(0).transpose((2, 0, 1))
        # b_gt_np_as_batch = b_gt.cpu().numpy().squeeze(0).transpose((2, 0, 1))
        #
        # for i in range(len(y_pred_np_as_batch)):
        #     lbl, num = ndimage.measurements.label(y_pred_np_as_batch[i] != 0)
        #     # Count the number of elements per label
        #     count = np.bincount(lbl.flat)
        #     # Select the largest blob
        #     maxi = np.argmax(count[1:]) + 1
        #     # Remove the other blobs
        #     y_pred_np_as_batch[i][lbl != maxi] = 0
        #
        # # should be still valid to use resampled spacing for metrics here
        # voxel_spacing = np.asarray([[abs(meta_dict['resampled_affine'][0, 0, 0].cpu().numpy()),
        #                              abs(meta_dict['resampled_affine'][0, 1, 1].cpu().numpy())]]).repeat(
        #                             repeats=len(y_pred_np_as_batch), axis=0)
        # print(f"Cleaning took {round(time.time() - start_time, 4)} (s).")
        # from rl4echo.utils.Metrics import is_anatomically_valid
        # from rl4echo.utils.temporal_metrics import check_temporal_validity
        # anat_errors = is_anatomically_valid(y_pred_np_as_batch)
        # temporal_valid, _ = check_temporal_validity(y_pred_np_as_batch.transpose((0, 2, 1)),
        #                                                               voxel_spacing[0])
        # validated = int(all(anat_errors)) and temporal_valid and thresh_validated
        # if not validated:
            print(batch['image_meta_dict']['case_identifier'][0])
            self.actor.actor.net.load_state_dict(self.initial_test_params, strict=False)

            start_time = time.time()
            self.ttoverfit(b_img)
            print(f"\nTTOverfit took {round(time.time() - start_time, 4)} (s).")

            start_time = time.time()
            prev_actions = self.tta_predict(b_img) if self.hparams.tta else self.predict(b_img).argmax(dim=1)
            print(f"\nPrediction took {round(time.time() - start_time, 4)} (s).")

            # f, ax = plt.subplots(1, 3, figsize=(12, 6))
            # ax[0].imshow(b_img[0, 0, ..., reward_frame_min].cpu().numpy().T, cmap='gray')
            # from matplotlib.colors import LinearSegmentedColormap
            # custom_cmap = LinearSegmentedColormap.from_list("custom", [(0, 0, 0), (0, 1, 0), (1, 0, 0)], N=3)
            # ax[0].imshow(first_prev_actions[0, ..., reward_frame_min].cpu().numpy().T, alpha=0.4, cmap=custom_cmap)
            #
            # # ax[1].imshow(first_prev_actions[0, ..., reward_frame_min].cpu().numpy().T, cmap=custom_cmap)
            # ax[1].imshow(prev_rewards_min[0, ..., reward_frame_min].cpu().numpy().T, cmap='gray', vmin=0, vmax=1)
            #
            # ax[2].imshow(b_img[0, 0, ..., reward_frame_min].cpu().numpy().T, cmap='gray')
            # ax[2].imshow(prev_actions[0, ..., reward_frame_min].cpu().numpy().T, alpha=0.4, cmap=custom_cmap)
            #
            # for a in ax:
            #     a.get_xaxis().set_visible(False)
            #     a.get_yaxis().set_visible(False)
            # ax[0].set_title("Initial Prediction", fontsize=12)
            # ax[1].set_title("Reward Map", fontsize=12)
            # ax[2].set_title("TTO Prediction", fontsize=12)
            # plt.savefig(f"{batch['image_meta_dict']['case_identifier'][0]}_TTO.png")
            # plt.show()


        else:
            return {"reward": 0}
        if batch['image_meta_dict']['case_identifier'][0] in dicom_list_AVTV:
            f, axs = plt.subplots(1, 3, animated=True, figsize=(12, 6), tight_layout=True)
            ax = axs[0]
            bk = ax.imshow(b_img.cpu().numpy()[0, 0, ..., 0].T, cmap='gray')
            from matplotlib.colors import LinearSegmentedColormap
            custom_cmap = LinearSegmentedColormap.from_list("custom", [(0, 0, 0), (0, 1, 0), (1, 0, 0)], N=3)
            # custom_cmap2 = LinearSegmentedColormap.from_list("custom2", [(0, 0, 0), (1, 0, 0), (1, 1, 0)], N=3)
            p1 = ax.imshow(first_prev_actions.cpu().numpy()[0, ..., 0].T, cmap=custom_cmap, alpha=0.4)
            bk2 = axs[2].imshow(b_img.cpu().numpy()[0, 0, ..., 0].T, cmap='gray')
            p2 = axs[2].imshow(prev_actions.cpu().numpy()[0, ..., 0].T, cmap=custom_cmap, alpha=0.4)

            # prev_rewards = torch.stack(self.reward_func.predict_full_sequence(prev_actions, b_img, b_gt), dim=0)
            # prev_rewards = torch.minimum(prev_rewards[0], prev_rewards[1])

            rew = axs[1].imshow(prev_rewards_min.cpu().numpy()[0, ..., 0].T, cmap='gray', vmin=0, vmax=1)

            axs[0].set_title("Initial Prediction", fontsize=12)
            axs[1].set_title("Reward Map", fontsize=12)
            axs[2].set_title("TTO Prediction", fontsize=12)

            for a in axs:
                a.get_xaxis().set_visible(False)
                a.get_yaxis().set_visible(False)
            def update(i):
                bk.set_array(b_img.cpu().numpy()[0, 0, ..., i].T)
                bk2.set_array(b_img.cpu().numpy()[0, 0, ..., i].T)
                p1.set_array(first_prev_actions.cpu().numpy()[0, ..., i].T)
                p2.set_array(prev_actions.cpu().numpy()[0, ..., i].T)
                rew.set_array(prev_rewards_min.cpu().numpy()[0, ..., i].T)
                return bk, bk2, p1, p2, rew

            from matplotlib import animation
            animation_fig = animation.FuncAnimation(f, update, frames=b_img.shape[-1], interval=100, blit=False,
                                                    repeat_delay=10, )
            animation_fig.save(f"./TTO_AVTV/{batch['image_meta_dict']['case_identifier'][0]}.gif")
            # plt.show()
            plt.close()

        prev_rewards = torch.stack(self.reward_func.predict_full_sequence(prev_actions, b_img, b_gt), dim=0)
        prev_rewards_mean = torch.mean(prev_rewards, dim=0)

        # MAKE REWARD FIG HERE
        # import matplotlib.pyplot as plt
        # from rl4echo.utils.temporal_metrics import get_temporal_consistencies
        # pred_as_b = prev_actions.cpu().numpy().squeeze(0).transpose((2, 0, 1))
        #
        # fig, axes = plt.subplots(1, 5, tight_layout=True, figsize=(16,4))
        # bk = axes[0].imshow(b_img.cpu().numpy()[0, ..., 0].T, animated=True, cmap='gray', interpolation='none')
        # from matplotlib.colors import LinearSegmentedColormap
        # custom_cmap = LinearSegmentedColormap.from_list("custom", [(0, 0, 0), (0, 1, 0), (1, 0, 0)], N=3)
        # im = axes[0].imshow(pred_as_b[0].T, animated=True, alpha=0.4, cmap=custom_cmap)
        # axes[0].set_title("Segmentation Map")
        # anat = axes[1].imshow(prev_rewards[0].cpu().numpy()[0, ..., 0].T, animated=True, cmap='gray', vmin=0, vmax=1)
        # axes[1].set_title("Anatomical Reward")
        # prev_rewards[1] = ((prev_rewards[1] - prev_rewards[1].min()) / (prev_rewards[1].max()-prev_rewards[1].min()))
        # lm = axes[2].imshow(prev_rewards[1].cpu().numpy()[0, ..., 0].T, animated=True, cmap='gray', vmin=0, vmax=1)
        # axes[2].set_title("Landmark Reward")
        # merged_rew = torch.minimum(prev_rewards[0], prev_rewards[1])
        # merged = axes[3].imshow(merged_rew.cpu().numpy()[0, ..., 0].T, animated=True, cmap='gray', vmin=0, vmax=1)
        # axes[3].set_title("Merged Reward Map")
        # from rl4echo.utils.temporal_metrics import get_temporal_consistencies
        # import scipy
        # temp_constistencies, measures_1d = get_temporal_consistencies(pred_as_b, skip_measurement_metrics=True)
        # temp_constistencies = scipy.ndimage.gaussian_filter1d(
        #     np.array(list(temp_constistencies.values())).astype(np.float), 1.1, axis=1)
        # tempo_rew = merged_rew.clone().cpu().numpy()[0]
        # temp_constistencies = torch.tensor(temp_constistencies).mean(dim=0)
        # tc_penalty = torch.ones(len(temp_constistencies)) + (temp_constistencies / temp_constistencies.max() * 0.01)
        # tc_penalty.cpu().numpy()
        #
        # print(len(tc_penalty))
        #
        # for j in range(tempo_rew.shape[-1]):
        #     frame_penalty =  tc_penalty[j].item()
        #     if frame_penalty != 1:
        #         tempo_rew[..., j] = scipy.ndimage.gaussian_filter(tempo_rew[..., j], sigma=frame_penalty * 3.5)
        #         tempo_rew[..., j] = tempo_rew[..., j] - tempo_rew[..., j].min()
        #         tempo_rew[..., j] = tempo_rew[..., j] / tempo_rew[..., j].max()
        #
        # tempo = axes[4].imshow(tempo_rew[..., 0].T, animated=True, cmap='gray', vmin=0, vmax=1)
        # axes[4].set_title("Merged w/ Temporal Penalty")
        #
        # for a in axes:
        #     a.get_xaxis().set_visible(False)
        #     a.get_yaxis().set_visible(False)
        # def update(i):
        #     fig.suptitle(f"Frame {i}")
        #     bk.set_array(b_img.cpu().numpy()[0, ..., i].T)
        #     im.set_array(pred_as_b[i].T)
        #     anat.set_array(prev_rewards[0].cpu().numpy()[0, ..., i].T)
        #     lm.set_array(prev_rewards[1].cpu().numpy()[0, ..., i].T)
        #     merged.set_array(merged_rew.cpu().numpy()[0, ..., i].T)
        #     tempo.set_array(tempo_rew[..., i].T)
        #     return bk, im, anat, lm, merged, tempo
        #
        # from matplotlib import animation
        #
        # animation_fig = animation.FuncAnimation(fig, update, frames=prev_rewards.shape[-1], interval=100, blit=False,
        #                                         repeat_delay=10, )
        # animation_fig.save("REWARD_ANIMFIG_di-30F6-5C3B-EA3A.gif")
        # plt.show()
        #

        start_time = time.time()
        y_pred_np_as_batch = prev_actions.cpu().numpy().squeeze(0).transpose((2, 0, 1))
        b_gt_np_as_batch = b_gt.cpu().numpy().squeeze(0).transpose((2, 0, 1))

        for i in range(len(y_pred_np_as_batch)):
            lbl, num = ndimage.measurements.label(y_pred_np_as_batch[i] != 0)
            # Count the number of elements per label
            count = np.bincount(lbl.flat)
            # Select the largest blob
            maxi = np.argmax(count[1:]) + 1
            # Remove the other blobs
            y_pred_np_as_batch[i][lbl != maxi] = 0

        # should be still valid to use resampled spacing for metrics here
        voxel_spacing = np.asarray([[abs(meta_dict['resampled_affine'][0, 0, 0].cpu().numpy()),
                                     abs(meta_dict['resampled_affine'][0, 1, 1].cpu().numpy())]]).repeat(
                                    repeats=len(y_pred_np_as_batch), axis=0)
        print(f"Cleaning took {round(time.time() - start_time, 4)} (s).")

        # logs = full_test_metrics(y_pred_np_as_batch, b_gt_np_as_batch, voxel_spacing, self.device)
        # logs.update({"test/reward": torch.mean(prev_rewards_mean.type(torch.float))})
        logs = {"test/reward": torch.mean(prev_rewards_mean.type(torch.float))}
        print(logs)

        if self.hparams.vae_on_test:
            start_time = time.time()
            corrected, corrected_validity, ae_comp, _ = self.pred_corrector.correct_single_seq(
                b_img.squeeze(0), prev_actions.squeeze(0), voxel_spacing)
            # actions_unsampled_clean = actions_unsampled_clean[None,]
            corrected = corrected.transpose((2, 0, 1))
            vae_logs = full_test_metrics(corrected, b_gt_np_as_batch, voxel_spacing, self.device, prefix="test_vae", verbose=False)
            vae_logs.update({"test_vae/vae_comp": ae_comp})
            logs.update(vae_logs)
            print(f"VAE took {round(time.time() - start_time, 4)} (s).")

        if self.hparams.worst_frame_thresholds:
            # skip if rewards are too low according to thresholds
            reward_indices = [self.reward_func.get_reward_index(key) for key in self.hparams.worst_frame_thresholds.keys()]
            reward_frame_mins = np.take(prev_rewards.cpu().numpy().mean(axis=(1, 2, 3)).min(axis=1), reward_indices, 0)
            thresholds = np.asarray(list(self.hparams.worst_frame_thresholds.values()))
            validated = (reward_frame_mins > thresholds).all()
            if validated:
                fname = meta_dict.get('case_identifier')[0]
                print(f"{self.trainer.datamodule.get_approx_gt_subpath(fname).rsplit('/', 1)[0]}/{fname} - "
                      f"Min frame reward higher than threshold: "
                      f"{reward_frame_mins} vs  thresh:{thresholds}")
                logs.update({f'{k.replace("test", "test_validated")}': v for k, v in logs.items()})
                self.log("test_validated/count", 1)
            else:
                self.log("test_validated/count", 0)

        start_time = time.time()
        # for logging v
        # Use only first 4 for visualization, avoids having to implement sliding window inference for critic
        _, _, _, _, v, _ = self.actor.evaluate(b_img[..., :4], prev_actions[..., :4])

        if self.trainer.global_rank == 0 and batch_idx % 1 == 1232314:
            log_video(self.logger, img=b_gt, background=b_img.squeeze(0), title='test_GroundTruth', number=batch_idx,
                         epoch=self.current_epoch)
            log_video(self.logger, img=prev_actions, background=b_img.squeeze(0), title='test_Prediction',
                         number=batch_idx, epoch=self.current_epoch)
            if v.shape == prev_actions[..., :4].shape:
                log_sequence(self.logger, img=v, title='test_v_function', number=batch_idx,
                          img_text=v.mean(), epoch=self.current_epoch)
            log_video(self.logger, img=prev_rewards_mean, title='test_RewardMap',
                      number=batch_idx, epoch=self.current_epoch)
            if self.hparams.vae_on_test:
                log_video(self.logger, img=corrected.transpose((1, 2, 0))[None,], background=b_img.squeeze(0),
                          title='test_VAE_corrected', number=batch_idx, epoch=self.current_epoch)

        self.log_dict(logs, sync_dist=True)
        print(f"Logging took {round(time.time() - start_time, 4)} (s).")
        # import h5py
        # with h5py.File('3d_anatomical_reward_LM+ANAT+T_LAST_TEMPSCALED.h5', 'a') as f:
        #     for i in range(len(b_img)):
        #         dicom = meta_dict.get("case_identifier")[0]
        #         if dicom not in f:
        #             f.create_group(dicom)
        #         f[dicom]['img'] = (b_img[i].cpu().numpy().squeeze(0) * 255 ).astype(np.uint8)
        #         f[dicom]['gt'] = b_gt[i].cpu().numpy().astype(np.uint8)
        #         f[dicom]['pred'] = y_pred_np_as_batch.transpose((1, 2, 0)).astype(np.uint8)
        #         clean_reward = self.reward_func.predict_full_sequence(torch.tensor(y_pred_np_as_batch.transpose((1, 2, 0))[None,], device=self.device), b_img, b_gt)
        #         f[dicom]['reward_map'] = clean_reward[0][i].cpu().numpy() #prev_rewards_mean[i].cpu().numpy()
        #         f[dicom]['accuracy_map'] = (y_pred_np_as_batch.transpose((1, 2, 0)) != b_gt[i].cpu().numpy()).astype(np.uint8)

        if self.hparams.save_on_test:
            #prev_actions = prev_actions.squeeze(0).cpu().detach().numpy()
            prev_actions = y_pred_np_as_batch.transpose((1, 2, 0))
            original_shape = meta_dict.get("original_shape").cpu().detach().numpy()[0]

            fname = meta_dict.get("case_identifier")[0]
            spacing = meta_dict.get("original_spacing").cpu().detach().numpy()[0]
            resampled_affine = meta_dict.get("resampled_affine").cpu().detach().numpy()[0]
            save_dir = os.path.join(self.trainer.default_root_dir, f"testing_raw_LM+ANAT_TTO_AVTV_NEW/{self.trainer.datamodule.get_approx_gt_subpath(fname).rsplit('/', 1)[0]}/")

            final_preds = np.expand_dims(prev_actions, 0)
            transform = tio.Resample(spacing)
            croporpad = tio.CropOrPad(original_shape)
            final_preds = croporpad(transform(tio.LabelMap(tensor=final_preds, affine=resampled_affine))).numpy()[0]

            self.save_mask(final_preds, fname, spacing.astype(np.float64), save_dir)

        return logs

    def on_test_start(self) -> None:  # noqa: D102
        super().on_test_start()

        if self.trainer.world_size > 1:
            print(f"\nWorld size is {self.trainer.world_size}, default to skip worse frame threshold and vae_on_test")
            self.hparams.worst_frame_thresholds = None
            self.hparams.vae_on_test = False

        if self.trainer.datamodule is None:
            sw_batch_size = 2
        else:
            sw_batch_size = self.trainer.datamodule.hparams.batch_size

        self.inferer = SlidingWindowInferer(
            roi_size=self.actor.actor.net.patch_size,
            sw_batch_size=sw_batch_size,
            overlap=0.5,
            mode='gaussian',
            cache_roi_weight_map=True,
        )

        self.reward_func.prepare_for_full_sequence(self.trainer.datamodule.hparams.batch_size)
        self.initial_test_params = copy.deepcopy(self.actor.actor.net.state_dict())


    def predict(
        self, image: Union[Tensor, MetaTensor], apply_softmax: bool = True
    ) -> Union[Tensor, MetaTensor]:
        """Predict 2D/3D images with sliding window inference.

        Args:
            image: Image to predict.
            apply_softmax: Whether to apply softmax to prediction.

        Returns:
            Aggregated prediction over all sliding windows.

        Raises:
            NotImplementedError: If the patch shape is not 2D nor 3D.
            ValueError: If 3D patch is requested to predict 2D images.
        """
        if len(image.shape) == 5:
            if len(self.actor.actor.net.patch_size) == 3:
                # Pad the last dimension to avoid 3D segmentation border artifacts
                pad_len = 6 if image.shape[-1] > 6 else image.shape[-1] - 1
                image = pad(image, (pad_len, pad_len, 0, 0, 0, 0), mode="reflect")
                pred = self.predict_3D_3Dconv_tiled(image, apply_softmax)
                # Inverse the padding after prediction
                return pred[..., pad_len:-pad_len]
            else:
                raise ValueError("Check your patch size. You dummy.")
        if len(image.shape) == 4:
            raise ValueError("No 2D images here. You dummy.")

    def tta_predict(
        self, image: Union[Tensor, MetaTensor], apply_softmax: bool = True
    ) -> Union[Tensor, MetaTensor]:
        """Predict with test time augmentation.

        Args:
            image: Image to predict.
            apply_softmax: Whether to apply softmax to prediction.

        Returns:
            Aggregated prediction over number of flips.
        """
        preds = self.predict(image, apply_softmax)
        factors = [1.1, 0.9, 1.25, 0.75]
        translations = [40, 60, 80, 120]
        rotations = [5, 10, -5, -10]

        for factor in factors:
            preds += self.predict(adjust_contrast(
                image.permute((4, 0, 1, 2, 3)), factor).permute((1, 2, 3, 4, 0)), apply_softmax)

        def x_translate_left(img, amount=20):
            return pad(img, (0, 0, 0, 0, amount, 0), mode="constant")[:, :, :-amount, :, :]
        def x_translate_right(img, amount=20):
            return pad(img, (0, 0, 0, 0, 0, amount), mode="constant")[:, :, amount:, :, :]
        def y_translate_up(img, amount=20):
            return pad(img, (0, 0, amount, 0, 0, 0), mode="constant")[:, :, :, :-amount, :]
        def y_translate_down(img, amount=20):
            return pad(img, (0, 0, 0, amount, 0, 0), mode="constant")[:, :, :, amount:, :]

        for translation in translations:
            preds += x_translate_right(self.predict(x_translate_left(image, translation), apply_softmax),
                                       translation)
            preds += x_translate_left(self.predict(x_translate_right(image, translation), apply_softmax),
                                      translation)
            preds += y_translate_down(self.predict(y_translate_up(image, translation), apply_softmax),
                                      translation)
            preds += y_translate_up(self.predict(y_translate_down(image, translation), apply_softmax),
                                    translation)

        # TODO: optimize this for compute time
        for rotation in rotations:
            rotated = torch.zeros_like(image)
            for i in range(image.shape[-1]):
                rotated[0, :, :, :, i] = rotate(image[0, :, :, :, i], angle=rotation)
            rot_pred = self.predict(rotated, apply_softmax)
            for i in range(image.shape[-1]):
                rot_pred[0, :, :, :, i] = rotate(rot_pred[0, :, :, :, i], angle=-rotation)
            preds += rot_pred

        preds /= len(factors) + len(translations) * 4 + len(rotations) + 1
        return preds.argmax(dim=1)

    def predict_3D_3Dconv_tiled(
        self, image: Union[Tensor, MetaTensor], apply_softmax: bool = True
    ) -> Union[Tensor, MetaTensor]:
        """Predict 3D image with 3D model.

        Args:
            image: Image to predict.
            apply_softmax: Whether to apply softmax to prediction.

        Returns:
            Aggregated prediction over all sliding windows.

        Raises:
            ValueError: If image is not 3D.
        """
        if not len(image.shape) == 5:
            raise ValueError("image must be (b, c, w, h, d)")

        if apply_softmax:
            return softmax_helper(self.sliding_window_inference(image))
        else:
            return self.sliding_window_inference(image)

    def sliding_window_inference(
        self, image: Union[Tensor, MetaTensor]
    ) -> Union[Tensor, MetaTensor]:
        """Inference using sliding window.

        Args:
            image: Image to predict.

        Returns:
            Predicted logits.
        """
        return self.inferer(
            inputs=image,
            network=self.actor.actor.net,
        )

    def save_mask(
        self, preds: np.ndarray, fname: str, spacing: np.ndarray, save_dir: Union[str, Path]
    ) -> None:
        """Save segmentation mask to the given save directory.

        Args:
            preds: Predicted segmentation mask.
            fname: Filename to save.
            spacing: Spacing to save the segmentation mask.
            save_dir: Directory to save the segmentation mask.
        """
        print(f"Saving segmentation for {fname}... in {save_dir}")

        os.makedirs(save_dir, exist_ok=True)

        preds = preds.astype(np.uint8)
        itk_image = sitk.GetImageFromArray(rearrange(preds, "w h d ->  d h w"))
        itk_image.SetSpacing(spacing)
        sitk.WriteImage(itk_image, os.path.join(save_dir, str(fname) + ".nii.gz"))

    def on_test_end(self) -> None:
        actor_save_path = self.hparams.actor_save_path if self.hparams.actor_save_path else \
            f"{self.trainer.log_dir}/{self.trainer.logger.version}/actor.ckpt"
        actor_save_path = Path(actor_save_path)
        actor_save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.actor.actor.net.state_dict(), actor_save_path)
        print(f"actor saved at: {actor_save_path}")

        critic_save_path = self.hparams.critic_save_path if self.hparams.critic_save_path else \
            f"{self.trainer.log_dir}/{self.trainer.logger.version}/critic.ckpt"
        critic_save_path = Path(critic_save_path)
        critic_save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.actor.critic.net.state_dict(), critic_save_path)
        print(f"critic saved at: {critic_save_path}")

    def predict_step(self, batch: dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0) -> Any:
        # must be batch size 1 as images have varied sizes
        b_img, meta_dict = batch['img'].squeeze(0), batch['image_meta_dict']
        id = meta_dict['case_identifier'][0]
        voxel_spacing = np.asarray([abs(meta_dict['resampled_affine'][0, 0, 0].cpu().numpy()),
                                     abs(meta_dict['resampled_affine'][0, 1, 1].cpu().numpy())])

        # could use full sequence later and split into subsecquions here
        actions, _, _ = self.rollout(b_img, torch.zeros_like(b_img).squeeze(1), sample=True)
        actions_unsampled, _, _ = self.rollout(b_img, torch.zeros_like(b_img).squeeze(1), sample=False)

        corrected, corrected_validity, ae_comp, actions_unsampled_clean = self.pred_corrector.correct_single_seq(b_img.squeeze(0), actions_unsampled.squeeze(0), voxel_spacing)
        actions_unsampled_clean = actions_unsampled_clean[None,]
        corrected = corrected[None,]

        initial_params = copy.deepcopy(self.actor.actor.net.state_dict())
        itr = 0

        self.trainer.datamodule.add_to_train(id)

        action_uns_anatomical_validity = [
            check_segmentation_validity(actions_unsampled_clean[0, ..., i].T, voxel_spacing, [0, 1, 2])
            for i in range(actions_unsampled_clean.shape[-1])]
        #
        # action_uns_temporal_validity = check_temporal_validity(actions_unsampled.squeeze(0).cpu().numpy(), voxel_spacing)\
        #     if action_uns_anatomical_validity else False
        # print(action_uns_temporal_validity)
        #
        # f, ax = plt.subplots(1, 4)
        # ax[0].imshow(actions_unsampled_clean[0, ..., 0].T, cmap='grey')
        # ax[1].imshow(actions_unsampled_clean[0, ..., 1].T, cmap='grey')
        # ax[2].imshow(actions_unsampled_clean[0, ..., 2].T, cmap='grey')
        # ax[3].imshow(actions_unsampled_clean[0, ..., 3].T, cmap='grey')
        # plt.title(action_uns_anatomical_validity)

        #
        # corrected_temporal_validity = check_temporal_validity(corrected.squeeze(0), voxel_spacing) \
        #     if corrected_validity else False
        # print(corrected_temporal_validity)
        # f, ax = plt.subplots(1, 4)
        # ax[0].imshow(corrected[0, ..., 0].T, cmap='grey')
        # ax[1].imshow(corrected[0, ..., 1].T, cmap='grey')
        # ax[2].imshow(corrected[0, ..., 2].T, cmap='grey')
        # ax[3].imshow(corrected[0, ..., 3].T, cmap='grey')
        # plt.title(corrected_validity)
        # print(ae_comp)
        # plt.show()

        if ae_comp > 0.95 and all(action_uns_anatomical_validity):
            self.trainer.datamodule.add_to_gt(id)

            # path = self.trainer.datamodule.get_approx_gt_subpath(id)
            # approx_gt_path = self.trainer.datamodule.hparams.approx_gt_dir + '/approx_gt/' + path
            # Path(approx_gt_path).parent.mkdir(parents=True, exist_ok=True)
            # hdr = nib.Nifti1Header()
            # nifti = nib.Nifti1Image(convert_to_numpy(np.round(actions_unsampled_clean).squeeze(0)),
            #                         np.diag(np.asarray([-1, -1, 1, 0])), hdr)
            # nifti.to_filename(approx_gt_path)

            if self.hparams.predict_do_model_perturb:
                for j, multiplier in enumerate([0.1, 0.15, 0.2, 0.25]):  # have been adapted from 2d
                    # get random seed based on time to maximise randomness of noise and subsequent predictions
                    # explore as much space around policy as possible
                    time_seed = int(round(datetime.now().timestamp())) + j
                    torch.manual_seed(time_seed)

                    # load initial params so noise is not compounded
                    self.actor.actor.net.load_state_dict(initial_params, strict=True)

                    # add noise to params
                    with torch.no_grad():
                        for param in self.actor.actor.net.parameters():
                            param.add_(torch.randn(param.size()).to(next(self.actor.actor.net.parameters()).device) * multiplier)

                    # make prediction
                    deformed_action, *_ = self.actor.actor(b_img)
                    if len(deformed_action.shape) > 4:
                        deformed_action = deformed_action.argmax(dim=1)
                    else:
                        deformed_action = torch.round(deformed_action)

                    # f, (ax1, ax2) = plt.subplots(1, 2)
                    # ax1.set_title(f"Good initial action")
                    # ax1.imshow(actions_unsampled[..., 0].cpu().numpy().T)
                    #
                    # ax2.set_title(f"Deformed network's action")
                    # ax2.imshow(deformed_action[..., 0].cpu().numpy().T)
                    # plt.show()

                    if deformed_action.sum() == 0:
                        continue

                    filename = f"{batch_idx}_{itr}_{time_seed}_{self.trainer.global_rank}_weights.nii.gz"
                    save_to_reward_dataset(self.hparams.predict_save_dir,
                                           filename,
                                           convert_to_numpy(b_img.squeeze(0)),
                                           convert_to_numpy(actions_unsampled_clean),
                                           convert_to_numpy(deformed_action))
                self.actor.actor.net.load_state_dict(initial_params)
            if self.hparams.predict_do_img_perturb:
                contrast_factors = [0.4, 0.05]  # check this !!!
                for factor in contrast_factors:
                    in_img = copy.deepcopy(b_img)
                    in_img = adjust_contrast(in_img.permute((4, 0, 1, 2, 3)), factor).permute((1, 2, 3, 4, 0))
                    # in_img = ((in_img - in_img.mean()) * factor + in_img.mean())
                    # in_img /= in_img.max()

                    # make prediction
                    contr_action, *_ = self.actor.actor(in_img)
                    if len(contr_action.shape) > 4:
                        contr_action = contr_action.argmax(dim=1)
                    else:
                        contr_action = torch.round(contr_action)

                    # f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
                    # ax1.set_title(f"Img")
                    # ax1.imshow(b_img[0, ..., 0].cpu().numpy().T)
                    #
                    # ax2.set_title(f"Contrast Img")
                    # ax2.imshow(in_img[0, ..., 0].cpu().numpy().T, vmin=0, vmax=1)
                    #
                    # ax3.set_title(f"Good action")
                    # ax3.imshow(actions_unsampled[..., 0].cpu().numpy().T)
                    #
                    # ax4.set_title(f"contrast action")
                    # ax4.imshow(contr_action[..., 0].cpu().numpy().T)
                    # plt.show()

                    # f, (ax1) = plt.subplots(1)
                    # # ax1.set_title(f"Bad initial action")
                    # ax1.imshow(actions_unsampled[i, ...].cpu().numpy().T, cmap='gray')
                    # ax1.axis('off')
                    # plt.savefig("/data/good_initial.png", bbox_inches='tight', pad_inches=0)
                    #
                    # f2, (ax2) = plt.subplots(1)
                    # ax2.imshow(contr_action[0, ...].cpu().numpy().T, cmap='gray')
                    # ax2.axis('off')
                    # plt.savefig('/data/deformed.png', bbox_inches='tight', pad_inches=0)
                    #
                    # f3, (ax3) = plt.subplots(1)
                    # ax3.imshow(b_img[i, ...].cpu().numpy().T, cmap='gray')
                    # ax3.axis('off')
                    # plt.savefig('/data/def_image.png', bbox_inches='tight', pad_inches=0)
                    #
                    # f4, (ax4) = plt.subplots(1)
                    # ax4.axis('off')
                    # ax4.imshow((actions_unsampled[i] == contr_action[0]).cpu().numpy().T, cmap='gray')
                    # plt.savefig('/data/def_diff.png', bbox_inches='tight', pad_inches=0)
                    #
                    # plt.show()


                    if contr_action.sum() == 0:
                        continue

                    time_seed = int(round(datetime.now().timestamp())) + int(factor*10)
                    filename = f"{batch_idx}_{itr}_{time_seed}_{self.trainer.global_rank}_contrast.nii.gz"
                    save_to_reward_dataset(self.hparams.predict_save_dir,
                                           filename,
                                           convert_to_numpy(b_img.squeeze(0)),
                                           convert_to_numpy(actions_unsampled_clean),
                                           convert_to_numpy(contr_action))

                gaussian_blurs = [0.3, 0.6]
                for blur in gaussian_blurs:
                    in_img = b_img.clone()
                    in_img += torch.randn(in_img.size()).to(next(self.actor.actor.net.parameters()).device) * blur
                    in_img /= in_img.max()

                    # make prediction
                    blurred_action, *_ = self.actor.actor(in_img)
                    if len(blurred_action.shape) > 4:
                        blurred_action = blurred_action.argmax(dim=1)
                    else:
                        blurred_action = torch.round(blurred_action)

                    # f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
                    # ax1.set_title(f"Img")
                    # ax1.imshow(b_img[0, ..., 0].cpu().numpy().T)
                    #
                    # ax2.set_title(f"Blurred Img")
                    # ax2.imshow(in_img[0, ..., 0].cpu().numpy().T, vmin=0, vmax=1)
                    #
                    # ax3.set_title(f"Good action")
                    # ax3.imshow(actions_unsampled[..., 0].cpu().numpy().T)
                    #
                    # ax4.set_title(f"Blurred action")
                    # ax4.imshow(blurred_action[..., 0].cpu().numpy().T)
                    # plt.show()

                    if blurred_action.sum() == 0:
                        continue

                    time_seed = int(round(datetime.now().timestamp())) + int(blur*100)
                    filename = f"{batch_idx}_{itr}_{time_seed}_{self.trainer.global_rank}_blur.nii.gz"
                    save_to_reward_dataset(self.hparams.predict_save_dir,
                                           filename,
                                           convert_to_numpy(b_img.squeeze(0)),
                                           convert_to_numpy(actions_unsampled_clean),
                                           convert_to_numpy(blurred_action))

        else:
            if corrected_validity:
                # f, (ax1, ax2) = plt.subplots(1, 2)
                # ax1.set_title(f"Bad action")
                # ax1.imshow(actions[..., 0].cpu().numpy().T)
                #
                # ax2.set_title(f"Corrected action")
                # ax2.imshow(corrected[..., 0].T)
                # plt.show()

                if self.hparams.predict_do_corrections:
                    filename = f"{batch_idx}_{itr}_{int(round(datetime.now().timestamp()))}_{self.trainer.global_rank}_correction.nii.gz"
                    save_to_reward_dataset(self.hparams.predict_save_dir,
                                           filename,
                                           convert_to_numpy(b_img.squeeze(0)),
                                           convert_to_numpy(corrected),
                                           convert_to_numpy(actions))

        self.trainer.datamodule.update_dataframe()
        # make sure initial params are back at end of step
        self.actor.actor.net.load_state_dict(initial_params)
        self.predicted_rows += [self.trainer.datamodule.df.loc[self.trainer.datamodule.df['dicom_uuid'] == id]]

    def on_predict_epoch_end(self) -> None:
        # for multi gpu cases, save intermediate file before sending to main csv
        print(f"Saving temporary rows file to : {f'{self.temp_files_path}/temp_pred_{self.trainer.global_rank}.csv'}")
        pd.concat(self.predicted_rows).to_csv(f"{self.temp_files_path}/temp_pred_{self.trainer.global_rank}.csv")
