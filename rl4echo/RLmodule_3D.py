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
from torchvision.transforms.functional import adjust_contrast

from patchless_nnunet.utils.inferers import SlidingWindowInferer
from patchless_nnunet.utils.softmax import softmax_helper
from rl4echo.utils.Metrics import accuracy, dice_score
from rl4echo.utils.correctors import AEMorphoCorrector
from rl4echo.utils.file_utils import save_to_reward_dataset
from rl4echo.utils.logging_helper import log_sequence, log_video
from rl4echo.utils.tensor_utils import convert_to_numpy
from rl4echo.utils.test_metrics import full_test_metrics
from vital.metrics.camus.anatomical.utils import check_segmentation_validity


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
                 save_on_test=False,
                 vae_on_test=False,
                 worst_frame_thresholds={"anatomical": 0.985},
                 save_csv_after_predict=None,
                 val_batch_size=4,
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

        self.patch_size = list([b_img.shape[-3], b_img.shape[-2], 4])
        self.inferer.roi_size = self.patch_size

        start_time = time.time()
        prev_actions = self.predict(b_img).argmax(dim=1)
        print(f"\nPrediction took {round(time.time() - start_time, 4)} (s).")

        prev_rewards = torch.stack(self.reward_func.predict_full_sequence(prev_actions, b_img, b_gt), dim=0)
        prev_rewards_mean = torch.mean(prev_rewards, dim=0)

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

        logs = full_test_metrics(y_pred_np_as_batch, b_gt_np_as_batch, voxel_spacing, self.device)
        logs.update({"test/reward": torch.mean(prev_rewards_mean.type(torch.float))})

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

        if self.trainer.global_rank == 0 and batch_idx % 1 == 0:
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

        # with h5py.File('3d_anatomical_reward_100randompred_wclean.h5', 'a') as f:
        #     for i in range(len(b_img)):
        #         dicom = meta_dict.get("case_identifier")[0]
        #         if dicom not in f:
        #             f.create_group(dicom)
        #         f[dicom]['img'] = b_img[i].cpu().numpy().squeeze(0)
        #         f[dicom]['gt'] = b_gt[i].cpu().numpy()
        #         f[dicom]['pred'] = y_pred_np_as_batch.transpose((1, 2, 0))
        #         clean_reward = self.reward_func.predict_full_sequence(torch.tensor(y_pred_np_as_batch.transpose((1, 2, 0))[None,], device=self.device), b_img, b_gt)
        #         f[dicom]['reward_map'] = clean_reward[i].cpu().numpy() #prev_rewards_mean[i].cpu().numpy()
        #         f[dicom]['accuracy_map'] = (prev_actions[i].cpu().numpy() != b_gt[i].cpu().numpy()).astype(np.uint8)

        if self.hparams.save_on_test:
            #prev_actions = prev_actions.squeeze(0).cpu().detach().numpy()
            prev_actions = y_pred_np_as_batch.transpose((1, 2, 0))
            original_shape = meta_dict.get("original_shape").cpu().detach().numpy()[0]

            fname = meta_dict.get("case_identifier")[0]
            spacing = meta_dict.get("original_spacing").cpu().detach().numpy()[0]
            resampled_affine = meta_dict.get("resampled_affine").cpu().detach().numpy()[0]
            save_dir = os.path.join(self.trainer.default_root_dir, f"testing_raw/{self.trainer.datamodule.get_approx_gt_subpath(fname).rsplit('/', 1)[0]}/")

            final_preds = np.expand_dims(prev_actions, 0)
            transform = tio.Resample(spacing)
            croporpad = tio.CropOrPad(original_shape)
            final_preds = croporpad(transform(tio.LabelMap(tensor=final_preds, affine=resampled_affine))).numpy()[0]

            self.save_mask(final_preds, fname, spacing.astype(np.float64), save_dir)

        return logs

    def on_test_start(self) -> None:  # noqa: D102
        super().on_test_start()
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
        if self.trainer.global_rank == 0:
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
            print(f"critic saved at: {actor_save_path}")

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

            path = self.trainer.datamodule.get_approx_gt_subpath(id)
            approx_gt_path = self.trainer.datamodule.hparams.approx_gt_dir + '/approx_gt/' + path
            Path(approx_gt_path).parent.mkdir(parents=True, exist_ok=True)
            hdr = nib.Nifti1Header()
            nifti = nib.Nifti1Image(convert_to_numpy(np.round(actions_unsampled_clean).squeeze(0)),
                                    np.diag(np.asarray([-1, -1, 1, 0])), hdr)
            nifti.to_filename(approx_gt_path)

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
        pd.concat(self.predicted_rows).to_csv(f"{self.temp_files_path}/temp_pred_{self.trainer.global_rank}.csv")
