import copy
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Union, Optional

import matplotlib.pyplot as plt
import h5py
import SimpleITK as sitk
import nibabel as nib
import numpy as np
from einops import rearrange
from lightning import LightningModule
import torch
import torchio as tio
from monai.data import MetaTensor
from scipy import ndimage
from torchvision.transforms.functional import adjust_contrast
from torch import Tensor
from torch.nn.functional import pad

from patchless_nnunet.utils.inferers import SlidingWindowInferer
from patchless_nnunet.utils.softmax import softmax_helper
from vital.metrics.camus.anatomical.utils import check_segmentation_validity
from vital.data.camus.config import Label

from rl4echo.utils.Metrics import accuracy, dice_score, is_anatomically_valid
from rl4echo.utils.file_utils import get_img_subpath, save_to_reward_dataset
from rl4echo.utils.logging_helper import log_sequence, log_video
from rl4echo.utils.tensor_utils import convert_to_numpy
from rl4echo.utils.test_metrics import dice, hausdorff


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
                 save_on_test=False,
                 save_csv_after_predict=None,
                 *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        #TODO: use automatic hparams

        self.actor = actor
        self.reward_func = reward
        if hasattr(self.reward_func, 'net'):
            self.register_module('rewardnet', self.reward_func.net)

        self.actor_save_path = actor_save_path
        self.critic_save_path = critic_save_path

        self.save_uncertainty_path = save_uncertainty_path

        self.predict_save_dir = predict_save_dir
        self.pred_corrector = corrector

        self.predict_do_model_perturb = predict_do_model_perturb
        self.predict_do_img_perturb = predict_do_img_perturb
        self.predict_do_corrections = predict_do_corrections

        self.save_on_test = save_on_test
        self.save_csv_after_predict = save_csv_after_predict
        self.predicted_rows = []

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
        b_img, b_gt, b_use_gt = batch['img'].squeeze(0), batch['gt'].squeeze(0), batch['use_gt']

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
        if self.trainer.local_rank == 0:
            idx = random.randint(0, len(b_img) - 1)  # which image to log
            log_sequence(self.logger, img=b_img[idx], title='Image', number=batch_idx, epoch=self.current_epoch)
            log_sequence(self.logger, img=b_gt[idx].unsqueeze(0), title='GroundTruth', number=batch_idx,
                         epoch=self.current_epoch)
            log_sequence(self.logger, img=prev_actions[idx].unsqueeze(0), title='Prediction', number=batch_idx,
                      img_text=prev_rewards[idx].mean(), epoch=self.current_epoch)
            if prev_rewards.shape == prev_actions.shape:
                log_sequence(self.logger, img=prev_rewards[idx].unsqueeze(0), title='RewardMap', number=batch_idx,
                             epoch=self.current_epoch)

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
        b_img, b_gt, meta_dict = batch['img'], batch['gt'], batch['image_meta_dict']

        self.patch_size = list([b_img.shape[-3], b_img.shape[-2], 4])
        self.inferer.roi_size = self.patch_size

        start_time = time.time()
        prev_actions = self.predict(b_img).argmax(dim=1)
        print(f"\nPrediction took {round(time.time() - start_time, 4)} (s).")

        # this would not work with a neural net (no sliding window)
        # should there be a test method in the reward class (which, in the case of 3d rewards
        # would have a different method, otherwise, only redirect to __call__())
        prev_rewards = self.reward_func(prev_actions[..., :4], b_img[..., :4], b_gt[..., :4])

        # without sliding window
        # b_img = b_img[..., :4]
        # b_gt = b_gt[..., :4]
        # prev_actions, prev_log_probs, prev_rewards = self.rollout(b_img, b_gt, sample=False)
        # loss, critic_loss, metrics_dict = self.compute_policy_loss((b_img, prev_actions, prev_rewards,
        #                                                             prev_log_probs, b_gt, False))

        acc = accuracy(prev_actions, b_img, b_gt)
        simple_dice = dice_score(prev_actions, b_gt)

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

        start_time = time.time()
        test_dice = dice(y_pred_np_as_batch, b_gt_np_as_batch, labels=(Label.BG, Label.LV, Label.MYO),
                         exclude_bg=True, all_classes=True)
        test_dice_epi = dice((y_pred_np_as_batch != 0).astype(np.uint8), (b_gt_np_as_batch != 0).astype(np.uint8),
                             labels=(Label.BG, Label.LV), exclude_bg=True, all_classes=False)
        print(f"Dice took {round(time.time() - start_time, 4)} (s).")

        start_time = time.time()
        test_hd = hausdorff(y_pred_np_as_batch, b_gt_np_as_batch, labels=(Label.BG, Label.LV, Label.MYO),
                            exclude_bg=True, all_classes=True, voxel_spacing=voxel_spacing)
        test_hd_epi = hausdorff((y_pred_np_as_batch != 0).astype(np.uint8), (b_gt_np_as_batch != 0).astype(np.uint8),
                                labels=(Label.BG, Label.LV), exclude_bg=True, all_classes=False,
                                voxel_spacing=voxel_spacing)['Hausdorff']
        print(f"HD took {round(time.time() - start_time, 4)} (s).")

        start_time = time.time()
        anat_errors = is_anatomically_valid(y_pred_np_as_batch)
        print(f"AV took {round(time.time() - start_time, 4)} (s).")

        logs = {#'test_loss': loss,
                "test_reward": torch.mean(prev_rewards.type(torch.float)),
                'test_acc': acc.mean(),
                "test_dice": simple_dice.mean(),
                "test_anat_valid": anat_errors.mean(),
                'dice_epi': test_dice_epi,
                'hd_epi': test_hd_epi,
                }
        logs.update(test_dice)
        logs.update(test_hd)

        start_time = time.time()
        # for logging v
        # Use only first 4 for visualization, avoids having to implement sliding window inference for critic
        _, _, _, _, v, _ = self.actor.evaluate(b_img[..., :4], prev_actions[..., :4])

        if self.trainer.local_rank == 0:
            for i in range(len(b_img)):
                log_video(self.logger, img=b_img[i], title='test_Image', number=batch_idx * (i + 1),
                             epoch=self.current_epoch)
                log_video(self.logger, img=b_gt[i].unsqueeze(0), background=b_img[i], title='test_GroundTruth', number=batch_idx * (i + 1),
                             epoch=self.current_epoch)
                log_video(self.logger, img=prev_actions[i].unsqueeze(0), background=b_img[i], title='test_Prediction',
                             number=batch_idx * (i + 1), img_text=simple_dice[i].mean(), epoch=self.current_epoch)
                if v.shape == prev_actions[..., :4].shape:
                    log_sequence(self.logger, img=v[i].unsqueeze(0), title='test_v_function', number=batch_idx * (i + 1),
                              img_text=v[i].mean(), epoch=self.current_epoch)
                if prev_rewards.shape[:-1] == prev_actions.shape[:-1]:
                    log_sequence(self.logger, img=prev_rewards[i].unsqueeze(0), title='test_RewardMap', number=batch_idx * (i + 1),
                              img_text=prev_rewards[i].mean(), epoch=self.current_epoch)

        self.log_dict(logs)
        print(f"Logging took {round(time.time() - start_time, 4)} (s).")
        # if self.save_uncertainty_path:
        #     with h5py.File(self.save_uncertainty_path, 'a') as f:
        #         for i in range(len(b_img)):
        #             dicom = batch['id'][i] + "_" + batch['instant'][i]
        #             if dicom not in f:
        #                 f.create_group(dicom)
        #             f[dicom]['img'] = b_img[i].cpu().numpy()
        #             f[dicom]['gt'] = b_gt[i].cpu().numpy()
        #             f[dicom]['pred'] = prev_actions[i].cpu().numpy()
        #             f[dicom]['reward_map'] = prev_rewards[i].cpu().numpy()
        #             f[dicom]['accuracy_map'] = (prev_actions[i].cpu().numpy() != b_gt[i].cpu().numpy()).astype(np.uint8)

        if self.save_on_test:
            prev_actions = prev_actions.squeeze(0).cpu().detach().numpy()
            original_shape = meta_dict.get("original_shape").cpu().detach().numpy()[0]

            save_dir = os.path.join(self.trainer.default_root_dir, "testing_raw")

            fname = meta_dict.get("case_identifier")[0]
            spacing = meta_dict.get("original_spacing").cpu().detach().numpy()[0]
            resampled_affine = meta_dict.get("resampled_affine").cpu().detach().numpy()[0]

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
        if self.actor_save_path:
            torch.save(self.actor.actor.net.state_dict(), self.actor_save_path)

            actor_net = shrink_perturb(copy.deepcopy(self.actor.actor.net))
            torch.save(actor_net.state_dict(), self.actor_save_path.replace('.ckpt', '_s-p.ckpt'))
        if self.critic_save_path:
            torch.save(self.actor.critic.net.state_dict(), self.critic_save_path)
            critic_net = shrink_perturb(copy.deepcopy(self.actor.critic.net))
            torch.save(critic_net.state_dict(), self.critic_save_path.replace('.ckpt', '_s-p.ckpt'))

    def predict_step(self, batch: dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0) -> Any:
        # must be batch size 1 as images have varied sizes
        b_img, meta_dict = batch['img'].squeeze(0), batch['image_meta_dict']
        id = meta_dict['case_identifier'][0]

        # could use full sequence later and split into subsecquions here
        actions, _, _ = self.rollout(b_img, torch.zeros_like(b_img).squeeze(1), sample=True)
        actions_unsampled, _, _ = self.rollout(b_img, torch.zeros_like(b_img).squeeze(1), sample=False)

        b_img_as_batch = b_img.squeeze(0).permute((3, 0, 1, 2))
        actions_unsampled_as_batch = actions_unsampled.squeeze(0).permute((2, 0, 1))
        corrected, corrected_validity, ae_comp = self.pred_corrector.correct_batch(b_img_as_batch, actions_unsampled_as_batch)
        corrected = corrected.transpose((1, 2, 3, 0))

        initial_params = copy.deepcopy(self.actor.actor.net.state_dict())
        itr = 0

        # valid for all frames
        corrected_validity = corrected_validity.min()
        ae_comp = ae_comp.mean()

        self.trainer.datamodule.add_to_train(id)
        if ae_comp > 0.95 and all([check_segmentation_validity(actions_unsampled_as_batch[i].cpu().numpy().T, (1.0, 1.0),
                                                             [0, 1, 2]) for i in range(len(actions_unsampled_as_batch))]):
            self.trainer.datamodule.add_to_gt(id)

            path = self.trainer.datamodule.get_approx_gt_subpath(id)
            approx_gt_path = self.trainer.datamodule.hparams.approx_gt_dir + '/approx_gt/' + path
            Path(approx_gt_path).parent.mkdir(parents=True, exist_ok=True)
            hdr = nib.Nifti1Header()
            nifti = nib.Nifti1Image(convert_to_numpy(torch.round(actions_unsampled).squeeze(0)),
                                    np.diag(np.asarray([-1, -1, 1, 0])), hdr)
            nifti.to_filename(approx_gt_path)

            if self.predict_do_model_perturb:
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

                    filename = f"{batch_idx}_{itr}_{time_seed}_{self.trainer.local_rank}_weights.nii.gz"
                    save_to_reward_dataset(self.predict_save_dir,
                                           filename,
                                           convert_to_numpy(b_img.squeeze(0)),
                                           convert_to_numpy(actions_unsampled),
                                           convert_to_numpy(deformed_action))
                self.actor.actor.net.load_state_dict(initial_params)
            if self.predict_do_img_perturb:
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
                    filename = f"{batch_idx}_{itr}_{time_seed}_{self.trainer.local_rank}_contrast.nii.gz"
                    save_to_reward_dataset(self.predict_save_dir,
                                           filename,
                                           convert_to_numpy(b_img.squeeze(0)),
                                           convert_to_numpy(actions_unsampled),
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
                    filename = f"{batch_idx}_{itr}_{time_seed}_{self.trainer.local_rank}_blur.nii.gz"
                    save_to_reward_dataset(self.predict_save_dir,
                                           filename,
                                           convert_to_numpy(b_img.squeeze(0)),
                                           convert_to_numpy(actions_unsampled),
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

                if self.predict_do_corrections:
                    filename = f"{batch_idx}_{itr}_{int(round(datetime.now().timestamp()))}_{self.trainer.local_rank}_correction.nii.gz"
                    save_to_reward_dataset(self.predict_save_dir,
                                           filename,
                                           convert_to_numpy(b_img.squeeze(0)),
                                           convert_to_numpy(corrected),
                                           convert_to_numpy(actions))

        self.trainer.datamodule.update_dataframe()
        # make sure initial params are back at end of step
        self.actor.actor.net.load_state_dict(initial_params)
        # self.predicted_rows += [self.trainer.datamodule.df.loc[self.trainer.datamodule.df['dicom_uuid'] == id]]
        self.predicted_rows += [torch.tensor(self.trainer.local_rank)[None,]]

    def on_predict_epoch_end(self) -> None:
        rows = torch.cat(self.predicted_rows, dim=0)
        self.predicted_rows = self.all_gather(rows)
        if self.trainer.local_rank == 0:
            print(f"After gather len: {len(self.predicted_rows)}")

