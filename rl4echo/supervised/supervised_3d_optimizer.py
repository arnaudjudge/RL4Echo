import copy
import random
import time
from datetime import datetime
from typing import Dict, Any

import lightning
import numpy as np
from scipy import ndimage
# import pytorch_lightning as pl
import torch
from torch import nn, Tensor
from torchmetrics.classification import Dice

from patchless_nnunet.utils.softmax import softmax_helper
from patchless_nnunet.utils.tensor_utils import sum_tensor
from vital.metrics.camus.anatomical.utils import check_segmentation_validity
# from vital.models.segmentation.unet import UNet
from vital.data.camus.config import Label

from rl4echo.utils.Metrics import accuracy, dice_score, is_anatomically_valid
from rl4echo.utils.file_utils import save_to_reward_dataset
from rl4echo.utils.logging_helper import log_sequence, log_video
from rl4echo.utils.tensor_utils import convert_to_numpy
from rl4echo.utils.test_metrics import dice, hausdorff

from patchless_nnunet.models.patchless_nnunet_module import nnUNetPatchlessLitModule

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, output, target):
        intersection = torch.sum(target * output)
        return 1 - ((2. * intersection) / (torch.sum(target) + torch.sum(output)))


class Supervised3DOptimizer(nnUNetPatchlessLitModule):
    def __init__(self, ckpt_path=None, corrector=None, predict_save_dir=None, **kwargs):
        super().__init__(**kwargs)

        self.save_test_results = False
        self.ckpt_path = ckpt_path
        self.predict_save_dir = predict_save_dir
        self.pred_corrector = corrector

        self.dice = Dice()

    def forward(self, x):
        out = self.net.forward(x)
        if self.net.num_classes > 1:
            out = torch.softmax(out, dim=1)
        else:
            out = torch.sigmoid(out).squeeze(1)
        return out

    def configure_optimizers(self):
        # add weight decay so predictions are less certain, more randomness?
        return torch.optim.Adam(self.parameters(), lr=0.001, weight_decay=0)

    def training_step(self, batch: dict[str, Tensor], *args, **kwargs) -> Dict:
        x, y = batch['img'].squeeze(0), batch['gt'].squeeze(0)

        y_hat = self.forward(x)

        loss = self.loss(y_hat, y)

        logs = {
            'loss': loss,
        }

        self.log_dict(logs, prog_bar=True)
        return logs

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int):
        b_img, b_gt = batch['img'].squeeze(0), batch['gt'].squeeze(0)
        y_pred = self.forward(b_img)

        loss = self.loss(y_pred, b_gt)

        if self.net.num_classes > 1:
            y_pred = y_pred.argmax(dim=1)
        else:
            y_pred = torch.round(y_pred)

        acc = accuracy(y_pred, b_img, b_gt)
        dice = dice_score(y_pred, b_gt)
        #anat_err = has_anatomical_error(y_pred)

        logs = {'val_loss': loss,
                'val_acc': acc.mean(),
                'val_dice': dice.mean(),
                #'val_anat_errors': anat_err.mean(),
                }

        # log images
        idx = random.randint(0, len(b_img) - 1)  # which image to log
        log_sequence(self.logger, img=b_img[idx], title='Image', number=batch_idx, epoch=self.current_epoch)
        log_sequence(self.logger, img=b_gt[idx].unsqueeze(0), title='GroundTruth', number=batch_idx,
                     epoch=self.current_epoch)
        log_sequence(self.logger, img=y_pred[idx].unsqueeze(0), title='Prediction', number=batch_idx,
                     img_text=acc[idx].mean(), epoch=self.current_epoch)

        self.log_dict(logs)

        return logs

    # don't want these functions from parent class
    def on_validation_epoch_end(self) -> None:
        return

    def on_test_epoch_end(self) -> None:
        return

    def test_step(self, batch, batch_idx):
        b_img, b_gt, meta_dict = batch['img'], batch['gt'], batch['image_meta_dict']

        self.patch_size = list([b_img.shape[-3], b_img.shape[-2], self.hparams.sliding_window_len])
        self.inferer.roi_size = self.patch_size

        start_time = time.time()
        y_pred = self.predict(b_img)
        print(f"\nPrediction took {round(time.time() - start_time, 4)} (s).")


        loss = self.loss(y_pred, b_gt.long())

        if self.num_classes > 1:
            y_pred = y_pred.argmax(dim=1)
        else:
            y_pred = torch.round(y_pred)

        acc = accuracy(y_pred, b_img, b_gt)
        simple_dice = dice_score(y_pred, b_gt)

        y_pred_np_as_batch = y_pred.cpu().numpy().squeeze(0).transpose((2, 0, 1))
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
        voxel_spacing = np.asarray([[abs(meta_dict['resampled_affine'][0,0,0].cpu().numpy()),
                                    abs(meta_dict['resampled_affine'][0,1,1].cpu().numpy())]]).repeat(
            repeats=len(y_pred_np_as_batch), axis=0)

        test_dice = dice(y_pred_np_as_batch, b_gt_np_as_batch, labels=(Label.BG, Label.LV, Label.MYO),
                         exclude_bg=True, all_classes=True)
        test_dice_epi = dice((y_pred_np_as_batch != 0).astype(np.uint8), (b_gt_np_as_batch != 0).astype(np.uint8),
                             labels=(Label.BG, Label.LV), exclude_bg=True, all_classes=False)

        test_hd = hausdorff(y_pred_np_as_batch, b_gt_np_as_batch, labels=(Label.BG, Label.LV, Label.MYO),
                            exclude_bg=True, all_classes=True, voxel_spacing=voxel_spacing)
        test_hd_epi = hausdorff((y_pred_np_as_batch != 0).astype(np.uint8), (b_gt_np_as_batch != 0).astype(np.uint8),
                                labels=(Label.BG, Label.LV), exclude_bg=True, all_classes=False,
                                voxel_spacing=voxel_spacing)['Hausdorff']
        anat_errors = is_anatomically_valid(y_pred_np_as_batch)

        logs = {'test_loss': loss,
                'test_acc': acc.mean(),
                'test_dice': simple_dice.mean(),
                'test_anat_valid': anat_errors.mean(),
                'dice_epi': test_dice_epi,
                'hd_epi': test_hd_epi,
                }
        logs.update(test_dice)
        logs.update(test_hd)

        for i in range(len(b_img)):
            log_video(self.logger, img=b_img[i], title='test_Image', number=batch_idx * (i + 1),
                         epoch=self.current_epoch)
            log_video(self.logger, img=b_gt[i].unsqueeze(0), title='test_GroundTruth', number=batch_idx * (i + 1),
                         epoch=self.current_epoch)
            log_video(self.logger, img=y_pred[i].unsqueeze(0), title='test_Prediction', number=batch_idx * (i + 1),
                      img_text=simple_dice[i].mean(), epoch=self.current_epoch)

        self.log_dict(logs)

        return logs

    def on_test_end(self) -> None:
        self.save()

    def save(self) -> None:
        if self.ckpt_path:
            torch.save(self.net.state_dict(), self.ckpt_path)

    # not adapted for supervised 3d yet, may not be needed

    # def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
    #     b_img, b_gt, ids, inst = batch['img'], batch['gt'], batch['id'], batch.get('instant', None)
    #
    #     actions = self.forward(b_img)
    #     if self.output_shape[0] > 1:
    #         actions = actions.argmax(dim=1)
    #     else:
    #         actions = torch.round(actions)
    #
    #     corrected, corrected_validity, ae_comp = self.pred_corrector.correct_batch(b_img, actions)
    #
    #     initial_params = copy.deepcopy(self.net.state_dict())
    #     itr = 0
    #     df = self.trainer.datamodule.df
    #     for i in range(len(b_img)):
    #         # f, (ax1, ax2) = plt.subplots(1, 2)
    #         # ax1.set_title(f"action")
    #         # ax1.imshow(actions[i, ...].cpu().numpy().T)
    #         # ax2.set_title(f"Corrected action {ae_comp[i]}")
    #         # ax2.imshow(corrected[i, ...].T)
    #         # plt.show()
    #         self.trainer.datamodule.add_to_train(ids[i], inst[i] if inst else None)
    #         if ae_comp[i] > 0.95 and check_segmentation_validity(actions[i].cpu().numpy().T, (1.0, 1.0), [0, 1, 2]):
    #
    #             # force actions to resemble image?
    #             filename = f"{batch_idx}_{itr}_{i}_wrong_{int(round(datetime.now().timestamp()))}.nii.gz"
    #             save_to_reward_dataset(self.predict_save_dir,
    #                                    filename,
    #                                    convert_to_numpy(b_img[i]),
    #                                    np.expand_dims(convert_to_numpy(actions[i]), 0),
    #                                    np.expand_dims(convert_to_numpy(actions[random.randint(0, len(b_img)-1)]), 0))
    #
    #             for j, multiplier in enumerate([0.005, 0.01]):
    #                 # get random seed based on time to maximise randomness of noise and subsequent predictions
    #                 # explore as much space around policy as possible
    #                 time_seed = int(round(datetime.now().timestamp())) + i
    #                 torch.manual_seed(time_seed)
    #
    #                 # load initial params so noise is not compounded
    #                 self.net.load_state_dict(initial_params, strict=True)
    #
    #                 # add noise to params
    #                 with torch.no_grad():
    #                     for param in self.net.parameters():
    #                         param.add_(torch.randn(param.size()).cuda() * multiplier)
    #
    #                 # make prediction
    #                 deformed_action = self.forward(b_img[i].unsqueeze(0))
    #                 if self.output_shape[0] > 1:
    #                     deformed_action = deformed_action.argmax(dim=1)
    #                 else:
    #                     deformed_action = torch.round(deformed_action)
    #
    #                 # f, (ax1, ax2) = plt.subplots(1, 2)
    #                 # ax1.set_title(f"Good initial action")
    #                 # ax1.imshow(actions[i, ...].cpu().numpy().T)
    #                 #
    #                 # ax2.set_title(f"Deformed network's action")
    #                 # ax2.imshow(deformed_action[0, ...].cpu().numpy().T)
    #                 # plt.show()
    #
    #                 filename = f"{batch_idx}_{itr}_{i}_{time_seed}.nii.gz"
    #                 save_to_reward_dataset(self.predict_save_dir,
    #                                        filename,
    #                                        convert_to_numpy(b_img[i]),
    #                                        np.expand_dims(convert_to_numpy(b_gt[i]), 0),
    #                                        convert_to_numpy(deformed_action))
    #         else:
    #             if not corrected_validity[i]:
    #                 continue
    #
    #             # f, (ax1, ax2) = plt.subplots(1, 2)
    #             # ax1.set_title(f"Bad initial action")
    #             # ax1.imshow(actions[i, ...].cpu().numpy().T)
    #             #
    #             # ax2.set_title(f"Corrected action")
    #             # ax2.imshow(corrected[i, ...].T)
    #             # plt.show()
    #
    #             filename = f"{batch_idx}_{itr}_{i}_{int(round(datetime.now().timestamp()))}.nii.gz"
    #             save_to_reward_dataset(self.predict_save_dir,
    #                                    filename,
    #                                    convert_to_numpy(b_img[i]),
    #                                    convert_to_numpy(corrected[i]),
    #                                    np.expand_dims(convert_to_numpy(actions[i]), 0))
    #
    #     df.to_csv(self.trainer.datamodule.df_path)
    #     # make sure initial params are back at end of step
    #     self.net.load_state_dict(initial_params)
