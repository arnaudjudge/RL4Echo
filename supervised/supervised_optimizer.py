import copy
import json
from datetime import datetime
import random
from pathlib import Path
from typing import Dict, Any

import cv2
import nibabel as nib
import numpy as np
import pytorch_lightning as pl
import torch
from bdicardio.utils.morphological_and_ae import MorphologicalAndTemporalCorrectionAEApplicator
from bdicardio.utils.ransac_utils import ransac_sector_extraction
from bdicardio.utils.segmentation_validity import check_segmentation_for_all_frames, compare_segmentation_with_ae
from matplotlib import pyplot as plt
from pytorch_lightning.loggers import TensorBoardLogger
from scipy import ndimage
from torch import nn, Tensor
from vital.metrics.camus.anatomical.utils import check_segmentation_validity
from vital.models.segmentation.unet import UNet

from utils.Metrics import accuracy, dice_score
from vital.metrics.train.functional import differentiable_dice_score
from torchmetrics.classification import Dice

from utils.correctors import AEMorphoCorrector, RansacCorrector
from utils.file_utils import get_img_subpath, save_to_reward_dataset
from utils.logging_helper import log_image
from utils.tensor_utils import convert_to_numpy


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, output, target):
        intersection = torch.sum(target * output)
        return 1 - ((2. * intersection) / (torch.sum(target) + torch.sum(output)))


class SupervisedOptimizer(pl.LightningModule):
    def __init__(self, input_shape=(1, 256, 256), output_shape=(1, 256, 256), loss=nn.BCELoss(), ckpt_path=None, corrector=None, predict_save_dir=None, **kwargs):
        super().__init__(**kwargs)

        self.net = UNet(input_shape=input_shape, output_shape=output_shape)
        # self.net.load_state_dict(torch.load("./auto_iteration3/0/actor.ckpt"))
        self.input_shape = input_shape
        self.output_shape = output_shape

        self.loss = loss
        self.save_test_results = False
        self.ckpt_path = ckpt_path
        self.predict_save_dir = predict_save_dir
        self.pred_corrector = corrector

        self.dice = Dice()

    def forward(self, x):
        out = self.net.forward(x)
        if self.output_shape[0] > 1:
            out = torch.softmax(out, dim=1)
        else:
            out = torch.sigmoid(out).squeeze(1)
        return out

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def training_step(self, batch: dict[str, Tensor], *args, **kwargs) -> Dict:
        x, y = batch['img'], batch['mask']

        y_hat = self.forward(x)

        loss = self.loss(y_hat, y)

        logs = {
            'loss': loss,
        }

        self.log_dict(logs, prog_bar=True)
        return logs

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int):
        b_img, b_gt = batch['img'], batch['mask']
        y_pred = self.forward(b_img)

        loss = self.loss(y_pred, b_gt)

        if self.output_shape[0] > 1:
            y_pred = y_pred.argmax(dim=1)
        else:
            y_pred = torch.round(y_pred)

        acc = accuracy(y_pred, b_img, b_gt)
        dice = dice_score(y_pred, b_gt)
        logs = {'val_loss': loss,
                'val_acc': acc.mean(),
                'val_dice': dice.mean(),
                }

        # log images
        idx = random.randint(0, len(b_img) - 1)  # which image to log
        log_image(self.logger, img=b_img[idx].permute((0, 2, 1)), title='Image', number=batch_idx)
        log_image(self.logger, img=b_gt[idx].unsqueeze(0).permute((0, 2, 1)), title='GroundTruth', number=batch_idx)
        log_image(self.logger, img=y_pred[idx].unsqueeze(0).permute((0, 2, 1)), title='Prediction', number=batch_idx,
                  img_text=acc[idx].mean())

        self.log_dict(logs)
        return logs

    def test_step(self, batch, batch_idx):
        b_img, b_gt = batch['img'], batch['mask']
        y_pred = self.forward(b_img)

        loss = self.loss(y_pred, b_gt)

        if self.output_shape[0] > 1:
            y_pred = y_pred.argmax(dim=1)
        else:
            y_pred = torch.round(y_pred)

        acc = accuracy(y_pred, b_img, b_gt)
        dice = dice_score(y_pred, b_gt)

        if self.save_test_results:
            affine = np.diag(np.asarray([1, 1, 1, 0]))
            hdr = nib.Nifti1Header()
            for i in range(len(b_img)):
                mask_img = nib.Nifti1Image(y_pred[i, ...].cpu().numpy(), affine, hdr)
                mask_img.to_filename(f"./data/mask/mask_{batch_idx}_{i}.nii.gz")

                img = nib.Nifti1Image(b_img[i, ...].cpu().numpy(), affine, hdr)
                img.to_filename(f"./data/images/image_{batch_idx}_{i}.nii.gz")

        logs = {'test_loss': loss,
                'test_acc': acc.mean(),
                'test_dice': dice.mean()
                }

        for i in range(len(b_img)):
            log_image(self.logger, img=b_img[i].permute((0, 2, 1)), title='test_Image', number=batch_idx * (i + 1))
            log_image(self.logger, img=b_gt[i].unsqueeze(0).permute((0, 2, 1)), title='test_GroundTruth', number=batch_idx * (i + 1))
            log_image(self.logger, img=y_pred[i].unsqueeze(0).permute((0, 2, 1)), title='test_Prediction', number=batch_idx * (i + 1),
                      img_text=acc[i].mean())

        self.log_dict(logs)

        return logs

    def on_test_end(self) -> None:
        self.save()

    def save(self) -> None:
        if self.ckpt_path:
            torch.save(self.net.state_dict(), self.ckpt_path)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        b_img, b_gt, dicoms, inst = batch['img'], batch['mask'], batch['dicom'], batch.get('instant', None)

        actions = self.forward(b_img)
        if self.output_shape[0] > 1:
            actions = actions.argmax(dim=1)
        else:
            actions = torch.round(actions)

        corrected, corrected_validity, ae_comp = self.pred_corrector.correct_batch(b_img, actions)

        initial_params = copy.deepcopy(self.net.state_dict())
        itr = 0
        df = self.trainer.datamodule.df
        for i in range(len(b_img)):
            if ae_comp[i] > 0.95 and check_segmentation_validity(actions[i].cpu().numpy().T, (1.0, 1.0), list(set(np.unique(actions[i].cpu().numpy())))):

                for j, multiplier in enumerate([0.005, 0.01]):
                    # get random seed based on time to maximise randomness of noise and subsequent predictions
                    # explore as much space around policy as possible
                    time_seed = int(round(datetime.now().timestamp())) + i
                    torch.manual_seed(time_seed)

                    # load initial params so noise is not compounded
                    self.net.load_state_dict(initial_params, strict=True)

                    # add noise to params
                    with torch.no_grad():
                        for param in self.net.parameters():
                            param.add_(torch.randn(param.size()).cuda() * multiplier)

                    # make prediction
                    deformed_action = self.forward(b_img[i].unsqueeze(0))
                    if self.output_shape[0] > 1:
                        deformed_action = deformed_action.argmax(dim=1)
                    else:
                        deformed_action = torch.round(deformed_action)

                    filename = f"{batch_idx}_{itr}_{i}_{time_seed}.nii.gz"
                    save_to_reward_dataset(self.predict_save_dir,
                                           filename,
                                           convert_to_numpy(b_img[i]),
                                           np.expand_dims(convert_to_numpy(b_gt[i]), 0),
                                           convert_to_numpy(deformed_action))
            else:
                if not corrected_validity[i]:
                    continue

                filename = f"{batch_idx}_{itr}_{i}_{int(round(datetime.now().timestamp()))}.nii.gz"
                save_to_reward_dataset(self.predict_save_dir,
                                       filename,
                                       convert_to_numpy(b_img[i]),
                                       convert_to_numpy(corrected[i]),
                                       np.expand_dims(convert_to_numpy(actions[i]), 0))

        df.to_csv(self.trainer.datamodule.df_path)
        # make sure initial params are back at end of step
        self.net.load_state_dict(initial_params)
