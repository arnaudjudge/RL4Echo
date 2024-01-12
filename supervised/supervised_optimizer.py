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
from utils.file_utils import get_img_subpath
from utils.logging_helper import log_image


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, output, target):
        intersection = torch.sum(target * output)
        return 1 - ((2. * intersection) / (torch.sum(target) + torch.sum(output)))


class SupervisedOptimizer(pl.LightningModule):
    def __init__(self, input_shape=(1, 256, 256), output_shape=(1, 256, 256), loss=nn.BCELoss(), ckpt_path=None, predict_save_dir=None, **kwargs):
        super().__init__(**kwargs)

        self.net = UNet(input_shape=input_shape, output_shape=output_shape)
        # self.net.load_state_dict(torch.load("./auto_iteration3/0/actor.ckpt"))
        self.input_shape = input_shape
        self.output_shape = output_shape

        self.loss = loss
        self.save_test_results = False
        self.ckpt_path = ckpt_path
        self.predict_save_dir = predict_save_dir

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
        b_img, b_gt, dicoms, inst = batch['img'], batch['mask'], batch['dicom'], batch['instant']
        ae = MorphologicalAndTemporalCorrectionAEApplicator("nathanpainchaud/echo-arvae")
        actions = self.forward(b_img)
        if self.output_shape[0] > 1:
            actions = actions.argmax(dim=1)
        else:
            actions = torch.round(actions)

        corrected = np.empty_like(b_img.cpu().numpy())
        corrected_validity = np.empty(len(b_img))
        ae_comp = np.empty(len(b_img))
        for i, act in enumerate(actions):
            c, _, _ = ae.fix_morphological_and_ae(act.unsqueeze(-1).cpu().numpy())
            corrected[i] = c.transpose((2, 0, 1))
            corrected_validity[i] = check_segmentation_validity(corrected[i, 0, ...].T, (1.0, 1.0),
                                                                list(set(np.unique(corrected[i]))))
            ae_comp[i] = compare_segmentation_with_ae(act.unsqueeze(0).cpu().numpy(), corrected[i])

        initial_params = copy.deepcopy(self.net.state_dict())
        itr = 0
        df = self.trainer.datamodule.df
        for i in range(len(b_img)):
            if ae_comp[i] > 0.95 and check_segmentation_validity(actions[i].cpu().numpy().T, (1.0, 1.0), list(set(np.unique(actions[i].cpu().numpy())))):


                for j, multiplier in enumerate([0.005, 0.01]): #, 0.025, 0.04]):
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
                    action = self.forward(b_img[i].unsqueeze(0))
                    if self.output_shape[0] > 1:
                        action = action.argmax(dim=1)
                    else:
                        action = torch.round(action)

                    # f, (ax1, ax2) = plt.subplots(1, 2)
                    # ax1.set_title(f"Original")
                    # ax1.imshow(actions[i, ...].cpu().numpy())
                    # ax2.set_title(f"{multiplier}, seed: {time_seed}")
                    # ax2.imshow(action[0, ...].cpu().numpy())
                    # plt.show()

                    Path(f"{self.predict_save_dir}/images").mkdir(parents=True, exist_ok=True)
                    Path(f"{self.predict_save_dir}/gt").mkdir(parents=True, exist_ok=True)
                    Path(f"{self.predict_save_dir}/pred").mkdir(parents=True, exist_ok=True)

                    filename = f"{batch_idx}_{itr}_{i}_{time_seed}.nii.gz"
                    affine = np.diag(np.asarray([1, 1, 1, 0]))
                    hdr = nib.Nifti1Header()

                    nifti_img = nib.Nifti1Image(b_img[i].cpu().numpy(), affine, hdr)
                    nifti_img.to_filename(f"./{self.predict_save_dir}/images/{filename}")

                    nifti_gt = nib.Nifti1Image(np.expand_dims(b_gt[i].cpu().numpy(), 0), affine, hdr)
                    nifti_gt.to_filename(f"./{self.predict_save_dir}/gt/{filename}")

                    nifti_pred = nib.Nifti1Image(np.expand_dims(action[0].cpu().numpy(), 0), affine, hdr)
                    nifti_pred.to_filename(f"./{self.predict_save_dir}/pred/{filename}")
            else:
                try:
                    # Find each blob in the image
                    # lbl, num = ndimage.label(actions[i, ...].cpu().numpy())
                    # # Count the number of elements per label
                    # count = np.bincount(lbl.flat)
                    # # Select the largest blob
                    # maxi = np.argmax(count[1:]) + 1
                    # # Keep only the other blobs
                    # lbl[lbl != maxi] = 0
                    # corrected, *_ = ransac_sector_extraction(lbl, slim_factor=0.01, circle_center_tol=0.45, plot=False)
                    # corrected = np.expand_dims(corrected, 0)
                    # if corrected.sum() < actions[i, ...].cpu().numpy().sum() * 0.1:
                    #     continue

                    # corrected, _, _ = ae.fix_morphological_and_ae(actions[i].unsqueeze(-1).cpu().numpy())
                    # corrected = corrected.transpose((2, 0, 1))
                    # anat_validity = check_segmentation_validity(corrected[0, ...].T, (1.0, 1.0),
                    #                                             list(set(np.unique(corrected))))
                    # f, (ax1, ax2) = plt.subplots(1, 2)
                    # ax1.set_title(f"Original")
                    # ax1.imshow(actions[i, ...].cpu().numpy())
                    # ax2.set_title(f"corrected w/ anatomical val {anat_validity}")
                    # ax2.imshow(corrected[0, ...])
                    # plt.show()

                    if not corrected_validity[i]:
                        continue

                    Path(f"{self.predict_save_dir}/images").mkdir(parents=True, exist_ok=True)
                    Path(f"{self.predict_save_dir}/gt").mkdir(parents=True, exist_ok=True)
                    Path(f"{self.predict_save_dir}/pred").mkdir(parents=True, exist_ok=True)

                    filename = f"{batch_idx}_{itr}_{i}_{random.randint(1, 1000)}.nii.gz"
                    affine = np.diag(np.asarray([1, 1, 1, 0]))
                    hdr = nib.Nifti1Header()

                    nifti_img = nib.Nifti1Image(b_img[i].cpu().numpy(), affine, hdr)
                    nifti_img.to_filename(f"./{self.predict_save_dir}/images/{filename}")

                    nifti_gt = nib.Nifti1Image(corrected[i], affine, hdr)
                    nifti_gt.to_filename(f"./{self.predict_save_dir}/gt/{filename}")

                    nifti_pred = nib.Nifti1Image(np.expand_dims(actions[i].cpu().numpy(), 0), affine, hdr)
                    nifti_pred.to_filename(f"./{self.predict_save_dir}/pred/{filename}")
                except Exception as e:
                    print(e)
                    pass

        df.to_csv(self.trainer.datamodule.df_path)
        # make sure initial params are back at end of step
        self.net.load_state_dict(initial_params)
