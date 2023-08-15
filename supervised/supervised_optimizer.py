import random
from typing import Dict

import cv2
import nibabel as nib
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn
from vital.vital.models.segmentation.unet import UNet

from utils.Metrics import accuracy
from utils.file_utils import save_batch_to_dataset
from utils.logging_helper import log_image


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, output, target):
        intersection = torch.sum(target * output)
        return 1 - ((2. * intersection) / (torch.sum(target) + torch.sum(output)))


class SupervisedOptimizer(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.net = UNet(input_shape=(1, 256, 256), output_shape=(1, 256, 256))

        self.loss = nn.BCELoss()
        self.save_test_results = False

    def forward(self, x):
        return self.net.forward(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def training_step(self, batch, *args, **kwargs) -> Dict:
        x, y, *_ = batch

        y_hat = torch.sigmoid(self.forward(x))

        loss = self.loss(y_hat.squeeze(1), y)

        logs = {
            'loss': loss,
        }

        self.log_dict(logs, prog_bar=True)
        return logs

    def validation_step(self, batch, batch_idx: int):
        b_img, b_gt, *_ = batch
        y_pred = torch.sigmoid(self.forward(b_img))

        loss = self.loss(y_pred.squeeze(1), b_gt)

        y_pred = torch.round(y_pred)
        acc = accuracy(y_pred, b_img, b_gt.unsqueeze(1))

        logs = {'val_loss': loss,
                'val_acc': acc.mean(),
                }

        # log images
        idx = random.randint(0, len(b_img) - 1)  # which image to log
        log_image(self.logger, img=b_img[idx], title='Image', number=batch_idx)
        log_image(self.logger, img=b_gt[idx].unsqueeze(0), title='GroundTruth', number=batch_idx)
        log_image(self.logger, img=y_pred[idx], title='Prediction', number=batch_idx,
                  img_text=acc[idx].mean())

        self.log_dict(logs)
        return logs

    def test_step(self, batch, batch_idx):
        b_img, b_gt, *_ = batch
        y_pred = torch.sigmoid(self.forward(b_img))

        loss = self.loss(y_pred, b_gt.unsqueeze(1))

        y_pred = torch.round(y_pred)

        acc = accuracy(y_pred, b_img, b_gt.unsqueeze(1))

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
                }

        print(acc.mean())

        for i in range(len(b_img)):
            log_image(self.logger, img=b_img[i], title='test_Image', number=batch_idx * (i + 1))
            log_image(self.logger, img=b_gt[i].unsqueeze(0), title='test_GroundTruth', number=batch_idx * (i + 1))
            log_image(self.logger, img=y_pred[i], title='test_Prediction', number=batch_idx * (i + 1),
                      img_text=acc[i].mean())

        self.log_dict(logs)
        return logs


    def save(self) -> None:
        torch.save(self.net.state_dict(), 'supervised.ckpt')