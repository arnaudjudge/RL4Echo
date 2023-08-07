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

from Reward import accuracy


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
        x, y = batch

        y_hat = torch.sigmoid(self.forward(x))

        loss = self.loss(y_hat.squeeze(1), y)

        logs = {
            'loss': loss,
        }

        self.log_dict(logs, prog_bar=True)
        return logs

    def validation_step(self, batch, batch_idx: int):

        b_img, b_gt = batch
        y_pred = torch.sigmoid(self.forward(b_img))

        loss = self.loss(y_pred.squeeze(1), b_gt)

        y_pred = torch.round(y_pred)
        acc = accuracy(y_pred, b_img, b_gt.unsqueeze(1))

        logs = {'val_loss': loss,
                'val_acc': acc.mean(),
                }

        self.log_tb_images((b_img[0, ...].unsqueeze(0),
                            y_pred[0, ...].unsqueeze(0),
                            acc[0, ...].unsqueeze(0),
                            b_gt[0, ...].unsqueeze(0),
                            batch_idx))

        self.log_dict(logs)
        return logs

    def test_step(self, batch, batch_idx):
        b_img, b_gt = batch
        y_pred = torch.sigmoid(self.forward(b_img))

        loss = self.loss(y_pred, b_gt.unsqueeze(1))  # -differentiable_dice_score(y_pred, y_true)

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

        self.log_tb_images((b_img[0, ...].unsqueeze(0),
                            y_pred[0, ...].unsqueeze(0),
                            acc[0, ...].unsqueeze(0),
                            b_gt[0, ...].unsqueeze(0),
                            batch_idx), prefix='test_')

        self.log_dict(logs)
        return logs

    # TODO: REMOVE THIS CODE DUPLICATE
    def log_tb_images(self, viz_batch, prefix="") -> None:
        """
            Log images to tensor board (Could this be simply for any logger without change?)
        Args:
            viz_batch: batch of images and metrics to log
            prefix: prefix to add to image titles

        Returns:
            None
        """

        # Get tensorboard logger
        tb_logger = None
        for logger in self.trainer.loggers:
            if isinstance(logger, TensorBoardLogger):
                tb_logger = logger.experiment
                break
        if tb_logger is None:
            raise ValueError('TensorBoard Logger not found')

        idx = random.randint(0, len(viz_batch[0]) - 1)

        tb_logger.add_image(f"{prefix}Image", viz_batch[0][idx], viz_batch[4])
        tb_logger.add_image(f"{prefix}GroundTruth", viz_batch[3][idx].unsqueeze(0), viz_batch[4])

        def put_text(img, text):
            img = img.copy().astype(np.uint8) * 255
            return cv2.putText(img.squeeze(0), "{:.3f}".format(text), (0, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (125), 2)

        tb_logger.add_image(f"{prefix}Prediction", torch.tensor(
            put_text(viz_batch[1][idx].cpu().detach().numpy(), viz_batch[2][idx].float().mean().item())).unsqueeze(0),
                            viz_batch[4])

    def save(self) -> None:
        torch.save(self.net.state_dict(), 'supervised.ckpt')