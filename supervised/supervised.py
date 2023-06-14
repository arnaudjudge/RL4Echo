from typing import Dict

import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers
import torch
from torch import nn
from torch.distributions.categorical import Categorical
from vital.vital.models.segmentation.unet import UNet
from vital.vital.models.segmentation.enet import Enet
from vital.vital.metrics.train.functional import differentiable_dice_score
import random
import nibabel as nib
import numpy as np


class SupervisedOptimizer(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.net = UNet(input_shape=(1, 256, 256), output_shape=(2, 256, 256))

        self.loss = nn.BCELoss()

    def forward(self, x):
        return self.net.forward(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def training_step(self, batch, *args, **kwargs) -> Dict:
        x, y = batch

        y_hat = self.forward(x)

        loss = -differentiable_dice_score(y_hat, y)

        logs = {
            'loss': loss,
        }

        self.log_dict(logs)
        return logs

    def validation_step(self, batch, batch_idx: int):

        imgs, y_true = batch
        y_pred = self.forward(imgs)

        loss = -differentiable_dice_score(y_pred, y_true)

        y_pred = torch.argmax(nn.functional.softmax(y_pred, dim=1), dim=1)

        if batch_idx % 100:  # Log every 10 batches
            self.log_tb_images((imgs, y_true, y_pred, batch_idx))

        logs = {'val_loss': loss}
        self.log_dict(logs)

        return logs

    def test_step(self, batch, batch_idx):
        imgs, y_true = batch
        y_pred = self.forward(imgs)

        loss = -differentiable_dice_score(y_pred, y_true)

        y_pred = torch.argmax(nn.functional.softmax(y_pred, dim=1), dim=1)

        affine = np.diag(np.asarray([1, 1, 1, 0]))
        hdr = nib.Nifti1Header()
        for i in range(len(imgs)):
            mask_img = nib.Nifti1Image(y_pred[i, ...].cpu().numpy(), affine, hdr)
            mask_img.to_filename(f"./data/mask/mask_{batch_idx}_{i}.nii.gz")

            img = nib.Nifti1Image(imgs[i, ...].cpu().numpy(), affine, hdr)
            img.to_filename(f"./data/images/image_{batch_idx}_{i}.nii.gz")

        return {'loss': loss}

    def log_tb_images(self, viz_batch) -> None:

        # Get tensorboard logger
        tb_logger = None
        for logger in self.trainer.loggers:
            if isinstance(logger, pl_loggers.TensorBoardLogger):
                tb_logger = logger.experiment
                break

        if tb_logger is None:
            raise ValueError('TensorBoard Logger not found')

        idx = random.randint(0, len(viz_batch[0])-1)

        tb_logger.add_image(f"Image", viz_batch[0][idx], viz_batch[3])
        tb_logger.add_image(f"GroundTruth", viz_batch[1][idx].unsqueeze(0), viz_batch[3])
        tb_logger.add_image(f"Prediction", viz_batch[2][idx].unsqueeze(0), viz_batch[3])

    def save(self) -> None:
        torch.save(self.net.state_dict(), './supervised.ckpt')