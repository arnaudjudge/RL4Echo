import random
from typing import Dict

import random
from typing import Dict

import numpy as np
import torch
from torch import Tensor

from patchless_nnunet.models.patchless_nnunet_module import nnUNetPatchlessLitModule
from rl4echo.utils.logging_helper import log_sequence


class MaskSelfSupervised3DOptimizer(nnUNetPatchlessLitModule):
    def __init__(self, ckpt_path=None, mask_ratio=0.6, **kwargs):
        super().__init__(**kwargs)

        self.ckpt_path = ckpt_path

    def forward(self, x):
        out = self.net.forward(x)
        out = torch.sigmoid(out).squeeze(1)
        return out

    def configure_optimizers(self):
        # add weight decay so predictions are less certain, more randomness?
        return torch.optim.Adam(self.parameters(), lr=0.001, weight_decay=0)

    def training_step(self, batch: dict[str, Tensor], *args, **kwargs) -> Dict:
        x = batch['img'].squeeze(0)
        y = batch['img'].squeeze(0, 2).detach().clone()

        # mask some frames 60% was best for SimLVSeg
        n_masks = int(x.shape[-1] * self.hparams.mask_ratio + 0.5)
        masks_ids = np.sort(np.random.choice(x.shape[-1], n_masks, replace=False))
        x[..., masks_ids] = 0

        y_hat = self.forward(x)

        loss = self.loss(y_hat, y)

        logs = {
            'loss': loss,
        }

        self.log_dict(logs, prog_bar=True)
        return logs

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int):
        x = batch['img'].squeeze(0)
        y = batch['img'].squeeze(0, 2).detach().clone()

        n_masks = int(x.shape[-1] * self.hparams.mask_ratio + 0.5)
        masks_ids = np.sort(np.random.choice(x.shape[-1], n_masks, replace=False))
        x[..., masks_ids] = 0

        y_pred = self.forward(x)

        loss = self.loss(y_pred, y)

        logs = {'val/loss': loss}

        # log images
        if self.trainer.local_rank == 0:
            idx = random.randint(0, len(x) - 1)  # which image to log
            log_sequence(self.logger, img=x[idx], title='Image', number=batch_idx, epoch=self.current_epoch)
            log_sequence(self.logger, img=y[idx].unsqueeze(0), title='GroundTruth', number=batch_idx,
                         epoch=self.current_epoch)
            log_sequence(self.logger, img=y_pred[idx].unsqueeze(0), title='Prediction', number=batch_idx,
                         epoch=self.current_epoch)

        self.log_dict(logs)

        return logs

    # don't want these functions from parent class
    def on_validation_epoch_end(self) -> None:
        return

    def on_test_epoch_end(self) -> None:
        return

    def test_step(self, batch, batch_idx):
        x = batch['img'].squeeze(0)
        y = batch['img'].squeeze(0, 2).detach().clone()
        n_masks = int(x.shape[-1] * self.hparams.mask_ratio + 0.5)
        masks_ids = np.sort(np.random.choice(x.shape[-1], n_masks, replace=False))
        x[..., masks_ids] = 0

        y_pred = self.forward(x)

        loss = self.loss(y_pred, y)

        logs = {'test/loss': loss}

        # log images
        if self.trainer.local_rank == 0:
            idx = random.randint(0, len(x) - 1)  # which image to log
            log_sequence(self.logger, img=x[idx], title='Test_Image', number=batch_idx, epoch=self.current_epoch)
            log_sequence(self.logger, img=y[idx].unsqueeze(0), title='Test_GroundTruth', number=batch_idx,
                         epoch=self.current_epoch)
            log_sequence(self.logger, img=y_pred[idx].unsqueeze(0), title='Test_Prediction', number=batch_idx,
                         epoch=self.current_epoch)

        self.log_dict(logs)

        return logs

    def on_test_end(self) -> None:
        self.save()

    def save(self) -> None:
        if self.ckpt_path:

            from patchless_nnunet.models.components.unet_related.layers import OutputBlock
            self.net.output_block = OutputBlock(
                in_channels=self.net.filters[0],
                out_channels=3,  # change num classes
                dim=self.net.dim,
                bias=self.net.out_seg_bias,
            )

            torch.save(self.net.state_dict(), self.ckpt_path)
