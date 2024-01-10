import random
from typing import Dict

import pytorch_lightning as pl
import torch
from torch import nn
from torchvision import models
from vital.models.segmentation.unet import UNet

from utils.Metrics import accuracy
from utils.logging_helper import log_image


class RewardOptimizer(pl.LightningModule):
    def __init__(self, save_model_path=None, **kwargs):
        super().__init__(**kwargs)

        self.net = UNet(input_shape=(2, 256, 256), output_shape=(1, 256, 256))

        self.loss = nn.BCELoss()
        self.save_model_path = save_model_path

    def forward(self, x):
        out = self.net.forward(x)
        return out

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=1e-3)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "min")
        return {"optimizer": opt, "lr_scheduler": sch, "monitor": "val_loss"}

    def training_step(self, batch, *args, **kwargs) -> Dict:
        x, y = batch

        y_pred = torch.sigmoid(self.forward(x))
        loss = self.loss(y_pred, y)

        logs = {
            'loss': loss,
        }

        self.log_dict(logs)
        return logs

    def validation_step(self, batch, batch_idx: int):

        imgs, y_true = batch
        y_pred = torch.sigmoid(self.forward(imgs))
        loss = self.loss(y_pred, y_true)

        acc = accuracy(y_pred, imgs, y_true)

        self.log_dict({"val_loss": loss,
                       "val_acc": acc.mean()})

        # log images
        idx = random.randint(0, len(imgs) - 1)  # which image to log
        log_image(self.logger, img=imgs[idx], title='Image', number=batch_idx)
        log_image(self.logger, img=y_true[idx], title='GroundTruth', number=batch_idx)
        log_image(self.logger, img=y_pred[idx], title='Prediction', number=batch_idx,
                  img_text=acc[idx].mean())

        return {'loss': loss}

    def test_step(self, batch, batch_idx):
        imgs, y_true = batch
        y_pred = torch.sigmoid(self.forward(imgs))
        loss = self.loss(y_pred, y_true)

        acc = accuracy(y_pred, imgs, y_true)

        self.log_dict({"test_loss": loss,
                       "test_acc": acc.mean()})

        for i in range(len(imgs)):
            log_image(self.logger, img=imgs[i], title='test_Image', number=batch_idx * (i + 1))
            log_image(self.logger, img=y_true[i], title='test_GroundTruth', number=batch_idx * (i + 1))
            log_image(self.logger, img=y_pred[i], title='test_Prediction', number=batch_idx * (i + 1),
                      img_text=acc[i].mean())

        return {'loss': loss}

    def on_test_end(self) -> None:
        self.save_model()

    def save_model(self):
        if self.save_model_path:
            sd = self.net.state_dict()
            torch.save(sd, self.save_model_path)
