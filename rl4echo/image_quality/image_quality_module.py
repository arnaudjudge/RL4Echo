import pickle
import random
from typing import Dict

import torch
from lightning import LightningModule
from torch import nn, optim
from torch.nn import functional as F
from vital.models.segmentation.unet import UNet

from rl4echo.rewardnet.unet_heads import UNet_multihead
from rl4echo.utils.Metrics import accuracy
from rl4echo.utils.logging_helper import log_sequence, log_video


class IQ3DOptimizer(LightningModule):
    def __init__(self, net, loss=nn.BCELoss(), save_model_path=None, var_file=None, **kwargs):
        super().__init__(**kwargs)

        self.net = net
        self.loss = loss
        self.save_model_path = save_model_path

        from torchinfo import summary
        summary(self.net, input_size=(1, 1, 480, 560, 24))

        # class_counts = torch.tensor([48, 256, 248, 43, 5])  # your class counts
        # class_weights = 1.0 / class_counts.float()
        # class_weights = class_weights / class_weights.sum()  # normalize
        # print(f"Using class weights: {class_weights}")
        # self.loss = nn.CrossEntropyLoss(weight=class_weights.to(self.device))


    def forward(self, x):
        out = self.net(x)
        return out

    def configure_optimizers(self):
        opt = torch.optim.SGD(self.parameters(), lr=1e-4)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "min", patience=4)
        return {"optimizer": opt, "lr_scheduler": sch, "monitor": "val/loss"}

    def training_step(self, batch, *args, **kwargs) -> Dict:
        x, y = batch['img'], batch['label']

        # y_pred = self(x)
        # loss = self.loss(y_pred, y)
        y_pred = torch.sigmoid(self(x)).squeeze(0)
        # print(y_pred)
        loss = self.loss(y_pred, y)

        acc = (y == torch.round(y_pred))#.argmax(dim=1))

        logs = {
            'loss': loss,
            'acc': acc
        }

        self.log_dict(logs)
        return logs

    def validation_step(self, batch, batch_idx: int):
        x, y = batch['img'], batch['label']
        print(x.shape)
        print(y.shape)
        print(y)
        y_pred = torch.sigmoid(self(x)).squeeze(0)
        print(y_pred)
        loss = self.loss(y_pred, y)

        acc = (y == torch.round(y_pred))#.argmax(dim=1))

        self.log_dict({"val/loss": loss,
                       "val/acc": acc
                       })

        # log images
        if self.trainer.local_rank == 0:
            idx = random.randint(0, len(x) - 1)  # which image to log
            log_sequence(self.logger, img=x[idx].permute(0, 2, 3, 1), title=f'Image : label {y.item()}, pred {y_pred.item()}',
                         number=batch_idx, epoch=self.current_epoch)

        return {'loss': loss}

    def test_step(self, batch, batch_idx):
        x, y = batch['img'], batch['label']
        print(x.shape)
        print(y.shape)
        label = y
        # y_pred = self(x)
        #
        # loss = self.loss(y_pred, y)
        y_pred = torch.sigmoid(self(x)).squeeze(0)
        print(y_pred)
        loss = self.loss(y_pred, y)

        acc = (y == torch.round(y_pred))#.argmax(dim=1))

        self.log_dict({"test/loss": loss,
                       "test/acc": acc})

        if self.trainer.local_rank == 0:
            for i in range(len(x)):
                log_video(self.logger, img=x[i].permute(0, 2, 3, 1),
                          title=f'test_Image_Class - Label {label.item()} - Pred {y_pred.item()}',
                          number=batch_idx * (i + 1), epoch=self.current_epoch)

        return {'loss': loss}

    def on_test_end(self) -> None:
        self.save_model()

    def save_model(self):
        if self.save_model_path:
            sd = self.net.state_dict()
            torch.save(sd, self.save_model_path)
