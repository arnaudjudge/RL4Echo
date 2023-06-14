from typing import Dict

import pytorch_lightning as pl
import torch
from torch import nn
from torchvision import models


def get_resnet(input_channels, num_classes) -> nn.Module:
    discriminator = models.resnet18(num_classes=num_classes)
    discriminator.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    # discriminator = patch_module(discriminator)
    return discriminator


class RewardOptimizer(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.net = get_resnet(input_channels=2, num_classes=2)

        self.loss = nn.BCELoss()

    def forward(self, x):
        out = self.net.forward(x.type(torch.cuda.FloatTensor))
        return out

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=1e-6)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "min")
        return {"optimizer": opt, "lr_scheduler": sch, "monitor": "val_loss"}

    def training_step(self, batch, *args, **kwargs) -> Dict:
        x, y = batch

        y_pred = torch.softmax(self.forward(x), dim=-1)
        loss = self.loss(y_pred, y.type(torch.float))

        acc = (y.argmax(dim=-1) == y_pred.argmax(dim=-1)).float().mean()

        logs = {
            'loss': loss,
            'acc': acc
        }

        self.log_dict(logs)
        return logs

    def validation_step(self, batch, batch_idx: int):

        imgs, y_true = batch
        y_pred = torch.softmax(self.forward(imgs), dim=-1)
        loss = self.loss(y_pred, y_true.type(torch.float))

        acc = (y_true.argmax(dim=-1) == y_pred.argmax(dim=-1)).float().mean()

        self.log_dict({"val_loss": loss,
                       "val_acc": acc})

        return {'loss': loss}

    def test_step(self, batch, batch_idx):
        imgs, y_true = batch
        y_pred = torch.softmax(self.forward(imgs), dim=-1)
        out = torch.argmax(y_pred).cpu().numpy()

        from matplotlib import pyplot as plt
        plt.figure()
        plt.title(f'Label {y_true.argmax().cpu().numpy()} - Pred {out}')
        plt.imshow(imgs[0, 1, :, :].cpu().numpy().T)
        plt.show()

    def save_model(self, path):
        sd = self.net.state_dict()
        torch.save(sd, path)
