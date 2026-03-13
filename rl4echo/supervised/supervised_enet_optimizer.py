import copy
import random
from datetime import datetime
from typing import Dict, Any

import numpy as np
from lightning import LightningModule
import torch
from torch import nn, Tensor
from torchmetrics.classification import Dice
from vital.metrics.camus.anatomical.utils import check_segmentation_validity
from vital.models.segmentation.enet import Enet
from vital.data.camus.config import Label

from rl4echo.utils.Metrics import accuracy, dice_score, is_anatomically_valid
from rl4echo.utils.file_utils import save_to_reward_dataset
from rl4echo.utils.logging_helper import log_image
from rl4echo.utils.tensor_utils import convert_to_numpy
from rl4echo.utils.test_metrics import dice, hausdorff


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, output, target):
        intersection = torch.sum(target * output)
        return 1 - ((2. * intersection) / (torch.sum(target) + torch.sum(output)))

def replace_batchnorm_with_groupnorm(module, max_groups=8):
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            num_channels = child.num_features

            # choose largest valid group <= max_groups
            for g in reversed(range(1, max_groups + 1)):
                if num_channels % g == 0:
                    num_groups = g
                    break

            setattr(
                module,
                name,
                nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)
            )
        else:
            replace_batchnorm_with_groupnorm(child, max_groups)


class SupervisedEnetOptimizer(LightningModule):
    def __init__(self,
                 input_shape=(1, 256, 256),
                 output_shape=(1, 256, 256),
                 loss=nn.BCELoss(),
                 ckpt_path=None,
                 **kwargs
        ):
        super().__init__(**kwargs)

        self.net = Enet(input_shape=input_shape, output_shape=output_shape)
        replace_batchnorm_with_groupnorm(self.net)
        self.input_shape = input_shape
        self.output_shape = output_shape

        self.loss = loss
        self.ckpt_path = ckpt_path

        self.dice = Dice()

    def forward(self, x):
        out = self.net.forward(x)
        if self.output_shape[0] > 1:
            out = torch.softmax(out, dim=1)
        else:
            out = torch.sigmoid(out).squeeze(1)
        return out

    def configure_optimizers(self):
        # add weight decay so predictions are less certain, more randomness?
        return torch.optim.Adam(self.parameters(), lr=0.001, weight_decay=0)

    def training_step(self, batch: dict[str, Tensor], *args, **kwargs) -> Dict:
        x, y = batch['img'], batch['gt']

        y_hat = self.forward(x)

        loss = self.loss(y_hat, y)

        logs = {
            'loss': loss,
        }

        self.log_dict(logs, prog_bar=True)
        return logs

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int):
        b_img, b_gt = batch['img'], batch['gt']
        y_pred = self.forward(b_img)

        loss = self.loss(y_pred, b_gt)

        if self.output_shape[0] > 1:
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
        log_image(self.logger, img=b_img[idx].permute((0, 2, 1)),
                  title='Image',
                  number=batch_idx,
                  epoch=self.current_epoch)
        log_image(self.logger, img=b_gt[idx].unsqueeze(0).permute((0, 2, 1)),
                  title='GroundTruth',
                  number=batch_idx,
                  epoch=self.current_epoch)
        log_image(self.logger, img=y_pred[idx].unsqueeze(0).permute((0, 2, 1)),
                  title='Prediction',
                  number=batch_idx,
                  img_text=acc[idx].mean(),
                  epoch=self.current_epoch)

        self.log_dict(logs)
        return logs

    def test_step(self, batch, batch_idx):
        b_img, b_gt, voxel_spacing = batch['img'], batch['gt'], batch['vox']
        y_pred = self.forward(b_img)

        loss = self.loss(y_pred, b_gt)

        if self.output_shape[0] > 1:
            y_pred = y_pred.argmax(dim=1)
        else:
            y_pred = torch.round(y_pred)

        acc = accuracy(y_pred, b_img, b_gt)
        simple_dice = dice_score(y_pred, b_gt)
        logs = {'test_loss': loss,
                'test_acc': acc.mean(),
                'test_dice': simple_dice.mean(),
                }

        for i in range(len(b_img)):
            log_image(self.logger, img=b_img[i].permute((0, 2, 1)),
                      title='test_Image',
                      number=batch_idx * (i + 1),
                      epoch=self.current_epoch)
            log_image(self.logger, img=b_gt[i].unsqueeze(0).permute((0, 2, 1)),
                      title='test_GroundTruth',
                      number=batch_idx * (i + 1),
                      epoch=self.current_epoch)
            log_image(self.logger, img=y_pred[i].unsqueeze(0).permute((0, 2, 1)),
                      title='test_Prediction',
                      number=batch_idx * (i + 1),
                      img_text=simple_dice[i].mean(),
                      epoch=self.current_epoch)

        self.log_dict(logs)

        return logs

    def on_test_end(self) -> None:
        self.save()

    def save(self) -> None:
        if self.ckpt_path:
            torch.save(self.net.state_dict(), self.ckpt_path)


if __name__ == "__main__":

    enet_optimizer = SupervisedEnetOptimizer.load_from_checkpoint("/home/local/USHERBROOKE/juda2901/dev/RL4Echo/rl4echo/test_logs/RL4Echo/8fa6524c78904f998a46d55cd1ab8aa7/checkpoints/epoch=38-step=624.ckpt")

    torch.save(enet_optimizer.net.state_dict(), "/home/local/USHERBROOKE/juda2901/dev/vitalab/Echo-Toolkit/echotk/enet_sector.ckpt")


