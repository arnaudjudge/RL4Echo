import copy
import random
from datetime import datetime
from typing import Dict, Any

import nibabel as nib
import numpy as np
import pytorch_lightning as pl
import torch
from scipy import ndimage
from torch import nn, Tensor
from torchmetrics.classification import Dice
from vital.metrics.camus.anatomical.utils import check_segmentation_validity
from vital.models.segmentation.unet import UNet
from vital.data.camus.config import Label

from datamodules.ES_ED_datamodule import ESEDDataModule
from utils.Metrics import accuracy, dice_score, is_anatomically_valid
from utils.file_utils import save_to_reward_dataset
from utils.logging_helper import log_image
from utils.tensor_utils import convert_to_numpy
from utils.test_metrics import dice, hausdorff

from supervised.ts_it_helpers import *


from matplotlib import pyplot as plt

image_transforms = Compose([RandomApply_Customized([
                                   ColorJitter(brightness=0.6, contrast=0.6, saturation=0.4, hue=0),
                                   One_Of([
                                   GaussianBlur(kernel_size=(5,5)),
                                   RandomAdjustSharpness(sharpness_factor=1.5)
                                   ])
                                   ], p = 1)
                                   ])


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, output, target):
        intersection = torch.sum(target * output)
        return 1 - ((2. * intersection) / (torch.sum(target) + torch.sum(output)))


class TSITOptimizer(pl.LightningModule):
    def __init__(self, input_shape=(1, 256, 256), output_shape=(1, 256, 256), loss=nn.BCELoss(), ckpt_path=None, class_label=1, **kwargs):
        super().__init__(**kwargs)

        self.net = UNet(input_shape=input_shape, output_shape=output_shape)
        # self.net.load_state_dict(torch.load("./auto_iteration3/0/actor.ckpt"))
        self.input_shape = input_shape
        self.output_shape = output_shape

        self.loss = loss
        self.save_test_results = False
        self.ckpt_path = ckpt_path

        self.dice = Dice()
        print("TS-IT")

        self.icardio_dl = ESEDDataModule(
            data_dir='/home/local/USHERBROOKE/juda2901/dev/data/icardio/ES_ED_train_subset_affine/',
            csv_file='subset_official_test.csv',
            test_frac=0.1,
            class_label=class_label)
        self.class_label = class_label
        self.icardio_dl.setup('fit')
        self.test_loader_ensemble = self.icardio_dl.train_dataloader()
        self.val_test_loader_ensemble = self.icardio_dl.val_dataloader()
        self.test_loader_ensemble_iterator = iter(self.test_loader_ensemble)

        self.ens_loss = Ens_loss(thr=0.85)
        self.SemiSup_initial_epoch = 25
        # self.supervised_share = 1
        # self.ensemble_batch_size = 4
        self.GCC = 2
        self.LW = 0.5

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
        x, y = batch['img'], batch['gt'].float()

        y_hat = self.forward(x)

        loss = self.loss(y_hat, y)

        if self.current_epoch > self.SemiSup_initial_epoch - 1:
            try:
                Unsup_batch = next(self.test_loader_ensemble_iterator)
            except StopIteration:
                test_loader_ensemble_iterator = iter(self.test_loader_ensemble)
                Unsup_batch = next(test_loader_ensemble_iterator)
            imgs_T = Unsup_batch['img']
            imgs_T = imgs_T.to(device=self.device, dtype=torch.float32)

            rs = np.random.randint(2147483647, size=1)
            rs = rs[0]

            random.seed(rs)
            torch.random.manual_seed(rs)
            imgs_TA, NI = image_transforms(imgs_T, torch.zeros_like(imgs_T))

            masks_TA = self.forward(imgs_TA)

            with torch.no_grad():
                masks_T = self.forward(imgs_T)
            # import matplotlib.pyplot as plt
            # plt.figure()
            # plt.imshow(imgs_TA[0, 0, ...].cpu().numpy().T)
            # plt.figure()
            # plt.imshow(imgs_T[0, 0, ...].cpu().numpy().T)
            # plt.show()

            loss_T = self.ens_loss(masks_TA, masks_T)
            Ti = 1 / (self.trainer.max_epochs - self.SemiSup_initial_epoch)
            Lambda = self.LW * torch.exp(torch.tensor(0 - self.GCC * (1 - (self.current_epoch - self.SemiSup_initial_epoch) * Ti)))

            loss = loss + Lambda * loss_T
        else:
            Lambda = 0

        logs = {
            'loss': loss,
            'lambda': Lambda
        }

        self.log_dict(logs, prog_bar=True)
        return logs

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int):
        b_img, b_gt = batch['img'], batch['gt'].float()
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
        log_image(self.logger, img=b_img[idx].permute((0, 2, 1)), title='Image', number=batch_idx)
        log_image(self.logger, img=b_gt[idx].unsqueeze(0).permute((0, 2, 1)), title='GroundTruth', number=batch_idx)
        log_image(self.logger, img=y_pred[idx].unsqueeze(0).permute((0, 2, 1)), title='Prediction', number=batch_idx,
                  img_text=acc[idx].mean())

        self.log_dict(logs)
        return logs

    def test_step(self, batch, batch_idx):
        b_img, b_gt, voxel_spacing = batch['img'], batch['gt'].float(), batch['vox']
        y_pred = self.forward(b_img)

        loss = self.loss(y_pred, b_gt)

        if self.output_shape[0] > 1:
            y_pred = y_pred.argmax(dim=1)
        else:
            y_pred = torch.round(y_pred)

        acc = accuracy(y_pred, b_img, b_gt)
        simple_dice = dice_score(y_pred, b_gt)
        #anat_errors = is_anatomically_valid(y_pred)

        labels = (Label.BG, Label.LV)  # only 0 or 1 since done separately here
        y_pred_np = y_pred.cpu().numpy()
        b_gt_np = b_gt.cpu().numpy()

        for i in range(len(y_pred_np)):
            # if y_pred_np[i].sum() == 0:
            #     y_pred_np[i] = b_gt_np[i]
            lbl, num = ndimage.measurements.label(y_pred_np[i])
            # Count the number of elements per label
            count = np.bincount(lbl.flat)
            # Select the largest blob
            maxi = np.argmax(count[1:]) + 1
            # Remove the other blobs
            lbl[lbl != maxi] = 0
            lbl[lbl == maxi] = 1
            y_pred_np[i] = lbl.astype(np.uint8)

        test_dice = dice(y_pred_np, b_gt_np,
                         labels=labels, exclude_bg=True, all_classes=True)

        test_hd = hausdorff(y_pred_np, b_gt_np,
                         labels=labels, exclude_bg=True, all_classes=False, voxel_spacing=voxel_spacing.cpu().numpy())
        logs = {'test_loss': loss,
                'test_acc': acc.mean(),
                'test_dice': simple_dice.mean(),
                #'test_anat_valid': anat_errors.mean()
                }
        logs.update(test_dice)
        logs.update(test_hd)

        for i in range(len(b_img)):
            log_image(self.logger, img=b_img[i].permute((0, 2, 1)), title='test_Image', number=batch_idx * (i + 1))
            log_image(self.logger, img=b_gt[i].unsqueeze(0).permute((0, 2, 1)), title='test_GroundTruth', number=batch_idx * (i + 1))
            log_image(self.logger, img=y_pred[i].unsqueeze(0).permute((0, 2, 1)), title='test_Prediction', number=batch_idx * (i + 1),
                      img_text=simple_dice[i].mean())

        self.log_dict(logs)

        return logs

    def on_test_end(self) -> None:
        self.save()

    def save(self) -> None:
        if self.ckpt_path:
            torch.save(self.net.state_dict(), self.ckpt_path)

