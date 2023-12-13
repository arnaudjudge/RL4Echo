import copy
from datetime import datetime
import random
from pathlib import Path
from typing import Dict, Any

import cv2
import nibabel as nib
import numpy as np
import pytorch_lightning as pl
import torch
from bdicardio.utils.ransac_utils import ransac_sector_extraction
from pytorch_lightning.loggers import TensorBoardLogger
from scipy import ndimage
from torch import nn, Tensor
from vital.vital.models.segmentation.unet import UNet

from utils.Metrics import accuracy, dice_score
from vital.vital.metrics.train.functional import differentiable_dice_score
from torchmetrics.classification import Dice
from utils.file_utils import save_batch_to_dataset, save_batch_to_dataset_v2
from utils.logging_helper import log_image


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, output, target):
        intersection = torch.sum(target * output)
        return 1 - ((2. * intersection) / (torch.sum(target) + torch.sum(output)))


class SupervisedOptimizer(pl.LightningModule):
    def __init__(self, ckpt_path=None, predict_save_dir=None, **kwargs):
        super().__init__(**kwargs)

        self.net = UNet(input_shape=(1, 256, 256), output_shape=(1, 256, 256))
        # self.net.load_state_dict(torch.load("./CGPT_Loop/supervised.ckpt"))

        self.loss = nn.BCELoss()
        self.save_test_results = False
        self.ckpt_path = ckpt_path
        self.predict_save_dir = predict_save_dir

        self.dice = Dice()

    def forward(self, x):
        return self.net.forward(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def training_step(self, batch: dict[str, Tensor], *args, **kwargs) -> Dict:
        x, y = batch['img'], batch['mask']

        y_hat = torch.sigmoid(self.forward(x))

        loss = self.loss(y_hat.squeeze(1), y)

        logs = {
            'loss': loss,
        }

        self.log_dict(logs, prog_bar=True)
        return logs

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int):
        b_img, b_gt = batch['img'], batch['mask']
        y_pred = torch.sigmoid(self.forward(b_img))

        loss = self.loss(y_pred.squeeze(1), b_gt)

        # test_gt = b_gt.unsqueeze(1)
        # a = 1 - test_gt
        # d_tensor = torch.stack((a, test_gt), dim=1).squeeze(2)
        #
        # should_be_gt = torch.argmax(d_tensor, dim=1)
        # assert (should_be_gt == b_gt).all()

        # 1 - differentiable_dice_score(d_tensor, b_gt)

        y_pred = torch.round(y_pred)
        acc = accuracy(y_pred, b_img, b_gt.unsqueeze(1))
        dice = dice_score(y_pred, b_gt.unsqueeze(1))
        # d = self.dice(y_pred, b_gt.unsqueeze(1).type(torch.int64))
        logs = {'val_loss': loss,
                'val_acc': acc.mean(),
                'val_dice': dice.mean(),
                # 'val_d': d.mean()
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
        b_img, b_gt = batch['img'], batch['mask']
        y_pred = torch.sigmoid(self.forward(b_img))

        loss = self.loss(y_pred, b_gt.unsqueeze(1))

        y_pred = torch.round(y_pred)

        acc = accuracy(y_pred, b_img, b_gt.unsqueeze(1))
        dice = dice_score(y_pred, b_gt.unsqueeze(1))

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
            log_image(self.logger, img=b_img[i], title='test_Image', number=batch_idx * (i + 1))
            log_image(self.logger, img=b_gt[i].unsqueeze(0), title='test_GroundTruth', number=batch_idx * (i + 1))
            log_image(self.logger, img=y_pred[i], title='test_Prediction', number=batch_idx * (i + 1),
                      img_text=acc[i].mean())

        self.log_dict(logs)

        #save_batch_to_dataset(b_img, b_gt, y_pred, batch_idx, './simple_reward_net/dataset_supervised/')

        return logs

    def on_test_end(self) -> None:
        self.save()

    def save(self) -> None:
        if self.ckpt_path:
            torch.save(self.net.state_dict(), self.ckpt_path)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        b_img, b_gt, dicoms = batch['img'], batch['mask'], batch['dicom']
        actions = torch.sigmoid(self.forward(b_img))
        acc = accuracy(actions, b_img, b_gt.unsqueeze(1))

        initial_params = copy.deepcopy(self.net.state_dict())
        itr = 0
        df = self.trainer.datamodule.df
        for i in range(len(b_img)):
            if acc[i] > 0.99:
                df.loc[df['dicom_uuid'] == dicoms[i], self.trainer.datamodule.hparams.gt_column] = True
                df.loc[df['dicom_uuid'] == dicoms[i], self.trainer.datamodule.hparams.splits_column] = 'train'

                for j, multiplier in enumerate([0.005, 0.01]): #, 0.025, 0.04]):
                    # get random seed based on time to maximise randomness of noise and subsequent predictions
                    # explore as much space around policy as possible
                    time_seed = int(round(datetime.now().timestamp())) + i
                    torch.manual_seed(1695228288)

                    # load initial params so noise is not compounded
                    self.net.load_state_dict(initial_params, strict=True)

                    # add noise to params
                    with torch.no_grad():
                        for param in self.net.parameters():
                            param.add_(torch.randn(param.size()).cuda() * multiplier)

                    # make prediction
                    action = torch.sigmoid(self.forward(b_img[i].unsqueeze(0)))
                    action = torch.round(action)

                    # import matplotlib.pyplot as plt
                    # plt.figure()
                    # plt.title(f"{multiplier}, seed: {time_seed}")
                    # plt.imshow(action[0, 0, ...].cpu().numpy())
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

                    nifti_pred = nib.Nifti1Image(action[0].cpu().numpy(), affine, hdr)
                    nifti_pred.to_filename(f"./{self.predict_save_dir}/pred/{filename}")
            else:
                try:
                    # Find each blob in the image
                    lbl, num = ndimage.label(actions[i, 0, ...].cpu().numpy())
                    # Count the number of elements per label
                    count = np.bincount(lbl.flat)
                    # Select the largest blob
                    maxi = np.argmax(count[1:]) + 1
                    # Keep only the other blobs
                    lbl[lbl != maxi] = 0
                    ransac, *_ = ransac_sector_extraction(lbl, slim_factor=0.01, circle_center_tol=0.45, plot=False)

                    if ransac.sum() < actions[i, 0, ...].cpu().numpy().sum() * 0.1:
                        continue
                    # f, (ax1, ax2) = plt.subplots(1, 2)
                    # ax1.set_title(f"Original")
                    # ax1.imshow(actions[i, 0, ...].cpu().numpy())
                    #
                    # ax2.set_title(f"ransac")
                    # ax2.imshow(ransac)
                    # plt.show()

                    Path(f"{self.predict_save_dir}/images").mkdir(parents=True, exist_ok=True)
                    Path(f"{self.predict_save_dir}/gt").mkdir(parents=True, exist_ok=True)
                    Path(f"{self.predict_save_dir}/pred").mkdir(parents=True, exist_ok=True)

                    filename = f"{batch_idx}_{itr}_{i}_{random.randint(1, 1000)}.nii.gz"
                    affine = np.diag(np.asarray([1, 1, 1, 0]))
                    hdr = nib.Nifti1Header()

                    nifti_img = nib.Nifti1Image(b_img[i].cpu().numpy(), affine, hdr)
                    nifti_img.to_filename(f"./{self.predict_save_dir}/images/{filename}")

                    nifti_gt = nib.Nifti1Image(np.expand_dims(ransac, 0), affine, hdr)
                    nifti_gt.to_filename(f"./{self.predict_save_dir}/gt/{filename}")

                    nifti_pred = nib.Nifti1Image(actions[i].cpu().numpy(), affine, hdr)
                    nifti_pred.to_filename(f"./{self.predict_save_dir}/pred/{filename}")
                except:
                    pass

        df.to_csv(self.trainer.datamodule.df_path)
        # make sure initial params are back at end of step
        self.net.load_state_dict(initial_params)
