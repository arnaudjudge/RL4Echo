import json

import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset

from torch.utils.data import random_split, DataLoader
import pytorch_lightning as pl
# from vital.vital.utils.image.transform import resize_image
from typing import Optional, Tuple

import nibabel as nib

import numpy as np
import json


class RewardNetAutoDataset(Dataset):
    def __init__(self, data_path, labels_file, test=False):
        super().__init__()
        self.data_path = data_path

        with open(labels_file, 'r') as f:
            self.lbl = json.load(f)

        self.img_list = []
        self.mask_list = []
        self.rating_list = []
        for im in self.lbl.keys():
            self.img_list += [f"{self.data_path}/images/{im}"]
            self.mask_list += [f"{self.data_path}/mask/{im}"]

            if self.lbl[im] == 0:  # not modified, valid mask
                rating = np.asarray([0, 1])#np.ones((1))
            else:
                rating = np.asarray([1, 0])#np.zeros((1))
            self.rating_list += [rating]

        if test:
            self.img_list = self.img_list[24000:]
            self.mask_list = self.mask_list[24000:]
            self.rating_list = self.rating_list[24000:]
        else:
            self.img_list = self.img_list[:-1000]
            self.mask_list = self.mask_list[:-1000]
            self.rating_list = self.rating_list[:-1000]

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img = nib.load(self.img_list[idx]).get_fdata()
        mask = nib.load(self.mask_list[idx]).get_fdata()
        data = np.vstack((img, mask))

        return torch.tensor(data), torch.tensor(self.rating_list[idx])


class RewardNetHumanDataset(Dataset):
    def __init__(self, data_path, qc_file, test=False):
        super().__init__()
        self.data_path = data_path

        with open(qc_file, 'r') as f:
            self.qc = json.load(f)

        self.img_list = []
        self.mask_list = []
        self.rating_list = []
        for im_qc in self.qc[1]['data']:
            self.img_list += [f"{self.data_path}/images/{im_qc['filename']}.nii.gz"]
            self.mask_list += [f"{self.data_path}/mask/{im_qc['filename'].replace('image', 'mask')}.nii.gz"]

            if "Pass" in im_qc['status']:
                rating = np.asarray([0, 1])#np.ones((1))
            elif "Warning" in im_qc['status']:
                rating = np.asarray([1, 0])#np.zeros((1))
            elif "Fail" in im_qc['status']:
                rating = np.asarray([1, 0])#np.zeros((1))
            else:
                raise ValueError("STATUS NOT VALID")
            self.rating_list += [rating]

        if test:
            self.img_list = self.img_list[990:]
            self.mask_list = self.mask_list[990:]
            self.rating_list = self.rating_list[990:]
        else:
            self.img_list = self.img_list[-10:]
            self.mask_list = self.mask_list[-10:]
            self.rating_list = self.rating_list[-10:]

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img = nib.load(self.img_list[idx]).get_fdata()
        mask = np.expand_dims(nib.load(self.mask_list[idx]).get_fdata(), 0)
        data = np.vstack((img, mask))

        return torch.tensor(data), torch.tensor(self.rating_list[idx])


class RewardNetDataModule(pl.LightningDataModule):
    """
    DataModule used for semantic segmentation in geometric generalization project
    """

    def __init__(self, data_path, qc_file):
        super().__init__()
        self.data_path = data_path
        self.qc_file = qc_file

    def prepare_data(self):
        """
        Empty prepare_data method left in intentionally.
        https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html#prepare-data
        """
        pass

    def prepare_data_per_node(self):
        pass

    def setup(self, stage: Optional[str] = None):
        """
        Method to setup your datasets, here you can use whatever dataset class you have defined in Pytorch and prepare the data in order to pass it to the loaders later
        https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html#setup
        """

        # Assign train/val datasets for use in dataloaders
        # the stage is used in the Pytorch Lightning trainer method, which you can call as fit (training, evaluation) or test, also you can use it for predict, not implemented here

        if stage == "fit" or stage is None:
            train_set_full = RewardNetAutoDataset(self.data_path, self.qc_file)
            train_set_size = int(len(train_set_full) * 0.9)
            valid_set_size = len(train_set_full) - train_set_size
            self.train, self.validate = random_split(train_set_full, [train_set_size, valid_set_size])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test = RewardNetAutoDataset(self.data_path, self.qc_file, test=True)

    # define your dataloaders
    # again, here defined for train, validate and test, not for predict as the project is not there yet.
    def train_dataloader(self):
        return DataLoader(self.train, batch_size=32, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.validate, batch_size=32, num_workers=8)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=1, num_workers=8)


if __name__ == "__main__":

    #dl = RewardNetDataModule('../data/', '../data/20230518_reward_net_dataset_echoqcpy.json')
    dl = RewardNetDataModule('./dataset_augmented_reward', './dataset_augmented_reward/labels.json')
    dl.setup()
    for i in range(1):
       batch = next(iter(dl.train_dataloader()))
       print(batch[0].shape)
       print(batch[1].shape)

       from matplotlib import pyplot as plt
       plt.figure()
       plt.imshow(batch[0][0, 0, :, :].cpu().numpy().T)
       plt.show()
       plt.figure()
       plt.imshow(batch[0][0, 1, :, :].cpu().numpy().T)
       plt.show()

