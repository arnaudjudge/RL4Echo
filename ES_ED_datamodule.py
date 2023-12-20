import json
import os
from typing import Optional

import nibabel as nib
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torch.utils.data import random_split, DataLoader


def get_img_subpath(row):
    """
    Format string for path of image in file structure
    :param row: dataframe row with all columns filled in
    :return: string containing path to image file
    """
    return f"{row['study']}/{row['view'].lower()}/{row['dicom_uuid']}_img_{row['instant']}.nii.gz"


class ESEDDataset(Dataset):
    def __init__(self, df, data_path, subset_frac=1.0, available_gt=None, seed=0, test=False, *args,
                 **kwargs):
        super().__init__()
        self.df = df
        self.data_path = data_path
        self.test = test

        print(f"Test step: {self.test} , len of dataset {len(self.df)}")

        # pass down list of available ground truths
        self.use_gt = available_gt.fillna(False).to_numpy() if available_gt is not None else \
            np.asarray([False for _ in range(len(self.df))])
        print(f"Number of ground truths available: {self.use_gt.sum()}")

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, idx):
        sub_path = get_img_subpath(self.df.iloc[idx])

        img = np.expand_dims(nib.load(self.data_path + '/' + sub_path).get_fdata(), 0) / 255
        mask = nib.load(self.data_path + '/' + sub_path.replace("img", 'mask')).get_fdata()

        approx_gt_path = self.data_path + '/' + sub_path.replace("img", 'approx_gt')
        if self.use_gt[idx]:
            approx_gt = nib.load(approx_gt_path).get_fdata()
        else:
            approx_gt = np.zeros_like(mask)

        # mask = torch.tensor(mask)
        # mask = torch.stack([mask == i for i in range(np.max(mask.astype(np.int8)) + 1)], dim=0).sum(dim=1)

        return {'img': torch.tensor(img, dtype=torch.float32),
                'mask': torch.tensor(mask).type(torch.LongTensor),
                'approx_gt': torch.tensor(approx_gt, dtype=torch.float32),
                'use_gt': torch.tensor(self.use_gt[idx]),
                'dicom': self.df.iloc[idx]['dicom_uuid']
                }


class ESEDDataModule(pl.LightningDataModule):
    """
    DataModule used for semantic segmentation in geometric generalization project
    """

    def __init__(self,
                 data_dir,
                 csv_file,
                 gt_column=None,
                 splits_column='split_0',
                 subset_frac=1.0,
                 test_frac=0.1,
                 gt_frac=None,
                 seed=0,
                 *args, **kwargs):
        super().__init__()
        self.args = args
        self.kwargs = kwargs

        self.save_hyperparameters(logger=False)

        # open dataframe for dataset
        self.df_path = self.hparams.data_dir + '/' + self.hparams.csv_file
        self.df = pd.read_csv(self.df_path, index_col=0)

        self.train: Optional[torch.utils.Dataset] = None
        self.validate: Optional[torch.utils.Dataset] = None
        self.test: Optional[torch.utils.Dataset] = None
        self.pred: Optional[torch.utils.Dataset] = None

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

        self.df = self.df[(self.df['passed'] == True) & (self.df['relative_path'].notna())]

        # Do splits
        if self.hparams.splits_column and self.hparams.splits_column in self.df.columns:
            # splits are already defined in csv file
            print(f"Using split from column: {self.hparams.splits_column}")
            self.train_idx = self.df.index[self.df[self.hparams.splits_column] == 'train'].tolist()
            self.val_idx = self.df.index[self.df[self.hparams.splits_column] == 'val'].tolist()
            self.test_idx = self.df.index[self.df[self.hparams.splits_column] == 'test'].tolist()
            self.pred_idx = self.df.index[self.df[self.hparams.splits_column] == 'pred'].tolist()
        else:
            # create new splits, save if column name is given
            print(f"Creating new splits!")
            self.train_idx, val_and_test_idx = train_test_split(self.df.index.to_list(),
                                                                train_size=0.8,
                                                                random_state=self.hparams.seed)
            self.val_idx, self.test_idx = train_test_split(val_and_test_idx,
                                                           test_size=0.5,
                                                           random_state=self.hparams.seed)
            self.pred_idx = []

            if self.hparams.splits_column:
                print(f"Saving new split to column: {self.hparams.splits_column}")
                self.df.loc[self.train_idx, self.hparams.splits_column] = 'train'
                self.df.loc[self.val_idx, self.hparams.splits_column] = 'val'
                self.df.loc[self.test_idx, self.hparams.splits_column] = 'test'
                self.df.to_csv(self.hparams.data_dir + '/' + self.hparams.csv_file)

        if self.hparams.gt_frac:
            self.hparams.gt_column = 'default_gt'
            self.df[self.hparams.gt_column] = self.df.get(self.hparams.gt_column, False)
            use_gt = np.zeros(len(self.df))
            use_gt[:int(self.hparams.gt_frac * len(self.df))] = 1
            np.random.shuffle(use_gt)
            self.df[self.hparams.gt_column] = use_gt.astype(bool)

        if self.hparams.subset_frac:
            train_num = int(self.hparams.subset_frac * len(self.train_idx))
            self.train_idx = self.train_idx[:train_num]
            val_num = int(self.hparams.subset_frac * len(self.val_idx))
            self.val_idx = self.val_idx[:val_num]
            test_num = int(self.hparams.subset_frac * len(self.test_idx))
            self.test_idx = self.test_idx[:test_num]
            pred_num = int(self.hparams.subset_frac * len(self.pred_idx))
            self.pred_idx = self.pred_idx[-pred_num:]

        if stage == "fit" or stage is None:
            self.train = ESEDDataset(self.df.loc[self.train_idx],
                                       data_path=self.hparams.data_dir,
                                       subset_frac=self.hparams.subset_frac,
                                       seed=self.hparams.seed,
                                       available_gt=self.df.loc[self.train_idx].get(self.hparams.gt_column, None))

            self.validate = ESEDDataset(self.df.loc[self.val_idx],
                                          data_path=self.hparams.data_dir,
                                          subset_frac=self.hparams.subset_frac,
                                          seed=self.hparams.seed,
                                          available_gt=None)

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test = ESEDDataset(self.df.loc[self.test_idx],
                                      data_path=self.hparams.data_dir,
                                      subset_frac=self.hparams.subset_frac,
                                      seed=self.hparams.seed,
                                      available_gt=None,
                                      test=True)
        if stage == "predict":
            self.pred = ESEDDataset(self.df.loc[self.pred_idx],
                                      data_path=self.hparams.data_dir,
                                      subset_frac=self.hparams.subset_frac,
                                      seed=self.hparams.seed,
                                      available_gt=None,
                                      test=True)

    # define your dataloaders
    # again, here defined for train, validate and test, not for predict as the project is not there yet.
    def train_dataloader(self):
        return DataLoader(self.train, batch_size=32, num_workers=16, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.validate, batch_size=32, num_workers=16)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=32, num_workers=16)

    def predict_dataloader(self, ):
        return DataLoader(self.pred, batch_size=32, num_workers=16)


if __name__ == "__main__":
    dl = ESEDDataModule(data_dir='/home/local/USHERBROOKE/juda2901/dev/data/icardio/ES_ED_train_subset/',
                          csv_file='subset.csv',
                          #subset_frac=0.1,
                          test_frac=0.1)

    dl.setup()
    for i in range(1):
        batch = next(iter(dl.train_dataloader()))
        print(batch['img'].shape)
        print(batch['mask'].shape)
        print(batch['approx_gt'].shape)
