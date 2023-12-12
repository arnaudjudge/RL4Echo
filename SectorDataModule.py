import json
from typing import Optional

import nibabel as nib
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torch.utils.data import random_split, DataLoader


class SectorDataset(Dataset):
    def __init__(self, df, data_path, subset_frac=1.0, available_gt=1.0, seed=0, test=False, *args,
                 **kwargs):
        super().__init__()
        self.df = df
        self.data_path = data_path
        self.test = test

        # only use subset
        self.df = self.df.sample(n=int(subset_frac * len(self.df)), random_state=seed)

        print(f"Test step: {self.test} , len of dataset {len(self.df)}")

        # create list of available ground truths
        self.use_gt = np.zeros(len(self.df))
        self.use_gt[:int(available_gt * len(self.df))] = 1
        np.random.shuffle(self.use_gt)
        print(f"Number of ground truths available: {self.use_gt.sum()}")

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, idx):
        path_dict = json.loads(self.df.iloc[idx]['relative_path'].replace("\'", "\""))

        img = np.expand_dims(nib.load(self.data_path + '/raw/' + path_dict['raw']).get_fdata().mean(axis=2), 0)
        mask = nib.load(self.data_path + '/mask/' + path_dict['mask']).get_fdata()[:, :, 0]

        return torch.tensor(img, dtype=torch.float32), \
               torch.tensor(mask, dtype=torch.float32), \
               torch.tensor(self.use_gt[idx], dtype=torch.bool)


class SectorDataModule(pl.LightningDataModule):
    """
    DataModule used for semantic segmentation in geometric generalization project
    """

    def __init__(self, data_dir, csv_file, splits_column='split_0', subset_frac=1.0, test_frac=0.1, available_gt=1.0, seed=0, *args, **kwargs):
        super().__init__()
        self.args = args
        self.kwargs = kwargs

        self.save_hyperparameters(logger=False)

        # open dataframe for dataset
        self.df = pd.read_csv(self.hparams.data_dir + '/' + self.hparams.csv_file, index_col=0)

        self.train: Optional[torch.utils.Dataset] = None
        self.validate: Optional[torch.utils.Dataset] = None
        self.test: Optional[torch.utils.Dataset] = None
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
        else:
            # create new splits, save if column name is given
            print(f"Creating new splits!")
            self.train_idx, val_and_test_idx = train_test_split(self.df.index.to_list(),
                                                                train_size=0.8,
                                                                random_state=self.hparams.seed)
            self.val_idx, self.test_idx = train_test_split(val_and_test_idx,
                                                           test_size=0.5,
                                                           random_state=self.hparams.seed)
            if self.hparams.splits_column:
                print(f"Saving new split to column: {self.hparams.splits_column}")
                self.df.loc[self.train_idx, self.hparams.splits_column] = 'train'
                self.df.loc[self.val_idx, self.hparams.splits_column] = 'val'
                self.df.loc[self.test_idx, self.hparams.splits_column] = 'test'
                self.df.to_csv(self.hparams.data_dir + '/' + self.hparams.csv_file)

        if stage == "fit" or stage is None:
            self.train = SectorDataset(self.df.loc[self.train_idx],
                                           data_path=self.hparams.data_dir,
                                           subset_frac=self.hparams.subset_frac,
                                           seed=self.hparams.seed)

            self.validate = SectorDataset(self.df.loc[self.val_idx],
                                          data_path=self.hparams.data_dir,
                                          subset_frac=self.hparams.subset_frac,
                                          seed=self.hparams.seed)

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test = SectorDataset(self.df.loc[self.val_idx],
                                          data_path=self.hparams.data_dir,
                                          subset_frac=self.hparams.subset_frac,
                                          seed=self.hparams.seed,
                                          test=True)

    # define your dataloaders
    # again, here defined for train, validate and test, not for predict as the project is not there yet.
    def train_dataloader(self):
        return DataLoader(self.train, batch_size=32, num_workers=16)

    def val_dataloader(self):
        return DataLoader(self.validate, batch_size=32, num_workers=16)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=32, num_workers=16)


if __name__ == "__main__":
    dl = SectorDataModule(data_dir='/home/local/USHERBROOKE/juda2901/dev/data/icardio/train_subset_10k/',
                          csv_file='subset.csv',
                          subset_frac=0.1,
                          test_frac=0.1)

    dl.setup()
    for i in range(1):
        batch = next(iter(dl.train_dataloader()))
        print(batch[0].shape)
        print(batch[1].shape)
        print(batch[2].shape)
