import json
from typing import Optional

import nibabel as nib
import numpy as np
import pandas as pd
from lightning import LightningDataModule
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from rl4echo.utils.file_utils import get_img_subpath


class SectorDataset(Dataset):
    def __init__(self, df, data_path, approx_gt_path=None, subset_frac=1.0, available_gt=None, seed=0, test=False, *args,
                 **kwargs):
        super().__init__()
        self.df = df
        self.data_path = data_path
        self.approx_gt_path = approx_gt_path
        self.test = test

        print(f"Test step: {self.test} , len of dataset {len(self.df)}")

        # pass down list of available ground truths
        self.use_gt = available_gt.fillna(False).to_numpy() if available_gt is not None else \
            np.asarray([False for _ in range(len(self.df))])
        print(f"Number of ground truths available: {self.use_gt.sum()}")

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, idx):
        path_dict = json.loads(self.df.iloc[idx]['relative_path'].replace("\'", "\""))

        img = np.expand_dims(nib.load(self.data_path + '/raw/' + path_dict['raw']).get_fdata().mean(axis=2), 0)
        mask = nib.load(self.data_path + '/mask/' + path_dict['mask']).get_fdata()[:, :, 0]

        if self.use_gt[idx]:
            approx_gt_path = self.approx_gt_path + '/approx_gt/' + path_dict['mask'].replace("_mask", "")
            approx_gt = nib.load(approx_gt_path).get_fdata()[0, :, :]
        else:
            approx_gt = np.zeros_like(mask)

        return {'img': torch.tensor(img, dtype=torch.float32),
                'gt': torch.tensor(mask, dtype=torch.float32),
                'approx_gt': torch.tensor(approx_gt, dtype=torch.float32),
                'use_gt': torch.tensor(self.use_gt[idx], dtype=torch.bool),
                'id': self.df.iloc[idx]['dicom_uuid']
                }


class SectorDataModule(LightningDataModule):
    """
    DataModule used for semantic segmentation in geometric generalization project
    """

    def __init__(self,
                 data_dir,
                 csv_file,
                 approx_gt_dir=None,
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

        if self.hparams.subset_frac and type(self.hparams.subset_frac) == float:
            train_num = int(self.hparams.subset_frac * len(self.train_idx))
            self.train_idx = self.train_idx[:train_num]
            val_num = int(self.hparams.subset_frac * len(self.val_idx))
            self.val_idx = self.val_idx[:val_num]
            test_num = int(self.hparams.subset_frac * len(self.test_idx))
            self.test_idx = self.test_idx[:test_num]
            pred_num = int(self.hparams.subset_frac * len(self.pred_idx))
            self.pred_idx = self.pred_idx[-pred_num:]
        elif self.hparams.subset_frac and type(self.hparams.subset_frac) == int:
            self.train_idx = self.train_idx[:min(self.hparams.subset_frac, len(self.train_idx))]
            self.val_idx = self.val_idx[:min(self.hparams.subset_frac, len(self.val_idx))]
            self.test_idx = self.test_idx[:min(self.hparams.subset_frac, len(self.test_idx))]
            self.pred_idx = self.pred_idx[:min(self.hparams.subset_frac, len(self.pred_idx))]

        if stage == "fit" or stage is None:
            self.train = SectorDataset(self.df.loc[self.train_idx],
                                       data_path=self.hparams.data_dir,
                                       subset_frac=self.hparams.subset_frac,
                                       seed=self.hparams.seed,
                                       available_gt=self.df.loc[self.train_idx].get(self.hparams.gt_column, None),
                                       approx_gt_path=self.hparams.approx_gt_dir,
                                       )

            self.validate = SectorDataset(self.df.loc[self.val_idx],
                                          data_path=self.hparams.data_dir,
                                          subset_frac=self.hparams.subset_frac,
                                          seed=self.hparams.seed,
                                          available_gt=None)

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test = SectorDataset(self.df.loc[self.test_idx],
                                      data_path=self.hparams.data_dir,
                                      subset_frac=self.hparams.subset_frac,
                                      seed=self.hparams.seed,
                                      available_gt=None,
                                      test=True)
        if stage == "predict":
            self.pred = SectorDataset(self.df.loc[self.pred_idx],
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

    def add_to_train(self, id, instant):
        self.df.loc[(self.df['id'] == id), self.trainer.datamodule.hparams.splits_column] = 'train'

    def add_to_gt(self, id, instant):
        self.df.loc[(self.df['id'] == id), self.trainer.datamodule.hparams.gt_column] = True

    def update_dataframe(self):
        self.df.to_csv(self.df_path)

    def get_approx_gt_subpath(self, id, instant):
        return get_img_subpath(self.df.loc[self.df['dicom_uuid'] == id.iloc[0]])


if __name__ == "__main__":
    dl = SectorDataModule(data_dir='/home/local/USHERBROOKE/juda2901/dev/data/icardio/train_subset_10k/',
                          csv_file='subset.csv',
                          subset_frac=0.1,
                          test_frac=0.1)

    dl.setup()
    for i in range(1):
        batch = next(iter(dl.train_dataloader()))
        print(batch['img'].shape)
        print(batch['gt'].shape)
        print(batch['approx_gt'].shape)
