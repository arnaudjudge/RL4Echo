import json
from typing import Optional

import nibabel as nib
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset
from torch.utils.data import random_split, DataLoader


class SectorDataset(Dataset):
    def __init__(self, data_path, csv_file, subset_frac=1.0, test_frac=0.1, seed=0, test=False, *args, **kwargs):
        super().__init__()
        self.data_path = data_path

        self.df = pd.read_csv(csv_file, index_col=0)
        self.df = self.df[(self.df['passed'] == True) & (self.df['relative_path'].notna())]

        # only use subset
        self.df = self.df.sample(n=int(subset_frac*len(self.df)), random_state=seed)

        # split according to test_frac
        test_len = int(test_frac * len(self.df))
        if test:
            self.df = self.df.iloc[-test_len:]
        else:
            self.df = self.df.iloc[:-test_len]

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, idx):
        path_dict = json.loads(self.df.iloc[idx]['relative_path'].replace("\'", "\""))

        img = np.expand_dims(nib.load(self.data_path + '/raw/' + path_dict['raw']).get_fdata().mean(axis=2), 0)
        mask = nib.load(self.data_path + '/mask/' + path_dict['mask']).get_fdata()[:, :, 0]

        return torch.tensor(img, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)


class SectorDataModule(pl.LightningDataModule):
    """
    DataModule used for semantic segmentation in geometric generalization project
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.args = args
        self.kwargs = kwargs

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
            train_set_full = SectorDataset(*self.args, **self.kwargs)
            train_set_size = int(len(train_set_full) * 0.9)
            valid_set_size = len(train_set_full) - train_set_size
            self.train, self.validate = random_split(train_set_full, [train_set_size, valid_set_size])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test = SectorDataset(*self.args, **self.kwargs, test=True)

    # define your dataloaders
    # again, here defined for train, validate and test, not for predict as the project is not there yet.
    def train_dataloader(self):
        return DataLoader(self.train, batch_size=32, num_workers=16)

    def val_dataloader(self):
        return DataLoader(self.validate, batch_size=32, num_workers=16)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=32, num_workers=16)


if __name__ == "__main__":
    dl = SectorDataModule('/home/local/USHERBROOKE/juda2901/dev/data/icardio/train_subset/',
                          '/home/local/USHERBROOKE/juda2901/dev/data/icardio/train_subset/subset.csv',
                          0.1,
                          0.1)

    dl.setup()
    for i in range(1):
        batch = next(iter(dl.train_dataloader()))
        print(batch[0].shape)
        print(batch[1].shape)
