from datamodule import SectorDataModule
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from supervised import SupervisedOptimizer


if __name__ == '__main__':

    logger = TensorBoardLogger('../logs', name='supervised_unet')

    dl = SectorDataModule('/home/local/USHERBROOKE/juda2901/dev/data/icardio/train_subset/',
                          '/home/local/USHERBROOKE/juda2901/dev/data/icardio/train_subset/subset.csv')
    trainer = pl.Trainer(max_epochs=15, logger=logger, log_every_n_steps=1, gpus=1)

    model = SupervisedOptimizer()

    trainer.fit(train_dataloaders=dl, model=model)
    model.save()
    # trainer.test(model=model, dataloaders=dl)

