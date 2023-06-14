import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from reward_net_datamodule import RewardNetDataModule
from reward_net import RewardOptimizer


if __name__ == '__main__':

    logger = TensorBoardLogger('../logs', name='reward_network_auto')

    #dl =  RewardNetDataModule('../data/', './data/20230518_reward_net_dataset_echoqcpy.json')
    dl = RewardNetDataModule('./dataset_augmented_reward', './dataset_augmented_reward/labels.json')

    lr_monitor = LearningRateMonitor(logging_interval='step')
    model_ckpt = ModelCheckpoint(dirpath=logger.log_dir, filename='best.ckpt', monitor='val_acc')
    trainer = pl.Trainer(max_epochs=10, logger=logger, log_every_n_steps=1, gpus=1, callbacks=[lr_monitor])

    model = RewardOptimizer()

    trainer.fit(train_dataloaders=dl, model=model)

    trainer.test(model=model, dataloaders=dl)

    model.save_model(path='reward_model_state_dict_autodataset_50k.ckpt')