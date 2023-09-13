import os

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from pytorch_lightning import Trainer

from utils.instantiators import instantiate_callbacks

OmegaConf.register_new_resolver(
    "get_class_name", lambda name: name.split('.')[-1]
)


@hydra.main(version_base=None, config_path="config", config_name="RL_runner")
def main(cfg):
    os.environ["HYDRA_FULL_ERROR"] = "1"
    print(OmegaConf.to_yaml(cfg))

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    logger = instantiate(cfg.logger)

    model = instantiate(cfg.model)
    datamodule = instantiate(cfg.datamodule, seed=cfg.seed)

    callbacks = instantiate_callbacks(cfg.callbacks)

    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    trainer.fit(train_dataloaders=datamodule, model=model)
    if cfg.trainer.max_epochs > 0:
        ckpt_path = 'best'
    else:
        ckpt_path = None
    trainer.test(model=model, dataloaders=datamodule, ckpt_path=ckpt_path)

    if cfg.run_predict:
        trainer.predict(model=model, dataloaders=datamodule.test_dataloader(), ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
