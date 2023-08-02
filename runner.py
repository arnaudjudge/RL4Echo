import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from pytorch_lightning import Trainer

from RLmodule import RLmodule
from utils.instantiators import instantiate_callbacks

OmegaConf.register_new_resolver(
    "get_class_name", lambda name: name.split('.')[-1]
)


@hydra.main(version_base=None, config_path="config", config_name="runner")
def main(cfg):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    logger = instantiate(cfg.logger)

    reward = instantiate(cfg.reward, _partial_=True)
    model: RLmodule = instantiate(cfg.model, reward=reward)
    datamodule = instantiate(cfg.datamodule, seed=cfg.seed)

    callbacks = instantiate_callbacks(cfg.callbacks)

    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    trainer.fit(train_dataloaders=datamodule, model=model)
    trainer.test(model=model, dataloaders=datamodule, ckpt_path="best")


if __name__ == "__main__":
    main()


