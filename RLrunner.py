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


@hydra.main(version_base=None, config_path="config", config_name="RL_runner")
def main(cfg):
    print(OmegaConf.to_yaml(cfg))
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    logger = instantiate(cfg.logger)

    reward = instantiate(cfg.reward, _partial_=True)
    model: RLmodule = instantiate(cfg.model, reward=reward)
    datamodule = instantiate(cfg.datamodule, seed=cfg.seed)

    callbacks = instantiate_callbacks(cfg.callbacks)

    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    trainer.fit(train_dataloaders=datamodule, model=model)#, ckpt_path='/home/local/USHERBROOKE/juda2901/dev/RL4Echo/runs/PPO_morphological_v_func_unet/version_19/checkpoints/epoch=2-step=780.ckpt')#, ckpt_path='/home/local/USHERBROOKE/juda2901/dev/RL4Echo/runs/PPO_morphological_v_func_unet/version_1/checkpoints/epoch=2-step=780.ckpt')#, ckpt_path='/home/local/USHERBROOKE/juda2901/dev/RL4Echo/logs/PPO/version_179/checkpoints/epoch=6-step=1820.ckpt')
    trainer.test(model=model, dataloaders=datamodule, ckpt_path="best") # /home/local/USHERBROOKE/juda2901/dev/RL4Echo/runs/PPO_morphological_v_func_unet/version_96/checkpoints/epoch=3-step=1040.ckpt #/home/local/USHERBROOKE/juda2901/dev/RL4Echo/runs/PPO_morphological_v_func_unet/version_29/checkpoints/epoch=4-step=1300.ckpt


if __name__ == "__main__":
    main()


