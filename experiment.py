import hydra
import numpy as np
import torch
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import OmegaConf
from pytorch_lightning import Trainer

from utils.instantiators import instantiate_callbacks


@hydra.main(version_base=None, config_path="config", config_name="experiment")
def main(cfg):
    print(OmegaConf.to_yaml(cfg))

    # import needs to be after hydra decorator
    from runner import main

    # create global overrides
    glob_overrides = []
    if cfg.global_overrides:
        for k, v in cfg.global_overrides.items():
            glob_overrides += [f"{k}={v}"]

    # create overrides for experiments
    for k, v in cfg.exp_overrides.items():
        for value in v:
            overrides = [f"{k}={value}"]
            sub_cfg = compose(config_name=f"{cfg.runner}.yaml", overrides=glob_overrides+overrides)
            print(OmegaConf.to_yaml(sub_cfg))

            main(sub_cfg)


if __name__ == '__main__':
    main()
