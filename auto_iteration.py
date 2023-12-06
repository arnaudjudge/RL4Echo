from pathlib import Path

import hydra
import numpy as np
import torch
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import OmegaConf
from pytorch_lightning import Trainer

from utils.instantiators import instantiate_callbacks
from runner import main as runner_main


@hydra.main(version_base=None, config_path="config", config_name="experiment")
def main(cfg):
    GlobalHydra.instance().clear()
    initialize(version_base=None, config_path='./config')

    iterations = 5
    output_path = f'./auto_iteration2/{0}/'
    Path(output_path).mkdir(parents=True, exist_ok=True)

    # train supervised network for initial actor
    overrides = ['trainer.max_epochs=100',
                 'datamodule.subset_frac=0.01',
                 f'model.predict_save_dir={output_path}',
                 f'model.ckpt_path={output_path}actor.ckpt']
    sub_cfg = compose(config_name=f"supervised_runner.yaml", overrides=overrides)
    print(OmegaConf.to_yaml(sub_cfg))
    runner_main(sub_cfg)

    for i in range(1, iterations+1):
        # train reward net
        overrides = ['trainer.max_epochs=100',
                     f'datamodule.data_path={output_path}',
                     f'model.save_model_path={output_path}rewardnet.ckpt']
        sub_cfg = compose(config_name=f"reward_runner.yaml", overrides=overrides)
        print(OmegaConf.to_yaml(sub_cfg))
        runner_main(sub_cfg)

        next_output_path = f'./auto_iteration2/{i}/'
        Path(next_output_path).mkdir(parents=True, exist_ok=True)

        # train PPO model with fresh reward net
        overrides = [f"trainer.max_epochs=15",
                     f"model.actor.actor_pretrain_ckpt={output_path}actor.ckpt",
                     f"model.reward.state_dict_path={output_path}rewardnet.ckpt",
                     f"model.actor_save_path={next_output_path}actor.ckpt",
                     f"model.critic_save_path={next_output_path}critic.ckpt",
                     f'model.predict_save_dir={next_output_path}',
                     ]
        if Path(f"{output_path}critic.ckpt").exists():
            overrides += [f"model.actor.critic_pretrain_ckpt={output_path}critic.ckpt"]
        sub_cfg = compose(config_name=f"RL_runner.yaml", overrides=overrides)
        print(OmegaConf.to_yaml(sub_cfg))
        runner_main(sub_cfg)

        output_path = next_output_path


if __name__ == '__main__':
    main()
