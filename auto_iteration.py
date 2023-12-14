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
    output_path = f'./auto_iteration_entropy'
    Path(output_path+"/0/").mkdir(parents=True, exist_ok=True)
    main_overrides = [f"logger.save_dir={output_path}"]

    # train supervised network for initial actor
    overrides = main_overrides + ['trainer.max_epochs=100',
                                  'datamodule.subset_frac=0.005',
                                  'predict_subset_frac=0.1',
                                  f'model.predict_save_dir={output_path}',
                                  f'model.ckpt_path={output_path}/{0}/actor.ckpt']
    sub_cfg = compose(config_name=f"supervised_runner.yaml", overrides=overrides)
    print(OmegaConf.to_yaml(sub_cfg))
    runner_main(sub_cfg)

    for i in range(1, iterations+1):
        # train reward net
        overrides = main_overrides + ['trainer.max_epochs=50',
                                      f'datamodule.data_path={output_path}/',
                                      f'model.save_model_path={output_path}/{i-1}/rewardnet.ckpt']
        sub_cfg = compose(config_name=f"reward_runner.yaml", overrides=overrides)
        print(OmegaConf.to_yaml(sub_cfg))
        runner_main(sub_cfg)

        next_output_path = f'{output_path}/{i}/'
        Path(next_output_path).mkdir(parents=True, exist_ok=True)

        # train PPO model with fresh reward net
        overrides = main_overrides + [f"trainer.max_epochs=10",
                                      f"model.actor.actor_pretrain_ckpt={output_path}/{i-1}/actor.ckpt",
                                      f"model.reward.state_dict_path={output_path}/{i-1}/rewardnet.ckpt",
                                      f"model.actor_save_path={output_path}/{i}/actor.ckpt",
                                      f"model.critic_save_path={output_path}/{i}/critic.ckpt",
                                      f'model.predict_save_dir={output_path}',
                                      f"model.entropy_coeff=0.05"
                                      ]
        if Path(f"{output_path}/{i-1}/critic.ckpt").exists():
            overrides += [f"model.actor.critic_pretrain_ckpt={output_path}/{i-1}/actor.ckpt"]
        sub_cfg = compose(config_name=f"RL_runner.yaml", overrides=overrides)
        print(OmegaConf.to_yaml(sub_cfg))
        runner_main(sub_cfg)


if __name__ == '__main__':
    main()
