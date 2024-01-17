from datetime import datetime
from pathlib import Path

import hydra
import pandas as pd
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from runner import main as runner_main


@hydra.main(version_base=None, config_path="config", config_name="experiment")
def main(cfg):
    GlobalHydra.instance().clear()
    initialize(version_base=None, config_path='./config')

    iterations = 5
    output_path = f'./logs/auto_iteration_cardiac_NEW_METRICS'
    Path(output_path+"/0/").mkdir(parents=True, exist_ok=True)
    main_overrides = [f"logger.save_dir={output_path}"]
    experiment = 'es_ed'

    timestamp = '1705503605.136425' #datetime.now().timestamp()
    experiment_split_column = f"split_{timestamp}"
    experiment_gt_column = f"Gt_{timestamp}"

    # train supervised network for initial actor
    overrides = main_overrides + ['trainer.max_epochs=100',
                                  'datamodule.subset_frac=0.012',
                                  'predict_subset_frac=1000',
                                  f'model.predict_save_dir={output_path}',
                                  f'model.ckpt_path={output_path}/{0}/actor.ckpt',
                                  f'experiment=supervised_{experiment}']
    sub_cfg = compose(config_name=f"supervised_runner.yaml", overrides=overrides)
    print(OmegaConf.to_yaml(sub_cfg))

    # prepare dataset with custom split column
    df = pd.read_csv(sub_cfg.datamodule.data_dir + sub_cfg.datamodule.csv_file, index_col=0)
    df[experiment_split_column] = df.loc[:, sub_cfg.datamodule.splits_column]
    df[experiment_gt_column] = df.loc[:, sub_cfg.datamodule.gt_column]
    df.to_csv(sub_cfg.datamodule.data_dir + sub_cfg.datamodule.csv_file)
    sub_cfg.datamodule.splits_column = experiment_split_column
    sub_cfg.datamodule.gt_column = experiment_gt_column

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
        overrides = main_overrides + [f"trainer.max_epochs=5",
                                      f"datamodule.splits_column={experiment_split_column}",
                                      f"datamodule.gt_column={experiment_gt_column}",
                                      f"model.actor.actor.pretrain_ckpt={output_path}/{i-1}/actor.ckpt",
                                      f"model.reward.state_dict_path={output_path}/{i-1}/rewardnet.ckpt",
                                      f"model.actor_save_path={output_path}/{i}/actor.ckpt",
                                      f"model.critic_save_path={output_path}/{i}/critic.ckpt",
                                      f'model.predict_save_dir={output_path}',
                                      f"model.entropy_coeff=0.05",
                                      f'experiment=ppo_{experiment}'
                                      ]
        if Path(f"{output_path}/{i-1}/critic.ckpt").exists():
            overrides += [f"model.actor.critic.pretrain_ckpt={output_path}/{i-1}/critic.ckpt"]
        sub_cfg = compose(config_name=f"RL_runner.yaml", overrides=overrides)
        print(OmegaConf.to_yaml(sub_cfg))
        runner_main(sub_cfg)


if __name__ == '__main__':
    main()
