import os
from dotenv import load_dotenv
import hydra
import pytorch_lightning
from hydra.utils import instantiate
from omegaconf import OmegaConf
from pytorch_lightning import Trainer

from utils.instantiators import instantiate_callbacks

OmegaConf.register_new_resolver(
    "get_class_name", lambda name: name.split('.')[-1]
)


@hydra.main(version_base=None, config_path="config", config_name="eval")
def main(cfg):
    # Load any available `.env` file
    load_dotenv()

    os.environ["HYDRA_FULL_ERROR"] = "1"
    print(OmegaConf.to_yaml(cfg))

    pytorch_lightning.seed_everything(cfg.seed)

    logger = instantiate(cfg.logger)

    model = instantiate(cfg.model)
    datamodule = instantiate(cfg.datamodule, seed=cfg.seed)

    callbacks = instantiate_callbacks(cfg.callbacks)

    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    #ckpt_path = './logs/auto_iteration_es_ed_test_camus/PPO_RewardUnet_es_ed/version_3/checkpoints/epoch=2-step=3960.ckpt'
    #ckpt_path = './logs/auto_iteration_es_ed_test_camus/SupervisedOptimizer_es_ed/version_1/checkpoints/epoch=48-step=49.ckpt'

    #ckpt_path = './logs/es_ed_big/PPO_RewardUnet_es_ed/version_3/checkpoints/epoch=1-step=5140.ckpt'
    #ckpt_path = './logs/es_ed_big/PPO_RewardUnet_es_ed/version_4/checkpoints/epoch=2-step=9030.ckpt'

    #ckpt_path = './logs/camus_start_es_ed_ppo/PPO_RewardUnet_es_ed/version_3/checkpoints/epoch=1-step=3880.ckpt'

    #ckpt_path = './logs/camus_matched/PPO_RewardUnet_es_ed/version_3/checkpoints/epoch=4-step=9700.ckpt'

    # test with everything
    datamodule.hparams.subset_frac = 1.0
    trainer.test(model=model, dataloaders=datamodule, ckpt_path=cfg.ckpt_path)


if __name__ == "__main__":
    main()
