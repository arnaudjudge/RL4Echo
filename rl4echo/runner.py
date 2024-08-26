import os
from dotenv import load_dotenv
import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf
from pytorch_lightning import Trainer, seed_everything

from utils.instantiators import instantiate_callbacks

OmegaConf.register_new_resolver(
    "get_class_name", lambda name: name.split('.')[-1]
)


@hydra.main(version_base=None, config_path="config", config_name="supervised_3d_runner")
def main(cfg):
    # Load any available `.env` file
    load_dotenv()

    os.environ["HYDRA_FULL_ERROR"] = "1"
    print(OmegaConf.to_yaml(cfg))

    seed_everything(cfg.seed)

    logger = instantiate(cfg.logger)

    model = instantiate(cfg.model)
    datamodule = instantiate(cfg.datamodule, seed=cfg.seed)

    callbacks = instantiate_callbacks(cfg.callbacks)

    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    if cfg.train:
        trainer.fit(train_dataloaders=datamodule, model=model, ckpt_path='/home/local/USHERBROOKE/juda2901/dev/RL4Echo/rl4echo/test_logs/SupervisedOptimizer_supervised_default/version_63/checkpoints/epoch=8-step=4167.ckpt')

    if cfg.trainer.max_epochs > 0 and cfg.train:
        ckpt_path = 'best'
    else:
        ckpt_path = None

    # test with everything
    datamodule.hparams.subset_frac = 1.0
    trainer.test(model=model, dataloaders=datamodule, ckpt_path=ckpt_path)

    if getattr(cfg.model, "predict_save_dir", None):
        datamodule.hparams.subset_frac = cfg.predict_subset_frac
        datamodule.setup(stage="predict")
        trainer.predict(model=model, dataloaders=datamodule, ckpt_path=ckpt_path)


if __name__ == "__main__":
    load_dotenv()
    main()
