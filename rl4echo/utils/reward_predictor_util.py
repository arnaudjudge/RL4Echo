from pathlib import Path

import hydra
import numpy as np
import torch
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from lightning import Trainer, LightningModule
from matplotlib import animation
from omegaconf import OmegaConf
import torchio as tio
import nibabel as nib


class RL4EchoPredictor:
    def __init__(self, config_name,
                 anatomical_path='/home/local/USHERBROOKE/juda2901/dev/RL4Echo/rewardnet_5000best.ckpt',
                 common_spacing=(0.37, 0.37, 1),
                 shape_divisible_by=(32, 32, 4),
                 ):
        self.common_spacing = common_spacing
        self.shape_divisible_by = shape_divisible_by

        GlobalHydra.instance().clear()
        initialize(version_base="1.2", config_path='../config/', job_name="model")
        cfg = compose(config_name=f"{config_name}", overrides=["++trainer.max_epochs=1",
                                                               f"reward.state_dict_paths.anatomical={anatomical_path}"])
        print(OmegaConf.to_yaml(cfg))

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Running on device: {self.device}")

        self.trainer = Trainer(
            max_epochs=2,
            accelerator="gpu",
            devices=1,
            enable_progress_bar=False,
            enable_checkpointing=False,
            logger=False
        )

        self.model: LightningModule = hydra.utils.instantiate(cfg.reward)

    def get_desired_size(self, current_shape):
        # get desired closest divisible bigger shape
        x = int(np.ceil(current_shape[0] / self.shape_divisible_by[0]) * self.shape_divisible_by[0])
        y = int(np.ceil(current_shape[1] / self.shape_divisible_by[1]) * self.shape_divisible_by[1])
        z = current_shape[2]
        return x, y, z

    def predict_from_numpy(self, mask, img, affine):

        self.model.prepare_for_full_sequence()

        transform = tio.Resample(self.common_spacing)
        resampled = transform(tio.ScalarImage(tensor=np.expand_dims(img, 0), affine=affine))

        croporpad = tio.CropOrPad(self.get_desired_size(resampled.shape[1:]))
        resampled_cropped = croporpad(resampled)
        img = resampled_cropped.tensor.type(torch.float32)
        mask = croporpad(transform(tio.LabelMap(tensor=np.expand_dims(mask, 0),
                                                affine=affine))).tensor.type(torch.float32)

        return self.model.predict_full_sequence(mask, img, None)


if __name__ == "__main__":

    predictor = RL4EchoPredictor(config_name='reward/rewardunets_3d.yaml')

    p = Path("/data/icardio/subsets/5000_w_test/img/st-7878-DA71-710F/a2c/di-3AA8-D767-483C_0000.nii.gz")
    img_nifti = nib.load(p)
    img = img_nifti.get_fdata()
    seg = nib.load(p.as_posix().replace("img", "segmentation").replace("_0000", "")).get_fdata()
    print(img.shape)
    print(seg.shape)

    rew = predictor.predict_from_numpy(seg, img, img_nifti.affine)
    print(rew[0].shape)

    rew = rew[0].cpu().numpy().squeeze(0).transpose((2, 1, 0))

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 1)
    im = axes.imshow(rew[0, ...], animated=True, cmap='gray')

    def update(i):
        im.set_array(rew[i, ...])
        return im,


    animation_fig = animation.FuncAnimation(fig, update, frames=rew.shape[0], interval=100, blit=True,
                                            repeat_delay=10, )
    plt.show()

