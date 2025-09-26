from pathlib import Path

import h5py
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
from tqdm import tqdm


class RL4EchoPredictor:
    def __init__(self, config_name,
                 anatomical_path='/home/local/USHERBROOKE/juda2901/dev/RL4Echo/narval_lm+anat-run_RNet1.ckpt',
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
        for rnet in self.model.get_nets():
            rnet.to(self.device)

        self.model.prepare_for_full_sequence()

        # transform = tio.Resample(self.common_spacing)
        # resampled = transform(tio.ScalarImage(tensor=np.expand_dims(img, 0), affine=affine))

        # croporpad = tio.CropOrPad(self.get_desired_size(resampled.shape[1:]))
        # resampled_cropped = croporpad(resampled)
        # img = resampled_cropped.tensor.type(torch.float32)
        # mask = croporpad(transform(tio.LabelMap(tensor=np.expand_dims(mask, 0),
        #                                         affine=affine))).tensor.type(torch.float32)
        img = torch.tensor(img).type(torch.float32).unsqueeze(0)
        mask = torch.tensor(mask).type(torch.float32).unsqueeze(0)
        return self.model.predict_full_sequence(mask.to(self.device), img.to(self.device), None)[0]


if __name__ == "__main__":

    predictor = RL4EchoPredictor(config_name='reward/rewardunets_3d.yaml')

    # p = Path("/data/icardio/subsets/full_3DRL_subset_norm_TESTONLY/img/st-2997-0558-7950/a4c/di-1878-4DAB-4564_0000.nii.gz")
    # img_nifti = nib.load(p)
    # img = img_nifti.get_fdata()
    # seg = nib.load(p.as_posix().replace("img", "segmentation").replace("_0000", "")).get_fdata()
    # print(img.shape)
    # print(seg.shape)
    #
    # rew = predictor.predict_from_numpy(seg, img, img_nifti.affine)
    # print(rew[0].shape)
    #
    # rew = rew[0].cpu().numpy().squeeze(0).transpose((2, 1, 0))
    #
    # import matplotlib.pyplot as plt
    # fig, axes = plt.subplots(1, 1)
    # im = axes.imshow(rew[0, ...], animated=True, cmap='gray')
    #
    # def update(i):
    #     im.set_array(rew[i, ...])
    #     return im,
    #
    #
    # animation_fig = animation.FuncAnimation(fig, update, frames=rew.shape[0], interval=100, blit=True,
    #                                         repeat_delay=10, )
    # plt.show()
    h5_ = h5py.File("./../../3dUNC_MCDropout_baseline3dunet.h5", "r")
    #./../../3d_anatomical_reward_LM+ANAT_RNET1_TEMPSCALED_NEW_NARVAL_POLICY_CARDINAL.h5
    with h5py.File("./../../3dUNC_Ensemble_baseline3dunet.h5", "a") as h5:

        for k in tqdm(h5.keys(), total=len(h5.keys())):
            imgs = np.array(h5_[k]['img']) / 255
            preds = np.array(h5[k]['pred'])
            gts = np.array(h5[k]['gt'])

            rew = predictor.predict_from_numpy(preds, imgs, np.diag([0.37, 0.37, 1, 0]))
            print(rew.shape)

            rew = rew.cpu().numpy().squeeze(0)
            assert rew.shape == preds.shape
            del h5[k]['reward_map+LM']
            h5[k]['reward_map+LM'] = rew

            # imgs = imgs.transpose((2, 1, 0))
            # import matplotlib.pyplot as plt
            # fig, axes = plt.subplots(1, 1)
            # im = axes.imshow(imgs[0, ...], animated=True, cmap='gray')
            #
            # def update(i):
            #     im.set_array(imgs[i, ...])
            #     return im,
            #
            #
            # animation_fig = animation.FuncAnimation(fig, update, frames=imgs.shape[0], interval=100, blit=True,
            #                                         repeat_delay=10, )
            #
            # preds = preds.transpose((2, 1, 0))
            # import matplotlib.pyplot as plt
            # fig2, axes2 = plt.subplots(1, 1)
            # im2 = axes2.imshow(preds[0, ...], animated=True, cmap='gray')
            #
            # def update(i):
            #     im2.set_array(preds[i, ...])
            #     return im2,
            #
            #
            # animation_fig2 = animation.FuncAnimation(fig, update, frames=preds.shape[0], interval=100, blit=True,
            #                                         repeat_delay=10, )
            #
            # rew = rew.transpose((2, 1, 0))
            # import matplotlib.pyplot as plt
            # fig3, axes3 = plt.subplots(1, 1)
            # im3 = axes3.imshow(rew[0, ...], animated=True, cmap='gray')
            #
            # def update(i):
            #     im3.set_array(rew[i, ...])
            #     return im3,
            #
            #
            # animation_fig3 = animation.FuncAnimation(fig, update, frames=rew.shape[0], interval=100, blit=True,
            #                                         repeat_delay=10, )
            # plt.show()





