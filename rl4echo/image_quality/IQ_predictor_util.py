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

from vital.utils.image.transform import resize_image


class IQPredictor:
    def __init__(self,
                 config_name,
                 ckpt_path="/home/local/USHERBROOKE/juda2901/dev/RL4Echo/ordinal_IQ.ckpt",
                 common_size=(256, 256, 32),
                 ):
        self.common_size = common_size

        GlobalHydra.instance().clear()
        initialize(version_base="1.2", config_path='../config/', job_name="model")
        cfg = compose(config_name=f"{config_name}", overrides=["++trainer.max_epochs=1",
                                                               f"model.model_weights={ckpt_path}"])
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

        self.model: LightningModule = hydra.utils.instantiate(cfg.model)
        self.model.eval()

    def predict_from_numpy(self, img):
        if int(img.max()) > 1:
            img = img / 255

        img = img.transpose((2, 0, 1))

        img = resize_image(img, self.common_size[:2])

        if len(img) > 60:
            img = img[:60]

        frames = np.round(np.linspace(0, len(img) - 1, self.common_size[-1])).astype(int)
        img1 = img[frames]
        img = torch.tensor(img1).unsqueeze(0).unsqueeze(0).type(torch.float32).to(self.device)

        pred = self.model(img)
        softmax = torch.softmax(pred, dim=1).cpu().detach().numpy()[0]
        # print(softmax)
        expected_value = np.sum(np.arange(len(softmax)) * softmax)
        # print(expected_value)
        return pred.argmax(dim=1), expected_value, img1


if __name__ == "__main__":

    predictor = IQPredictor(config_name='model/image_quality_3d.yaml')

    # paths = [Path("/data/icardio/subsets/5000_w_test/img/st-7878-DA71-710F/a2c/di-3AA8-D767-483C_0000.nii.gz")]
    paths = [p for p in Path("/data/icardio/processed/img/").rglob("*.nii.gz")][:100]

    # test_idx = [134, 268, 48, 291, 530, 494, 192, 198, 596, 57, 270, 512, 71, 230, 186, 584, 115, 281, 466, 53, 375, 502, 304, 578, 132, 557, 122, 533, 416, 383, 356, 317, 536, 569, 7, 500, 33, 352, 525, 159, 461, 123, 141, 318, 493, 411, 61, 468, 451, 96, 465, 60, 118, 114, 106, 391, 345, 164, 91, 567]
    import pandas as pd
    # df = pd.read_csv("/data/icardio/processed/processed.csv")
    df = pd.read_csv("/data/icardio/processed_IQ_ordinal.csv")
    df['IQ_expval'] = df.get("IQ_expval", None)
    df['IQ_class'] = df.get("IQ_class", None)
    # print(len(df))
    df = df[df['view'].isin(["A2C", "A4C"])]
    # df = df[df["split_validated"] == "test"]

    paths = [Path(f"/data/icardio/processed/img/{row['study']}/{str(row['view']).lower()}/{row['dicom_uuid']}_0000.nii.gz") for _, row in df[df["IQ_expval"].isna()].iterrows()]

    # h5 = h5py.File("./../../100first_CARDINAL.h5", "r")
    print(len(paths))
    count = 0
    for p in tqdm(paths, total=len(paths)):
        dicom = p.stem.replace('_0000.nii', '')

        if not p.exists():
            df.loc[df['dicom_uuid'] == dicom, 'IQ_expval'] = -1
            df.loc[df['dicom_uuid'] == dicom, 'IQ_class'] = -1
            continue
        img_nifti = nib.load(p)
        img = img_nifti.get_fdata()
        # print(dicom)

        # segmentation = Path("/data/icardio/processed/segmentation/") / str(p.relative_to("/data/icardio/processed/img/")).replace("_0000", "")
        # seg = nib.load(segmentation).get_fdata()

        # rew = np.asarray(h5[dicom]['reward_map']).squeeze(0).transpose((2, 1, 0))

        iq, expected_val, img1 = predictor.predict_from_numpy(img.copy())

        # img = img.transpose((2, 0, 1))
        # seg = seg.transpose((2, 0, 1))
        # img = img1
        import matplotlib
        # matplotlib.use('TkAgg')
        # import matplotlib.pyplot as plt
        # fig, axes = plt.subplots(1, 2)
        # im = axes[0].imshow(img[0, ...].T, animated=True, cmap='gray')
        # im1 = axes[1].imshow(img[0, ...].T, animated=True, cmap='gray')
        # from matplotlib.colors import LinearSegmentedColormap
        # custom_cmap = LinearSegmentedColormap.from_list("custom", [(0, 0, 0), (0, 1, 0), (1, 0, 0)], N=3)
        # se = axes[1].imshow(seg[0, ...].T, animated=True, cmap=custom_cmap, alpha=0.3)
        # fig.suptitle(f"{dicom} - IQ argmax: {iq.item()} - Expected value: {expected_val: .2f}")
        #
        # #re = axes[1].imshow(rew[0, ...], animated=True, cmap='gray')
        #
        # def update(i):
        #     im.set_array(img[i, ...].T)
        #     se.set_array(seg[i, ...].T)
        #     im1.set_array(img[i, ...])
        #     return im, se, im1
        #
        # animation_fig = animation.FuncAnimation(fig, update, frames=img.shape[0], interval=100, blit=True,
        #                                         repeat_delay=10, )
        # plt.show()
        # Path(f"IQ_TEST_SET").mkdir(exist_ok=True)
        # animation_fig.save(f"IQ_100gifs_CARDINAL/{dicom}.gif")
        # plt.close()

        df.loc[df['dicom_uuid'] == dicom, 'IQ_expval'] = expected_val
        df.loc[df['dicom_uuid'] == dicom, 'IQ_class'] = iq.item()
        # if count > 100:
        #     break
        # count += 1
    # df = df.dropna(subset=["IQ_expval"])
    df.to_csv("/data/icardio/processed_IQ_ordinal.csv")

