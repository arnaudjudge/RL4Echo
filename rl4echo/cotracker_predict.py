from pathlib import Path
from typing import Tuple

import hydra
import nibabel as nib
import numpy as np
import torch
from dotenv import load_dotenv
from echotk.tracking.create_mesh import get_mesh
from echotk.tracking.mesh2mask import masks_from_meshes
from echotk.metrics.segmentation_metrics import dice

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from monai import transforms
from monai.data import DataLoader, ArrayDataset, MetaTensor
from monai.transforms import MapTransform
from monai.transforms import ToTensord
from omegaconf import DictConfig

from patchless_nnunet import utils, setup_root
from vital.data.camus.config import Label
from vital.utils.image.us.measure import EchoMeasure

log = utils.get_pylogger(__name__)


class CoTrackerPreprocess(MapTransform):
    """Load and preprocess data path given in dictionary keys.

    Dictionary must contain the following key(s): "image" and/or "label".
    """

    def __init__(
            self, keys, common_spacing, inference_dir,
    ) -> None:
        """Initialize class instance.

        Args:
            keys: Keys of the corresponding items to be transformed.
            common_spacing: Common spacing to resample the data.
        """
        super().__init__(keys)
        self.keys = keys
        self.common_spacing = np.array(common_spacing)

    def __call__(self, data: dict[str, str]):
        # load data
        d = dict(data)
        d['image'] = d['image'].to(torch.float32)

        seg = d['segmentation']
        lm_reward = d['reward_1']

        lv_area = EchoMeasure.structure_area(seg, labels=Label.LV)
        es_index = np.argmin(lv_area)
        init_index = lm_reward.mean(axis=(1, 2))[es_index - 4:es_index + 4].argmax() + (es_index - 4)

        _, _, es_mesh = get_mesh(seg[init_index][None,])
        es_mesh = es_mesh[0]
        d['init_mesh'] = es_mesh.astype(np.float32)
        d['init_index'] = init_index
        return d


class CoTrackerPredictor:
    @classmethod
    def main(cls) -> None:
        """Runs the requested experiment."""
        # Set up the environment
        cls.pre_run_routine()

        # Run the system with config loaded by @hydra.main
        cls.run_system()

    @classmethod
    def pre_run_routine(cls) -> None:
        """Sets-up the environment before running the training/testing."""
        # Load environment variables from `.env` file if it exists
        # Load before hydra main to allow for setting environment variables with ${oc.env:ENV_NAME}
        setup_root()

    @staticmethod
    def load_subject(img_path, seg_folder):
        nifti_img = nib.load(img_path)
        img = nifti_img.get_fdata()  # add batch/channel dim
        hdr = nifti_img.header
        aff = nifti_img.affine

        seg = nib.load(next(Path(seg_folder).rglob(img_path.name.replace("_0000", "")))).get_fdata()
        reward_0 = nib.load(next(Path(seg_folder).rglob(img_path.name.replace("_0000", "_0_reward")))).get_fdata()
        reward_1 = nib.load(next(Path(seg_folder).rglob(img_path.name.replace("_0000", "_1_reward")))).get_fdata()

        return img, hdr, aff, seg, reward_0, reward_1

    @staticmethod
    def get_array_dataset(input_path, segmentation_path, apply_eq_hist=False):
        tensor_list = []
        # find all nifti files in input_path
        # open and get relevant information
        # add to list of data
        input_path = Path(input_path)

        # Handle single NIfTI file
        if input_path.is_file() and (input_path.suffix == ".nii" or "".join(input_path.suffixes[-2:]) == ".nii.gz"):
            nifti_files = [input_path]
        # Handle folder of NIfTI files
        elif input_path.is_dir():
            nifti_files = list(input_path.rglob('*.nii*'))[1:2]
        else:
            raise ValueError(f"Invalid input path: {input_path}")

        for nifti_file_p in nifti_files:
            if nifti_file_p.name.removesuffix('_0000.nii.gz')[-1] == '0':
                continue
            img, hdr, aff, seg, reward_0, reward_1 = CoTrackerPredictor.load_subject(nifti_file_p, segmentation_path)

            # same meta for all (image, segmentation and rewards)
            meta = {
                "filename_or_obj": nifti_file_p.stem.split('.')[0].removesuffix("_0000"),
                "pixdim": hdr['pixdim'],
                "original_affine": aff
            }
            subject_dict = {'image': MetaTensor(img.transpose((2, 1, 0)), meta=meta),
                            'segmentation': seg.transpose((2, 1, 0)),
                            'reward_0': reward_0.transpose((2, 1, 0)),
                            'reward_1': reward_1.transpose((2, 1, 0)),
                            }
            tensor_list.append(subject_dict)
        return tensor_list

    @staticmethod
    @hydra.main(version_base="1.3", config_path="config", config_name="cotracker_predict")
    def run_system(cfg: DictConfig) -> Tuple[dict, dict]:
        """Predict unseen cases with a given checkpoint.

        Currently, this method only supports inference for nnUNet models.

        This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
        failure. Useful for multiruns, saving info about the crash, etc.

        Args:
            cfg (DictConfig): Configuration composed by Hydra.

        Returns:
            Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.

        Raises:
            ValueError: If the checkpoint path is not provided.
        """
        if not cfg.ckpt_path:
            raise ValueError("ckpt_path must not be empty!")

        ts = torch.jit.load(cfg.ckpt_path, map_location='cuda')

        preprocessed = CoTrackerPreprocess(keys=['image', 'segmentation', 'reward_0', 'reward_1'], common_spacing=cfg.common_spacing, inference_dir=cfg.output_path)
        tf = transforms.compose.Compose([preprocessed, ToTensord(keys=['image', 'init_mesh'], track_meta=True)])

        numpy_arr_data = CoTrackerPredictor.get_array_dataset(cfg.input_path, cfg.segmentation_path, cfg.apply_eq_hist)
        dataset = ArrayDataset(img=numpy_arr_data, img_transform=tf)

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=1,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            shuffle=False,
        )

        log.info("Starting predicting!")
        for item in dataloader:
            video = item['image'][0].cuda()
            print(video.shape)
            init_mesh = item['init_mesh'][0].cuda()
            init_index = item['init_index'][0]

            with torch.inference_mode():
                tracks, vis_mask = ts(video, init_mesh, index=init_index)

            print(f'Output shape: {tracks.shape}')

            tracks = tracks.cpu().numpy().squeeze()

            video = video.cpu()

            # save stuff
            base_output_path = Path(f"{cfg.output_path}/{video._meta['filename_or_obj']}")
            base_output_path.parent.mkdir(exist_ok=True, parents=True)
            np.save(f"{base_output_path}.npy", tracks)

            # print(dice(mask, (seg.cpu().numpy()==2), labels=[1]))
            # dices = []
            # for i in range(len(mask)):
            #     dices += [dice(mask[i], (seg[i].cpu().numpy()==2), labels=[1])]
            # dices = np.asarray(dices)
            # print(dices.argmin(), dices.min())

            # TODO: CLEAN UP GIF SECTION ONCE METRICS AND STUFF IS SET
            if cfg.save_as_gif:
                seg = item['segmentation'][0]
                lm_reward = item['reward_1'][0]
                mask = masks_from_meshes(tracks, video.shape[1:])
                error_map = (mask == (seg.cpu().numpy()==2)).astype(float)
                # error_map *= lm_reward.cpu().numpy()

                fig, ax = plt.subplots(1, 4, figsize=(12, 3))
                im = ax[0].imshow(video[0], cmap="gray", animated=True)
                scat = ax[0].scatter([], [], c="r", s=5, animated=True)

                im2 = ax[1].imshow(video[0], cmap="gray", animated=True)
                segm = ax[1].imshow(seg[0], cmap="gray", animated=True, alpha=0.35)

                lm = ax[2].imshow(lm_reward[0], cmap="gray", animated=True)

                e = ax[3].imshow(error_map[0], cmap="gray", animated=True)

                def update(i):
                    im.set_data(video[i])
                    im2.set_data(video[i])
                    scat.set_offsets(tracks[i, :, :2])
                    segm.set_data(seg[i])
                    lm.set_data(lm_reward[i])
                    e.set_data(error_map[i])
                    return im, scat, im2, segm, lm, e

                ani = FuncAnimation(fig, update, frames=len(video), blit=True, interval=100)  # ~10 fps
                ani.save(f"{base_output_path}.gif", writer=PillowWriter(fps=10))
                plt.close(fig)


def main():
    """Run the script."""
    load_dotenv()

    CoTrackerPredictor.main()


if __name__ == '__main__':
    CoTrackerPredictor.main()