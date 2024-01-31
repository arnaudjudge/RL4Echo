from typing import Optional

import pandas as pd
import torch.utils.data

from pathlib import Path
from typing import Dict, Tuple, Union
from typing import Optional

import h5py
import torch
import torch.utils.data
from pytorch_lightning.trainer.states import TrainerFn
from skimage.exposure import exposure
from torch import Tensor
from torchvision.transforms.functional import to_tensor
from vital.data.camus.config import CamusTags
from vital.data.camus.data_module import CamusDataModule
from vital.data.camus.dataset import Camus
from vital.data.config import Subset
from vital.utils.image.transform import segmentation_to_tensor

ItemId = Tuple[str, int]
InstantItem = Dict[str, Union[str, Tensor]]
RecursiveInstantItem = Dict[str, Union[str, Tensor, Dict[str, InstantItem]]]


class CamusRL(Camus):
    def _get_instant_item(self, index: int) -> InstantItem:
        """Fetches data and metadata related to an instant (single image/groundtruth pair + metadata).

        Args:
            index: Index of the instant sample in the dataset's ``self.item_list``.

        Returns:
            Data and metadata related to an instant.
        """
        patient_view_key, instant = self.item_list[index]

        # Collect data
        with h5py.File(self.root, "r") as dataset:
            view_imgs, view_gts = self._get_data(dataset, patient_view_key, CamusTags.img_proc, CamusTags.gt_proc)

        # Format data
        img = view_imgs[instant]
        gt = self._process_target_data(view_gts[instant])
        img, gt = to_tensor(img), segmentation_to_tensor(gt)

        # Apply transforms on the data
        if self.transforms:
            img, gt = self.transforms(img, gt)

        # Compute attributes on the data
        frame_pos = torch.tensor([instant / len(view_imgs)])
        #gt_attrs = get_segmentation_attributes(gt, self.labels)

        return {
            CamusTags.id: f"{patient_view_key}/{instant}",
            CamusTags.group: patient_view_key,
            CamusTags.img: img,
            CamusTags.gt: gt,
            CamusTags.frame_pos: frame_pos,
            #**gt_attrs,
        }


class CamusRLDataModule(CamusDataModule):
    """Implementation of the ``VitalDataModule`` for the CAMUS dataset."""

    def __init__(
            self,
            dataset_path: Union[str, Path],
            **kwargs,
    ):
        """Initializes class instance.

        Args:
            dataset_path: Path to the HDF5 dataset.
            labels: Labels of the segmentation classes to take into account (including background). If None, target all
                labels included in the data.
            fold: ID of the cross-validation fold to use.
            use_sequence: Enable use of full temporal sequences.
            num_neighbors: Number of neighboring frames on each side of an item's frame to include as part of an item's
                data.
            neighbor_padding: Mode used to determine how to pad neighboring instants at the beginning/end of a sequence.
                The options mirror those of the ``mode`` parameter of ``numpy.pad``.
            **kwargs: Keyword arguments to pass to the parent's constructor.
        """
        super().__init__(dataset_path, **kwargs)

    def setup(self, stage: Optional[str] = None) -> None:  # noqa: D102
        if stage == TrainerFn.FITTING:
            self.datasets[Subset.TRAIN] = CamusRL(image_set=Subset.TRAIN, **self._dataset_kwargs)
        if stage in [TrainerFn.FITTING, TrainerFn.VALIDATING]:
            self.datasets[Subset.VAL] = CamusRL(image_set=Subset.VAL, **self._dataset_kwargs)
        if stage == TrainerFn.TESTING:
            self.datasets[Subset.TEST] = CamusRL(image_set=Subset.TEST, **self._dataset_kwargs)
        if stage == TrainerFn.PREDICTING:
            self.datasets[Subset.PREDICT] = CamusRL(image_set=Subset.TEST, **self._dataset_kwargs)


if __name__ == "__main__":
    import nibabel as nib
    import numpy as np
    save_dir = Path('/home/local/USHERBROOKE/juda2901/dev/data/camus/RLcamus_matched_hist/')

    dl = CamusRLDataModule(dataset_path="/home/local/USHERBROOKE/juda2901/dev/data/camus/camus.h5", labels=[0, 1, 2],
                           batch_size=1)
    dl.setup(stage="test")

    row_list = []

    for i, batch in enumerate(dl.test_dataloader()):
        print(i)
        print(batch.keys())
        print(batch['id'][0])
        print(batch['img'].shape)
        print(batch['gt'].shape)

        affine = np.diag(np.asarray([-1, -1, 1, 0]))
        hdr = nib.Nifti1Header()

        img_path = save_dir / 'img' / (batch['id'][0] + '.nii.gz')
        img_path.parent.mkdir(parents=True, exist_ok=True)
        data = np.rot90(batch['img'].cpu().numpy()[0][0], k=1)

        d = nib.load(
            '/home/local/USHERBROOKE/juda2901/dev/data/icardio/ES_ED_train_subset/st-2B01-4362-9229/a2c/di-153F-C20A-1CD1_img_ES.nii.gz').get_fdata()

        import matplotlib.pyplot as plt

        # plt.figure()
        # plt.hist(data.astype(np.uint8).flatten(), bins=256, range=[0, 255])

        hist, _ = np.histogram(d.flatten(), 256, [0, 255])
        cdf = hist.cumsum()
        cdf_m = np.ma.masked_equal(cdf, 0)
        cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
        cdf = np.ma.filled(cdf_m, 0).astype('uint8')
        data = cdf[data.astype(np.uint8)]
        from skimage import exposure as exp
        matched = exp.match_histograms(data, d, multichannel=False)

        # plt.figure()
        # plt.hist(data.flatten(), bins=256, range=[0, 255])
        #
        # plt.figure()
        # plt.hist(d.flatten(), bins=256, range=[0, 255])
        #
        # plt.figure()
        # plt.hist(matched.flatten(), bins=256, range=[0, 255])
        #
        # plt.show()

        # p1, p99 = np.percentile(data, (0.1, 99.1))
        # data = exposure.rescale_intensity(data, in_range=(p1, p99)) * 255

        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.imshow(data.T, cmap='gray')
        # plt.show()

        nifti_img = nib.Nifti1Image(matched, affine, hdr)
        nifti_img.to_filename(img_path)
        print(img_path)

        gt_path = save_dir / 'gt' / (batch['id'][0] + '.nii.gz')
        gt_path.parent.mkdir(parents=True, exist_ok=True)
        nifti_gt = nib.Nifti1Image(np.rot90(batch['gt'].cpu().numpy()[0], k=1), affine, hdr)
        nifti_gt.to_filename(gt_path)
        print(gt_path)

        row_list += [{'id': batch['id'][0],
                     'split_0': 'test',
                     'study': batch['id'][0].split('/')[0],
                     'view': batch['id'][0].split('/')[1],
                      'dicom_uuid': batch['id'][0].split('/')[2],
                      'valid_segmentation': True,
                     }]


    df = pd.DataFrame.from_dict(row_list)
    print(df)
    df.to_csv(save_dir / 'camus_test.csv')