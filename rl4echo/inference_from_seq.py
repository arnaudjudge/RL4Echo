from pathlib import Path

import numpy as np
import nibabel as nib
import pandas as pd
import skimage.exposure as exp
import torch
from scipy import ndimage
from torchio import Resize, ScalarImage, LabelMap
from tqdm import tqdm

from rl4echo.utils.Metrics import is_anatomically_valid
from vital.models.segmentation.unet import UNet

from matplotlib import pyplot as plt

# CSV_PATH = "/data/icardio/subsets/full_3DRL_subset_norm_TESTONLY/subset.csv"
# IMG_PATH = "/data/icardio/subsets/full_3DRL_subset_norm_TESTONLY/img/"
IMG_PATH = "./../../patchless-nnUnet/data/ftournoux_nifti/"
# OUT_PATH = "/data/icardio/subsets/full_3DRL_subset_norm_TESTONLY/2D_initial_actor_segmentation/"
CAP = 100000

def clean(s):
    lbl, num = ndimage.label(s != 0)
    # Count the number of elements per label
    count = np.bincount(lbl.flat)
    # Select the largest blob
    maxi = np.argmax(count[1:]) + 1
    # Remove the other blobs
    s[lbl != maxi] = 0
    return s


if __name__ == "__main__":

    # df = pd.read_csv(CSV_PATH, index_col=0)
    # df['MICCAI2024_frame_AV'] = df.get("MICCAI2024_frame_AV", None)
    # df['MICCAI2024_overall_AV'] = df.get("MICCAI2024_overall_AV", None)

    resize_down = Resize((256, 256, 1))

    model = UNet(input_shape=(1, 256, 256), output_shape=(3, 256, 256))
    model.load_state_dict(torch.load("/data/rl_logs_3d/run_4/3/actor.ckpt"))

    def get_segmentation(img):
        img = torch.tensor(img.astype(np.float32))
        if torch.cuda.is_available():
            model.cuda()
            img = img.cuda()
        out = torch.sigmoid(model(img))
        return out.cpu().detach().numpy()[0]
    j = 0

    #df_subset = df[(df['passed'] == True) & (df['MICCAI2024_overall_AV'].isna())]
    #for idx in tqdm(df_subset.index, total=min(CAP, len(df_subset.index))):
    #    row = df.loc[idx].to_dict()
    #    label_path = Path(OUT_PATH) / row['study'] / row['view'].lower() / f"{row['dicom_uuid']}.nii.gz"
    paths = [p for p in Path(IMG_PATH).rglob("*.nii.gz")]
    for p in paths:
        label_path = Path(p.as_posix().replace("ftournoux_nifti", "ftournoux_2d_seg"))
        if True: #not label_path.exists():
            img_path = p #f"{IMG_PATH}/{row['study']}/{row['view'].lower()}/{row['dicom_uuid']}_0000.nii.gz"
            nifti_img = nib.load(img_path)
            data = nifti_img.get_fdata()

            # seg = nib.load(label_path).get_fdata()
            data = data / 255
            # for i in range(data.shape[-1]):
            #     data[..., i] = exp.equalize_adapthist(data[..., i], clip_limit=0.01)
            resize_up = Resize((data.shape[0], data.shape[1], 1))
            label_seq = np.zeros_like(data, dtype=np.uint32)

            for i in range(data.shape[-1]):
                d = data[..., i]
                # d = exp.equalize_adapthist(d, clip_limit=0.01)
                d_small = resize_down(ScalarImage(tensor=d[None, ..., None], affine=nifti_img.affine))

                # segment and post-process
                s = get_segmentation(d_small.numpy().transpose((0, 3, 1, 2)))
                s = np.argmax(s, axis=0)
                lbl, num = ndimage.label(s != 0)
                # Count the number of elements per label
                count = np.bincount(lbl.flat)
                # Select the largest blob
                maxi = np.argmax(count[1:]) + 1
                # Remove the other blobs
                s[lbl != maxi] = 0

                label = LabelMap(tensor=s[None, ..., None], affine=d_small.affine)
                label = resize_up(label)

                label_seq[..., i] = label.numpy()[0, ..., 0]

            label_nifti = nib.Nifti1Image(label_seq, header=nib.Nifti1Header(), affine=nifti_img.affine)
            label_path.parent.mkdir(parents=True, exist_ok=True)
            nib.save(label_nifti, label_path)

            import matplotlib.pyplot as plt
            img = data
            final_preds = label_seq
            fig, axes = plt.subplots(1, 1)
            bk = axes.imshow(img[..., 0].T, animated=True, cmap='gray')
            im = axes.imshow(final_preds[..., 0].T, animated=True, alpha=0.4)


            def update(i):
                bk.set_array(img[..., i].T)
                im.set_array(final_preds[..., i].T)
                return bk, im,


            from matplotlib import animation

            animation_fig = animation.FuncAnimation(fig, update, frames=final_preds.shape[-1], interval=100, blit=True,
                                                    repeat_delay=10, )
            # animation_fig.save(f'{self.trainer.default_root_dir}/{fname}.gif')
            plt.show()


        else:
            label_seq = nib.load(label_path).get_fdata()

        # anatomical metrics
        # anatomical_validities = is_anatomically_valid(label_seq.transpose(2, 0, 1))

        # row['MICCAI2024_frame_AV'] = anatomical_validities.cpu().numpy().tolist()
        # row['MICCAI2024_overall_AV'] = all(anatomical_validities)
        # df.loc[idx] = row
        # df.to_csv(CSV_PATH)

        j += 1
        if j > CAP:
            break
