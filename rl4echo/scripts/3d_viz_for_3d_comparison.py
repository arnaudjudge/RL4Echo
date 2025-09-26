from pathlib import Path
import nibabel as nib
from matplotlib import pyplot as plt, animation
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
from rl4echo.inference_from_seq import IMG_PATH
from torchio import Resize, LabelMap
from scipy import ndimage


def as_batch(action):
    y_pred_np_as_batch = action.transpose((2, 0, 1))
    return y_pred_np_as_batch


def clean_blobs(action):
    for i in range(action.shape[-1]):
        try:
            lbl, num = ndimage.label(action[..., i] != 0)
            # Count the number of elements per label
            count = np.bincount(lbl.flat)
            # Select the largest blob
            maxi = np.argmax(count[1:]) + 1
            # Remove the other blobs
            action[..., i][lbl != maxi] = 0
        except:
            print("WARNING EMPTY SEGMENTATION FRAME")
    return action


def load_and_clean(p, img_size=None, skip_blobs=False):
    nifti = nib.load(p)
    img = nifti.get_fdata() if skip_blobs else clean_blobs(nifti.get_fdata())
    img = as_batch(img)
    if img_size:
        resize_up = Resize((img_size[0], img_size[1], img_size[2]))
        img = LabelMap(tensor=img[None,], affine=np.diag(nifti.header["pixdim"][1:3].tolist() + [1, 0]))
        img = resize_up(img).numpy().squeeze(0)
    return img


if __name__ == "__main__":
    GT_PATH = '/data/icardio/subsets/full_3DRL_subset_norm_TESTONLY/segmentation/'
    IMG_PATH = '/data/icardio/subsets/full_3DRL_subset_norm_TESTONLY/img/'
    SET_PATH = [
        # ("/home/local/USHERBROOKE/juda2901/dev/RL4Echo/testing_raw_CARDINAL_NEW_TESTSET/", "Baseline 3D U-Net"),
        ('/home/local/USHERBROOKE/juda2901/dev/ASCENT/ICARDIO_152TEST/inference_raw/', "nnU-Net"),
        # ("/home/local/USHERBROOKE/juda2901/dev/SAMUS/iCardio_testset_flipped/merged/", "SAMUS"),
        # ("/home/local/USHERBROOKE/juda2901/dev/MemSAM/SAVED_MASKS/", "MemSAM"),
        # ('/home/local/USHERBROOKE/juda2901/dev/RL4Echo/testing_raw_CARDINAL_FROM_MASK-SSL/', "MaskedSSL"),
        ('/data/icardio/subsets/full_3DRL_subset_norm_TESTONLY/2DMICCAI_segmentation/', "RL4Seg (2D)"),
        ('/home/local/USHERBROOKE/juda2901/dev/RL4Echo/testing_raw_LM+ANAT_BEST_NARVAL_TTA/', "RL4Seg3D_A+LM"),
        ('/home/local/USHERBROOKE/juda2901/dev/RL4Echo/testing_raw_ANAT-LM-T_NARVAL_TTA_LAST/', "RL4Seg3D_A+LM+T"),
        ('/home/local/USHERBROOKE/juda2901/dev/RL4Echo/testing_raw_LM+ANAT_TTO_AVTV_BEST/', "RL4Seg3D_TTO"),
    ]

    VIZ_PATH = "./viz_TTO_COMPARISON/"
    Path(VIZ_PATH).mkdir(exist_ok=True)
    df = pd.read_csv("/data/icardio/subsets/full_3DRL_subset_norm_TESTONLY/subset_official_splits.csv", index_col=0)
    df = df[df['split_official_test'] == 'test']
    dicoms = [p.name.replace(".nii.gz", "") for p in Path(SET_PATH[2][0]).rglob('*.nii.gz') if
              p.name.replace(".nii.gz", "") in df['dicom_uuid'].to_list()]

    dicoms = ["di-3C91-E35C-906B", "di-7061-409C-0CAF", "di-8075-490B-DF12", "di-1ECA-EAC3-8EAE"]
    dicoms = ['di-C910-B188-A16F', 'di-1063-8366-5D88', 'di-0CA9-8FC9-25BB', 'di-AE68-A41B-5185',
               'di-47EB-1516-2456', 'di-5A75-9C41-46D7', 'di-7798-9454-7280', 'di-8075-490B-DF12',
               'di-C9E0-1668-D365', 'di-E42E-19EA-16E7', 'di-9BBB-E0E8-ABFE', 'di-46D0-7328-762F',
               'di-0343-2269-9415', 'di-1FD4-CB18-6EFC', 'di-A540-DBDD-7C1F', 'di-056D-EBB0-1AF2',
               'di-BBFD-BFA0-F9FA', 'di-1F8E-37E5-57ED', 'di-C5EF-3C1E-3B51', 'di-B55F-84D9-6833',
               'di-5BB1-44DE-60BA', 'di-7291-29A4-97F8']

    print(len(dicoms))
    for d in dicoms:
        print(d)
        img_list = [(load_and_clean(next(Path(IMG_PATH).rglob(f"*{d}*")), skip_blobs=True), "Image")]
        img_list += [(load_and_clean(next(Path(GT_PATH).rglob(f"*{d}*"))), "GT")]
        img_list += [(load_and_clean(next(Path(p[0]).rglob(f"*{d}*")), img_size=img_list[0][0].shape), p[1]) for p in
                     SET_PATH]

        ## animation
        i = 0
        fig, ax = plt.subplots(1, 7, figsize=(16, 4))
        im = ax[0].imshow(img_list[0][0][i, ...].T, cmap='gray', interpolation='none')
        ax[0].axis('off')
        ax[0].set_title(img_list[0][1])
        axim = []
        axbk = []
        custom_cmap = LinearSegmentedColormap.from_list("custom", [(0, 0, 0), (0, 1, 0), (1, 0, 0)], N=3)

        for j in range(1, len(img_list)):
            axbk += [ax[j].imshow(img_list[0][0][i, ...].T, cmap='gray', interpolation='none')]
            axim += [ax[j].imshow(img_list[j][0][i, ...].T, cmap=custom_cmap, interpolation='none', alpha=0.45)]
            ax[j].axis('off')
            ax[j].set_title(img_list[j][1])


        def update(i):
            im.set_array(img_list[0][0][i, ...].T)
            for k in range(len(axbk)):
                axbk[k].set_array(img_list[0][0][i, ...].T)
            for k in range(len(axim)):
                axim[k].set_array(img_list[k + 1][0][i, ...].T)
            return im, axbk[0], axbk[1], axbk[2], axbk[3], axbk[4], axbk[5], axim[0], axim[1], axim[2], axim[3],  axim[4], axim[5],# axim[6], axim[7], axim[8],


        animation_fig = animation.FuncAnimation(fig, update, frames=len(img_list[0][0]), interval=100, blit=True,
                                                repeat_delay=10, )
        plt.subplots_adjust(left=0.005,
                            bottom=0.0,
                            right=0.995,
                            top=1.0,
                            wspace=0.015,
                            hspace=0.0)
        animation_fig.save(f"{VIZ_PATH}/{d}.gif")
        plt.show()
# di-89C0-9C48-1DFA