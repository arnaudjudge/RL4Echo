import warnings
from multiprocessing import Pool
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
# import matplotlib
# matplotlib.use("TkAgg")
from matplotlib import pyplot as plt, animation
from matplotlib.colors import LinearSegmentedColormap
from scipy import ndimage
from torchio import Resize, LabelMap
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from rl4echo.utils.correctors import AEMorphoCorrector
from rl4echo.utils.test_metrics import full_test_metrics

warnings.simplefilter(action='ignore', category=UserWarning)


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


def dict_mean(dict_list):
    mean_dict = {}
    for key in dict_list[0].keys():
        mean_dict[key] = sum(d[key] for d in dict_list) / len(dict_list)
    return mean_dict

def do(p, plot=False, split_mask=False):
    #
    # LOAD IMAGES
    #
    # print(p)
    pred_nifti = nib.load(p)
    pred = clean_blobs(pred_nifti.get_fdata())

    if split_mask:
        pred2_nifti = nib.load(p.as_posix().replace("/1/", "/2/"))
        pred2 = clean_blobs(pred2_nifti.get_fdata())
        pred = pred + pred2 * 2
        pred[pred > 2] = 2
        # pred = pred * 2

    gt_p = next(Path(GT_PATH).rglob(f"*{p.name}"))  # only one exists
    gt = nib.load(gt_p).get_fdata()

    img_p = gt_p.as_posix().replace(".nii.gz", "_0000.nii.gz").replace('segmentation', 'img')
    img_nifti = nib.load(img_p)
    img = img_nifti.get_fdata()

    #
    # METRICS
    #

    # resize if needed
    if pred.shape != gt.shape:
        resize_up = Resize((gt.shape[0], gt.shape[1], gt.shape[2]))
        pred = LabelMap(tensor=pred[None,], affine=np.diag(pred_nifti.header["pixdim"][1:3].tolist() + [1, 0]))
        pred = resize_up(pred).numpy().squeeze(0)

    pred_b = as_batch(pred)
    gt_b = as_batch(gt)
    # print(pred_b.shape, gt_b.shape, img.shape)

    # get voxel spacing
    voxel_spacing = np.asarray([img_nifti.header["pixdim"][1:3]]).repeat(repeats=len(pred_b), axis=0)

    pred_corrector = AEMorphoCorrector("nathanpainchaud/echo-arvae")
    corrected, _, _, _ = pred_corrector.correct_single_seq(
        torch.tensor(img), torch.tensor(pred), voxel_spacing)
    pred_b = as_batch(corrected)

    # compute metrics here
    logs = full_test_metrics(pred_b, gt_b, voxel_spacing, device="cpu", verbose=False)

    #
    # FIGURES
    #
    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        bk = []
        for ax in axes:
            bk += [ax.imshow(img[..., 0].T, animated=True, cmap='gray')]

        custom_cmap = LinearSegmentedColormap.from_list("custom", [(0, 0, 0), (0, 1, 0), (1, 0, 0)], N=3)
        im1 = axes[0].imshow(pred[..., 0].T, animated=True,
                             cmap=custom_cmap,
                             alpha=0.35,
                             interpolation='none')
        axes[0].set_title("Evaluated segmentation")
        axes[0].axis("off")
        im2 = axes[1].imshow(gt[..., 0].T, animated=True,
                             cmap=custom_cmap,
                             alpha=0.35,
                             interpolation='none')
        axes[1].set_title(f"GT")
        axes[1].axis("off")

        def update(i):
            im1.set_array(pred[..., i].T)
            im2.set_array(gt[..., i].T)
            for b in bk:
                b.set_array(img[..., i].T)
            return bk[0], bk[1], im1, im2

        animation_fig = animation.FuncAnimation(fig, update, frames=img.shape[-1], interval=100, blit=True,
                                                repeat_delay=10, )
        animation_fig.save(f"{GIF_PATH}/{p.stem.split('.')[0]}.gif")
        # plt.show()
        plt.close()
    return logs


if __name__ == "__main__":
    GT_PATH = '/data/icardio/subsets/full_3DRL_subset_norm_TESTONLY/segmentation/'
    # SET_PATH = "/home/local/USHERBROOKE/juda2901/dev/MemSAM/SAVED_MASKS/"
    # SET_PATH = "/home/local/USHERBROOKE/juda2901/dev/SAMUS/iCardio_testset_flipped/1/" # SET split_mask=False
    # SET_PATH = "/home/local/USHERBROOKE/juda2901/dev/RL4Echo/testing_raw_CARDINAL_NEW_TESTSET/"
    # SET_PATH = '/data/icardio/subsets/full_3DRL_subset_norm_TESTONLY/2DMICCAI_segmentation/'
    # SET_PATH = '/home/local/USHERBROOKE/juda2901/dev/ASCENT/ICARDIO_152TEST/inference_raw/'
    #SET_PATH = '/home/local/USHERBROOKE/juda2901/dev/MedSAM/iCardio/preds/MedSAM/'
    # SET_PATH = '/home/local/USHERBROOKE/juda2901/dev/RL4Echo/testing_raw_CARDINAL_FROM_MASK-SSL/'
    # SET_PATH = '/home/local/USHERBROOKE/juda2901/dev/RL4Echo/testing_raw_CARDINAL_NO_MASK-SSL/'
    SET_PATH = '/home/local/USHERBROOKE/juda2901/dev/RL4Echo/testing_raw_LM+ANAT_BEST_NARVAL/'
    
    GIF_PATH = None #'./gifs_RL4Seg_corrected/' # './gifs/'
    if GIF_PATH:
        Path(GIF_PATH).mkdir(exist_ok=True)

    paths = [p for p in Path(SET_PATH).rglob('*.nii.gz')]

    all_logs = []
    for idx, p in enumerate(tqdm(paths[::-1], total=len(paths))):
        all_logs += [do(p, plot=False, split_mask=False)]
        # if idx > 5:
        #     break
    # all_logs = []
    # with Pool(processes=10) as pool:
    #     all_logs = list(
    #         pool.starmap(
    #             do,
    #             zip(
    #                 paths[:3]
    #             )
    #         )
    #     )

    # all_logs = process_map(do, paths, max_workers=12, chunksize=1)

    #
    # AGGREGATION AND OUTPUT
    #
    final_results = dict_mean(all_logs)
    print(f"\n\n***\nAGGREGATED RESULTS : {len(all_logs)} sequences\n")
    for k, v in final_results.items():
        if torch.is_tensor(v):
            v = v.item()
        print(k, f"{v}")
