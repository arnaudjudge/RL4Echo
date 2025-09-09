import pandas as pd
from google.cloud import storage
from tqdm.contrib.concurrent import process_map
from tqdm import tqdm
from pathlib import Path
import numpy as np
import SimpleITK as sitk
from PIL import Image
from medpy.metric import dc, hd
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy import ndimage

def calculate_dice(triplicate):
    combo1 = dc(triplicate[0], triplicate[1])
    combo2 = dc(triplicate[0], triplicate[2])
    combo3 = dc(triplicate[1], triplicate[2])
    return np.mean([combo1, combo2, combo3])

def calculate_hd(triplicate, spacing=(1, 1)):
    combo1 = hd(triplicate[0], triplicate[1], voxelspacing=spacing)
    combo2 = hd(triplicate[0], triplicate[2], voxelspacing=spacing)
    combo3 = hd(triplicate[1], triplicate[2], voxelspacing=spacing)
    return np.mean([combo1, combo2, combo3])

def load_image(path):
    image = (np.array(Image.open(path).convert("L")) != 0).astype(np.uint8)
    if not np.any(image):
        return None
    # count and remove blobs
    lbl, num = ndimage.label(image != 0)
    # Count the number of elements per label
    count = np.bincount(lbl.flat)
    # Select the largest blob
    maxi = np.argmax(count[1:]) + 1
    # Remove the other blobs
    image[lbl != maxi] = 0
    return image


if __name__ == "__main__":

    df = pd.concat([pd.read_csv("/data/icardio/A4C_LV_triplicate_99443b57-8964-40d4-a532-39432d385283.csv"),
                    pd.read_csv("/data/icardio/A2C_LV_triplicate_f33921dd-03e9-4853-b873-c3264a7ed4df.csv")])
    df_image = pd.read_csv("/data/icardio/processed/processed.csv")
    collected = df.groupby(["dicom_uuid", "phase"]).agg(list).reset_index()
    collected = collected[collected["segmentation_uuid"].apply(len) >= 3]

    dices = []
    hds = []
    for idx, group in tqdm(collected.iterrows(), total=len(collected)):
        if group['dicom_uuid'] in df_image['dicom_uuid']:
            group["dicom_uuid"]
        good_group = True
        triplicate = []
        group_seg_id = group['segmentation_uuid']
        group_spacing = (group['dx'][0]*10, group['dy'][0]*10)
        group_frame_idx = group['frame_index']

        for seg_id in group_seg_id:
            seg_p = Path(f"/data/icardio/triplicate_segmentation/{seg_id}.png")
            if not seg_p.exists():
               good_group = False
               break
            img = load_image(seg_p)
            if img is None:
                good_group = False
            triplicate += [img]

            # if not max(group_frame_idx) - min(group_frame_idx) <= 3:
            #     good_group = False

        if good_group:
            dices.append(calculate_dice(triplicate))
            hds.append(calculate_hd(triplicate, group_spacing))

            # custom_cmap = LinearSegmentedColormap.from_list("custom", [(0, 0, 0), (0, 1, 0), (1, 0, 0), (0, 0, 1)], N=4)
            # plt.figure()
            # plt.title(f"{group['dicom_uuid']} - "
            #           f"{group['phase']} - "
            #           f"Dice: {calculate_dice(triplicate):.4f} - "
            #           f"HD: {calculate_hd(triplicate, group_spacing):.2f}")
            # plt.imshow(triplicate[0]+triplicate[1]+triplicate[2], cmap=custom_cmap)

            # # plt.figure()
            # # plt.imshow(triplicate[0])
            # # plt.figure()
            # # plt.imshow(triplicate[1])
            # # plt.figure()
            # # plt.imshow(triplicate[2])
            #
            # plt.show()

    dices = np.asarray(dices)
    hds = np.asarray(hds)
    print(f"{len(dices)} total valid groups out of {len(collected)},\n"
          f"DICE: {dices.mean()}, \n"
          f"HD: {hds.mean()})")

    print(dices.max())
    print(hds.min())

