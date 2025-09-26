from pathlib import Path

import pandas as pd
import numpy as np
import nibabel as nib
from matplotlib import animation


def get_quantile_indexes(df, quantiles, col, num_samples=1000, seed=42):
    idxes = []
    for i in range(1, len(quantiles)):
        low_q = df[col].quantile(quantiles[i - 1])
        high_q = df[col].quantile(quantiles[i])
        print(f"{low_q} - {high_q}")
        idxes += [df[(df[col] >= low_q) & (df[col] <= high_q)].sample(num_samples, random_state=seed).index.to_list()[:100]]
    return idxes


if __name__ == "__main__":
    df = pd.read_csv("/data/icardio/processed_IQ_ordinal.csv")
    df_segm_ref = pd.read_csv("/data/icardio/processed/subset_csv/merged_subset_w_split_0_1-18.csv")

    iq_thresh = 1.5
    iq_col = "IQ_expval"
    quantiles = [0, 0.25, 0.5, 0.75, 1]

    df = df[df[iq_col] > iq_thresh]
    df = df[df["dicom_uuid"].isin(df_segm_ref["dicom_uuid"])]

    q_idx = get_quantile_indexes(df, quantiles, iq_col)

    for i in range(len(q_idx)):
        paths = [
            Path(f"/data/icardio/processed/img/{row['study']}/{str(row['view']).lower()}/{row['dicom_uuid']}_0000.nii.gz")
            for _, row in df.loc[q_idx[i]].iterrows()
        ]
        print(f"Q{i+1}")
        print(paths)
        save_dir = Path(f"./NEW_TEST_SET/Q{i+1}/")
        save_dir.mkdir(exist_ok=True, parents=True)

        # for p in paths:
        #     img_nifti = nib.load(p)
        #     img = img_nifti.get_fdata().transpose((2, 1, 0))
        #     dicom = p.stem.replace('_0000.nii', '')
        #
        #     segmentation = Path("/data/icardio/processed/segmentation/") / str(p.relative_to("/data/icardio/processed/img/")).replace("_0000", "")
        #     seg = nib.load(segmentation).get_fdata().transpose((2, 1, 0))
        #
        #     import matplotlib
        #     # matplotlib.use('TkAgg')
        #     import matplotlib.pyplot as plt
        #     fig, axes = plt.subplots(1, 1)
        #     im = axes.imshow(img[0, ...], animated=True, cmap='gray')
        #     from matplotlib.colors import LinearSegmentedColormap
        #     custom_cmap = LinearSegmentedColormap.from_list("custom", [(0, 0, 0), (0, 1, 0), (1, 0, 0)], N=3)
        #     se = axes.imshow(seg[0, ...], animated=True, cmap=custom_cmap, alpha=0.3)
        #     fig.suptitle(f"{dicom} - IQ expval: {df[df['dicom_uuid'] == dicom]['IQ_expval'].item():.2f}")
        #
        #     def update(i):
        #         im.set_array(img[i, ...])
        #         se.set_array(seg[i, ...])
        #         return im, se,
        #
        #     animation_fig = animation.FuncAnimation(fig, update, frames=img.shape[0], interval=100, blit=True,
        #                                             repeat_delay=10, )
        #     # plt.show()
        #     animation_fig.save(save_dir / f"{dicom}.gif")
        #     plt.close()

    with pd.ExcelWriter(save_dir / "test_set.xlsx") as writer:

        # use to_excel function and specify the sheet_name and index
        # to store the dataframe in specified sheet
        df.loc[q_idx[0]][["dicom_uuid", "IQ_expval"]].to_excel(writer, sheet_name="Q1", index=False)
        df.loc[q_idx[1]][["dicom_uuid", "IQ_expval"]].to_excel(writer, sheet_name="Q2", index=False)
        df.loc[q_idx[2]][["dicom_uuid", "IQ_expval"]].to_excel(writer, sheet_name="Q3", index=False)
        df.loc[q_idx[3]][["dicom_uuid", "IQ_expval"]].to_excel(writer, sheet_name="Q4", index=False)