import numpy as np
import nibabel as nib
import pandas as pd
from tqdm.contrib.concurrent import process_map


def do(path):
    data = nib.load(path).get_fdata()
    return (path.split("/")[-1].split("_")[0], data.shape)


if __name__ == "__main__":
    PATH = '/data/icardio/subsets/full_3DRL_subset_norm/'

    df = pd.read_csv(f"{PATH}/subset_prelim_splits.csv", index_col=0)

    paths = []
    for row in df.iterrows():
        paths += [f"{PATH}/img/{row[1]['study']}/{row[1]['view'].lower()}/{row[1]['dicom_uuid']}_0000.nii.gz"]
    # paths = paths[:1000]
    all_shapes = process_map(do, paths, max_workers=24, chunksize=1)

    print(len(all_shapes))
    print(all_shapes)

    df["H"] = None
    df["W"] = None
    df["T"] = None
    df["Volume"] = None
    for tup in all_shapes:
        df.loc[df['dicom_uuid'] == tup[0], 'H'] = tup[1][0]
        df.loc[df['dicom_uuid'] == tup[0], 'W'] = tup[1][1]
        df.loc[df['dicom_uuid'] == tup[0], 'T'] = tup[1][2]
        df.loc[df['dicom_uuid'] == tup[0], 'Volume'] = tup[1][0] * tup[1][1] * tup[1][2]

    df.to_csv(f"{PATH}/subset_prelim_splits.csv")
    print(df)