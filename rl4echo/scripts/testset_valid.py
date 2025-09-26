import os
import shutil
from pathlib import Path

import pandas as pd


if __name__=="__main__":
    df_path = '/data/icardio/subsets/5000_w_test/subset.csv'

    df = pd.read_csv("/data/icardio/subsets/5000_w_test/subset.csv", index_col=0)
    df = df[df['split_validated'] == 'test']
    df['view'] = df['view'].str.lower()
    df['sub_path'] = df.apply(lambda x: '/'.join(x[['study', 'view', 'dicom_uuid']].dropna().values.tolist()), axis=1)

    test_sub_paths = df['sub_path'].tolist()

    print(test_sub_paths)

    for p in test_sub_paths:
        print(p)

        dest = Path(f"/home/local/USHERBROOKE/juda2901/dev/RL4Echo/test_set_validation/{p}.nii.gz")
        dest.parent.mkdir(exist_ok=True, parents=True)
        shutil.copy(f"/data/icardio/processed/segmentation/{p}.nii.gz", dest)

