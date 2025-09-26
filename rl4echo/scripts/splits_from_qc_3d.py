import json
from pathlib import Path

import pandas as pd

if __name__=="__main__":

    reviewer1 = '/home/local/USHERBROOKE/juda2901/dev/qc/potentialTESTset3d/'

    df_path = '/data/icardio/subsets/5000_w_test/subset.csv'
    df = pd.read_csv(df_path, index_col=0)
    # df = df.reset_index()

    df['split_validated'] = 'pred'
    # df.loc[df['split_validated'] != 'test', 'split_validated'] = 'pred'
    img_dict = {}

    data1 = []
    for p in Path(reviewer1).glob("*.json"):
        print(p)
        with open(p, "r") as f:
            d = json.load(f)
            data1 += d[1]['data']

    # data2 = []
    # for p in Path(reviewer2).glob("*.json"):
    #     print(p)
    #     with open(p, "r") as f:
    #         d = json.load(f)
    #         data2 += d[1]['data']

    count = 0
    for i in range(len(data1)):
        img1 = data1[i]
        # img2 = data2[1]['data'][i]

        if "Pass" in img1['status']: # and "Pass" in img2['status']:
            # if img1['filename'] != img2['filename']:
            #     raise Exception("SOMETHING WRONG")
            count += 1
            dicom = img1['filename'].split("_")[0]
            img_dict[dicom] = img_dict.get(dicom, [])

    print(count)
    for dicom, i in img_dict.items():
        df.loc[(df['dicom_uuid'] == dicom), 'split_validated'] = 'test'
    print(f"New patients in test: {count}")
    df.loc[df[df['split_validated'] == 'pred'].sample(n=100).index, 'split_validated'] = 'val'  # val set
    df.loc[df[df['split_validated'] == 'pred'].sample(n=1000).index, 'split_validated'] = 'train'  # initial train

    print(df['split_validated'].value_counts())
    df.to_csv(df_path.replace('_second.csv', '.csv'))