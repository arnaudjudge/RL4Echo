import json
import pandas as pd

if __name__=="__main__":

    qc_1 = '/home/local/USHERBROOKE/juda2901/dev/data/icardio/20240208_olivier_icardio_es_ed_testset.json'
    qc_2 = '/home/local/USHERBROOKE/juda2901/dev/data/icardio/20240209_NDUCH_icardio_es_ed_testset.json'

    df_path = '/home/local/USHERBROOKE/juda2901/dev/data/icardio/ES_ED_train_subset_2/subset_INITIAL.csv'
    df = pd.read_csv(df_path, index_col=0)

    df['split_0'] = 'pred'

    with open(qc_1, "r") as f1:
        with open(qc_2, "r") as f2:
            data1 = json.load(f1)
            data2 = json.load(f2)

            count = 0
            for i in range(len(data1[1]['data'])):
                img1 = data1[1]['data'][i]
                img2 = data2[1]['data'][i]

                if "Pass" in img1['status'] and "Pass" in img2['status']:
                    if img1['filename'] != img1['filename']:
                        raise Exception("SOMETHING WRONG")
                    count += 1
                    dicom = img1['filename'].split("_")[0]
                    inst = img1['filename'].split("_")[1]

                    df.loc[(df['dicom_uuid'] == dicom) & (df['instant'] == inst), 'split_0'] = 'test'

            print(count)
    df.loc[df[df['split_0'] == 'pred'].sample(n=100).index, 'split_0'] = 'val'  # val set
    df.loc[df[df['split_0'] == 'pred'].sample(n=100).index, 'split_0'] = 'train'  # initial train

    df.to_csv(df_path.replace('_INITIAL.csv', '_official_test.csv'))