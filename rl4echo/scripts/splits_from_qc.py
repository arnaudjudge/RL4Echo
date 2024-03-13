import json
import pandas as pd

if __name__=="__main__":

    #qc_1 = '/home/local/USHERBROOKE/juda2901/dev/data/icardio/20240214_olivier_icardio_es_ed_testset2.0.json'
    #qc_2 = '/home/local/USHERBROOKE/juda2901/dev/data/icardio/20240215_NDUCH_icardio_es_ed_testset2.0.json'

    qc_1 = '/home/local/USHERBROOKE/juda2901/dev/data/icardio/20240215_olivier_icardio_es_ed_testset3.0.json'
    qc_2 = '/home/local/USHERBROOKE/juda2901/dev/data/icardio/20240215_NDUCH_icardio_es_ed_testset3.0.json'

    df_path = '/home/local/USHERBROOKE/juda2901/dev/data/icardio/ES_ED_train_subset_affine/subset_official_test_second.csv'
    df = pd.read_csv(df_path, index_col=0)

    df.loc[df['split_0'] != 'test', 'split_0'] = 'pred'
    img_dict = {}
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
                    img_dict[dicom] = img_dict.get(dicom, []) + [inst]

            print(count)
    count = 0
    for dicom, i in img_dict.items():
        if len(i) < 2:
            continue
        count += 1
        for inst in i:
            df.loc[(df['dicom_uuid'] == dicom) & (df['instant'] == inst), 'split_0'] = 'test'
    print(f"New patients in test: {count}")
    df.loc[df[df['split_0'] == 'pred'].sample(n=100).index, 'split_0'] = 'val'  # val set
    df.loc[df[df['split_0'] == 'pred'].sample(n=100).index, 'split_0'] = 'train'  # initial train

    print(df[df['split_0'] == 'test'])
    df.to_csv(df_path.replace('_second.csv', '.csv'))