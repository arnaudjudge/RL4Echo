import numpy as np
import json
import pandas as pd

if __name__ == "__main__":

    df = pd.read_csv('metrics_1allowed.csv')

    with open('/home/local/USHERBROOKE/juda2901/dev/qc/temporal_test2/20240926_arnaud_temporal_test2.json', "r") as f:
        d = json.load(f)
        data = d[1]['data']

    passed = [d['filename'].replace("_0000", "") for d in data if 'Fail' not in d['status']]
    failed = [d['filename'].replace("_0000", "") for d in data if 'Fail' in d['status']]

    df_passed = df[df['dicom'].isin(passed)]
    df_failed = df[df['dicom'].isin(failed)]

    print(df_passed.describe())
    print(df_failed.describe())

    agreed = df[(df['dicom'].isin(passed) & df['valid']) | (df['dicom'].isin(failed) & ~df['valid'])]
    print(agreed)

