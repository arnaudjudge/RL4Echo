from pathlib import Path

import pandas as pd
import numpy as np
import nibabel as nib
from matplotlib import animation


def get_quantile_indexes(df, quantiles, col, num_samples=16, seed=42):
    idxes = []
    for i in range(1, len(quantiles)):
        low_q = df[col].quantile(quantiles[i - 1])
        high_q = df[col].quantile(quantiles[i])
        print(f"{low_q} - {high_q}")
        idxes += [df[(df[col] >= low_q) & (df[col] <= high_q)].sample(num_samples, random_state=seed).index.to_list()]
    return idxes


if __name__ == "__main__":
    df = pd.read_csv("/data/icardio/subsets/icardio_A234C/subset_A234C.csv")

    iq_thresh = 1.5
    iq_col = "IQ_expval"
    quantiles = [0, 0.25, 0.5, 0.75, 1]

    # no need, already filtered
    # df = df[df[iq_col] > iq_thresh]

    df = df[df['view'] == 'A3C']

    q_idx = get_quantile_indexes(df, quantiles, iq_col, 66)

    flat_q_idx_test = [item for sublist in q_idx for item in sublist[:16]]
    flat_q_idx_LM = [item for sublist in q_idx for item in sublist[16:]]

    df_test = df.loc[flat_q_idx_test]
    df_LM = df.loc[flat_q_idx_LM]

    print(len(df_test))
    print(len(list(set(df_test['study']))))
    print(df_test['IQ_expval'].describe())
    df_test.to_csv("/data/icardio/subsets/icardio_A234C/designatedA3CTestSet.csv", index=0)

    print(len(df_LM))
    df_LM.to_csv("/data/icardio/subsets/icardio_A234C/A3CLabelLM.csv", index=0)

