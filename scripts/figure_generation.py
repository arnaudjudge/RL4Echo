import numpy as np
import nibabel as nib
from pathlib import Path
from matplotlib import pyplot as plt


if __name__ == "__main__":
    output_path = Path("/data/rl_figure/figsv2/")
    output_path.mkdir(exist_ok=True)

    nnunet_path = '/data/nnunet_test/out3/inference_raw/'
    input_path = "/data/rl_figure/sup2"
    fig, ax = plt.subplots(3, 7, figsize=(6.5,3))
    i = 0
    for p in Path(input_path).rglob("*img.nii.gz"):
        img_path = p.as_posix()
        if img_path.split('/')[-2] not in ['di-FC9F-91EC-F095_ES',  "di-A29A-55AC-FE06_ES",  "di-FC9F-91EC-F095_ED"]: #"di-922F-5B7F-F23B_ES"]:"di-D743-7A76-069A_ES",'di-7B00-899D-99F7_ED',
            continue

        act_path = img_path.replace("img", "act")
        print(img_path)

        img = nib.load(img_path).get_fdata()
        gt = nib.load(img_path.replace("img", "gt")).get_fdata()
        sup = nib.load(act_path).get_fdata()
        udas = nib.load(act_path.replace("sup2", "UDAS")).get_fdata()
        rl = nib.load(act_path.replace("sup2", "RL4echo")).get_fdata()

        tsit1 = nib.load(act_path.replace("sup2", "TS-IT_1")).get_fdata()
        tsit2 = nib.load(act_path.replace("sup2", "TS-IT_2")).get_fdata()
        tsit12 = nib.load(act_path.replace("sup2", "TS-IT_12")).get_fdata()

        nnunet = nib.load(f"{nnunet_path}/{img_path.split('/')[-2]}.nii.gz").get_fdata()[..., 0]

        tsit = tsit2*3
        tsit[tsit1 != 0] = 1
        tsit[(tsit1 == tsit2) & (tsit1 != 0) & (tsit2 != 0)] = 2

        ax[i, 0].imshow(img.T, cmap='gray')
        ax[i, 1].imshow(gt.T, cmap='gray')
        ax[i, 2].imshow(sup.T, cmap='gray')
        ax[i, 4].imshow(udas.T, cmap='gray')
        ax[i, 5].imshow(tsit.T, cmap='gray')
        ax[i, 6].imshow(rl.T, cmap='gray')
        ax[i, 3].imshow(nnunet.T, cmap='gray')


        ax[i, 0].axis('off')
        ax[i, 1].axis('off')
        ax[i, 2].axis('off')
        ax[i ,3].axis('off')
        ax[i, 4].axis('off')
        ax[i, 5].axis('off')
        ax[i, 6].axis('off')
        i += 1
        # if i == 3:
        #     i = 0

    ax[0, 0].set_title("Input")
    ax[0, 0].title.set_size(7.5)
    ax[0, 1].set_title("GT")
    ax[0, 1].title.set_size(7.5)
    ax[0, 2].set_title("Baseline (U-Net)")
    ax[0, 2].title.set_size(7.5)
    ax[0, 3].set_title("nnUnet")
    ax[0, 3].title.set_size(7.5)
    ax[0, 4].set_title("UDAS")
    ax[0, 4].title.set_size(7.5)
    ax[0, 5].set_title("TS-IT")
    ax[0, 5].title.set_size(7.5)
    ax[0, 6].set_title("RL4Seg (ours)")
    ax[0, 6].title.set_size(7.5)

    plt.subplots_adjust(left=0.0, #0.1,
                        bottom=0.0, #0.1,
                        right=1.0, #0.9,
                        top=0.925, #0.9,
                        wspace=0.025, #0.05,
                        hspace=0.025) #0.05)
    # plt.subplot_tool()
    # plt.show()
    plt.savefig(f"{output_path}/examplesv2.png")
    # plt.subplot_tool()
    plt.show()
    # plt.close()
