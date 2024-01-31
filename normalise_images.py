from pathlib import Path
import nibabel as nib
from scipy import stats
import skimage.exposure as exp
from matplotlib import pyplot as plt


if __name__ == "__main__":
    data_path = "/home/local/USHERBROOKE/juda2901/dev/data/icardio/ES_ED_train_subset/"
    img_folder = ""
    output_path = "/home/local/USHERBROOKE/juda2901/dev/data/icardio/ES_ED_train_subset_posttreat/"

    for p in Path(data_path + img_folder).rglob('*_img_*.nii.gz'):
        print(p)
        img = nib.load(p)
        data = img.get_fdata()
        print(data.mean())
        print(data.std())

        # data = stats.zscore(data, axis=None)
        data = data / 255
        print(data.mean())
        print(data.std())
        plt.figure()
        plt.imshow(data)

        data = exp.equalize_adapthist(data, clip_limit=0.01)
        plt.figure()
        plt.imshow(data)
        plt.show()

        out_img = nib.Nifti1Image(data, img.affine, img.header)
        out_path = output_path + p.relative_to(data_path).as_posix()
        print(out_path)
        nib.save(out_img, out_path)

        print("\n")
