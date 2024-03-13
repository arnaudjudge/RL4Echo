from pathlib import Path
import nibabel as nib
from scipy import stats
import skimage.exposure as exp
from matplotlib import pyplot as plt


if __name__ == "__main__":
    data_path = "/home/local/USHERBROOKE/juda2901/dev/data/camus/RLcamus_affine/"
    img_folder = "img/"
    output_path = "/home/local/USHERBROOKE/juda2901/dev/data/camus/RLcamus_affine/"

    for p in Path(data_path + img_folder).rglob('*.nii.gz'):
        print(p)
        img = nib.load(p)
        data = img.get_fdata()
        print(data.mean())
        print(data.std())

        # data = stats.zscore(data, axis=None)
        data = data / 255
        print(data.mean())
        print(data.std())

        # f, (ax1, ax2) = plt.subplots(2, 2)
        # ax1[1].hist(data.flatten() * 255, bins=255, range=[2, 255])
        # ax1[1].set_title("Before Equalization")
        #
        # ax1[0].imshow(data.T * 255)
        # ax1[0].set_title("Before Equalization")

        data = exp.equalize_adapthist(data, clip_limit=0.01)
        # data = exp.equalize_hist(data)
        # data = exp.rescale_intensity(data)

        # plots
        # hist, _ = np.histogram(data.flatten(), 256, [0, 256])
        # cdf = hist.cumsum()
        # cdf_m = np.ma.masked_equal(cdf, 0)
        # cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
        # cdf = np.ma.filled(cdf_m, 0).astype('uint8')
        # data = cdf[data]

        # ax2[0].imshow(data.T * 255)
        # ax2[0].set_title("After Equalization")
        #
        # ax2[1].hist(data.flatten()*255, bins=255, range=[2, 255])
        # ax2[1].set_title("After Equalization")
        # plt.show()

        out_img = nib.Nifti1Image(data, img.affine, img.header)
        out_path = output_path + p.relative_to(data_path).as_posix()
        print(out_path)
        nib.save(out_img, out_path)

        print("\n")
