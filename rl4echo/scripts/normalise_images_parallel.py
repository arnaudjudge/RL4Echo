import multiprocessing
from pathlib import Path
import nibabel as nib
from scipy import stats
import skimage.exposure as exp
from matplotlib import pyplot as plt
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map


def process(p):
    # print(p)
    p = Path(p)
    img = nib.load(p)
    data = img.get_fdata()
    data = data / 255
    data = exp.equalize_adapthist(data, clip_limit=0.01)
    out_img = nib.Nifti1Image(data, img.affine, img.header)
    out_path = output_path + p.relative_to(data_path).as_posix()
    # print(out_path)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    nib.save(out_img, out_path)
    # print("\n")


if __name__ == "__main__":
    data_path = "/data/icardio/subsets/subset_40k_RL/"
    img_folder = "img/"
    output_path = "/data/icardio/subsets/subset_40k_RL/"

    paths = [p.as_posix() for p in Path(data_path + img_folder).rglob('*.nii.gz')]
    #pool = multiprocessing.Pool(12)
    #zip(*pool.map(process, paths))

    process_map(process, paths, max_workers=12, chunksize=1)

