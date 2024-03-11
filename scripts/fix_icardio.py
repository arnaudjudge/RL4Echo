import shutil
from pathlib import Path

if __name__ == "__main__":
    in_path = '/home/local/USHERBROOKE/juda2901/dev/data/icardio/ES_ED_train_subset/'
    out_path = '/home/local/USHERBROOKE/juda2901/dev/data/icardio/ES_ED_train_subset_no_adapthist/'

    for p in Path(in_path).rglob('*_img_*.nii.gz'):
        print(p)
        out = p.as_posix().replace("_img", "")
        out = out_path + 'img/' + Path(out).relative_to(in_path).as_posix()
        print(out)
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(p, out)

    for p in Path(in_path).rglob('*_mask_*.nii.gz'):
        print(p)
        out = p.as_posix().replace("_mask", "")
        out = out_path + 'gt/' + Path(out).relative_to(in_path).as_posix()
        print(out)
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(p, out)
