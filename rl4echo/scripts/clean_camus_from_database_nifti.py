import nibabel as nib
import pandas as pd
import numpy as np
import torchio as tio
from torchio.transforms import Resize
from pathlib import Path
import re

if __name__ == "__main__":
    camus_database_nifti = Path('/home/local/USHERBROOKE/juda2901/dev/data/camus/database_nifti/')
    out_path = Path("/home/local/USHERBROOKE/juda2901/dev/data/camus/RLcamus_from_db_nifti/")
    resize_transform = Resize((256, 256, 1))

    for p in camus_database_nifti.rglob("*.nii.gz"):
        if re.match(r"^.*(ES|ED).*$", p.as_posix()):
            items = p.stem.replace(".nii", "").split("_")
            patient = items[0]
            view = items[1]
            instant = items[2]

            out_subpath = Path(patient) / view.lower() / p.stem.replace(".nii", ".nii.gz").replace("_gt", "")

            img_nifti = nib.load(p)
            data = img_nifti.get_fdata()

            if re.match(r"^.*gt.*$", p.as_posix()):
                out = out_path / 'gt' / out_subpath
                out.parent.mkdir(parents=True, exist_ok=True)
                data = tio.LabelMap(tensor=data[None, ..., None], affine=img_nifti.affine)
                data = resize_transform(data)
                d = data.numpy()
                d[d == 3] = 0
                img = nib.Nifti1Image(d[0, ..., 0].astype(np.uint8), data.affine, img_nifti.header)
                img.to_filename(out)
            else:
                out = out_path / 'img' / out_subpath
                out.parent.mkdir(parents=True, exist_ok=True)
                data = tio.ScalarImage(tensor=data[None, ..., None], affine=img_nifti.affine)
                data = resize_transform(data)
                img = nib.Nifti1Image(data.numpy()[0, ..., 0].astype(np.uint8), data.affine, img_nifti.header)
                img.to_filename(out)
