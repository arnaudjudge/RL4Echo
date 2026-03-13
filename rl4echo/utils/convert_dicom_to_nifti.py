from pathlib import Path

import pydicom
import nibabel as nib
import numpy as np

def convert_dcm_to_nii(input_path, output_path):
    dcm = pydicom.dcmread(input_path)
    arr = dcm.pixel_array
    if len(arr.shape) > 3:
        arr = arr.mean(-1)
    data = arr

    spacing = None
    if 'PixelSpacing' in dcm:
        spacing = [float(x) for x in dcm.PixelSpacing]
    elif 'ImagerPixelSpacing' in dcm:
        spacing = [float(x) for x in dcm.ImagerPixelSpacing]
    elif 'SequenceOfUltrasoundRegions' in dcm:
        seq = dcm.SequenceOfUltrasoundRegions[0]
        if hasattr(seq, 'PhysicalDeltaX') and hasattr(seq, 'PhysicalDeltaY'):
            spacing = [abs(float(seq.PhysicalDeltaX)) * 10, abs(float(seq.PhysicalDeltaY)) * 10, 1.0]
    else:
        spacing = [0.37, 0.37]  # default fallback if no calibration info
    data = data.transpose((2, 1, 0))
    aff = np.diag([spacing[1], spacing[0], 1, 0])
    nib.Nifti1Image(data, aff).to_filename(output_path.as_posix().replace(".dcm", ".nii.gz"))


if __name__ == "__main__":

    input_path = "/data/ORCHID/dicoms/"
    output_path = "/data/ORCHID/nifti_raw/"

    for p in Path(input_path).rglob("*"):
        split_name = p.stem.split("_")
        patho = split_name[0]
        id = split_name[1]
        view = split_name[2][-1]
        constructed_out_path = Path(f"{output_path}/{patho}-{id}/{patho}{id}-{view}CH.nii.gz")
        constructed_out_path.parent.mkdir(parents=True, exist_ok=True)
        convert_dcm_to_nii(p, constructed_out_path)

