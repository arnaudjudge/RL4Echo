import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
from matplotlib import animation
import argparse
import SimpleITK as sitk
import pydicom
import os


def load_image(path):
    """
    Load a single medical image (NIfTI or DICOM).

    Args:
        path (str): Path to a single .nii/.nii.gz or .dcm file.

    Returns:
        tuple: (image_array, spacing)
            image_array (np.ndarray): Image data.
            spacing (tuple): Pixel spacing or voxel spacing.
    """
    ext = os.path.splitext(path)[-1].lower()

    if ext in [".nii", ".gz"]:  # NIfTI
        img = sitk.ReadImage(path)
        arr = sitk.GetArrayFromImage(img)
        return arr
    elif ext == ".dcm":  # Single DICOM file
        dcm = pydicom.dcmread(path)
        arr = dcm.pixel_array
        if len(arr.shape) > 3:
            arr = arr.mean(-1)
        return arr

    else:
        raise ValueError(f"Unsupported file type: {ext}")


def main():
    parser = argparse.ArgumentParser(description="Load a single NIfTI or DICOM image.")
    parser.add_argument(
        "input", type=str, help="Path to a single .nii/.nii.gz or .dcm file.")
    parser.add_argument(
        "--save", "-s", type=str, required=False, help="Path to a .gif file.")
    parser.add_argument(
        "--overlay", "-o", type=str, required=False, help="Path to a segmentation or overlay file.")
    args = parser.parse_args()

    arr = load_image(args.input)
    print(f"Loaded image with shape {arr.shape}")

    f, ax = plt.subplots()
    im = ax.imshow(arr[0, ...], animated=True, cmap='gray')
    if args.overlay:
        over = load_image(args.overlay)
        ov = ax.imshow(over[0, ...], animated=True, cmap='gray', alpha=0.4)
    def update(i):
        im.set_array(arr[i, ...])
        if args.overlay:
            ov.set_array(over[i, ...])
            return im, ov,
        return im,

    animation_fig = animation.FuncAnimation(f, update, frames=arr.shape[0], interval=100, blit=True, repeat_delay=10)

    if args.save:
        animation_fig.save(args.save)
        print(f"Saved gif to {args.save}")
        plt.close()
    else:
        plt.show()



if __name__ == "__main__":
    main()
