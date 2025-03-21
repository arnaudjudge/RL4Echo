import shutil
from pathlib import Path


if __name__ == "__main__":

    for p in Path('/data/landmarks_cardinal-icardio/segmentation/').rglob("*di-*.nii.gz"):
        img = '/data/icardio/subsets/potentialTESTSET_not40k/img/' + p.relative_to('/data/landmarks_cardinal-icardio/segmentation/').as_posix()
        img = img.replace(".nii", "_0000.nii")
        print(img)

        # gt = '/data/icardio/subsets/potentialTESTSET_not40k/segmentation/' + p.relative_to('/data/landmarks_cardinal-icardio/segmentation/').as_posix()

        Path('/data/landmarks_cardinal-icardio/img2/' + p.relative_to(
            '/data/landmarks_cardinal-icardio/segmentation/').as_posix()).parent.mkdir(parents=True, exist_ok=True)
        # Path('/data/landmarks_cardinal-icardio/gt/' + p.relative_to(
        #     '/data/landmarks_cardinal-icardio/segmentation/').as_posix()).parent.mkdir(parents=True, exist_ok=True)

        shutil.copy(img, '/data/landmarks_cardinal-icardio/img2/' + p.relative_to('/data/landmarks_cardinal-icardio/segmentation/').as_posix())
        # shutil.copy(gt, '/data/landmarks_cardinal-icardio/gt/' + p.relative_to(
        #     '/data/landmarks_cardinal-icardio/segmentation/').as_posix())