from pathlib import Path

import numpy as np
import nibabel as nib
from vital.data.camus.config import Label
from rl4echo.utils.Metrics import is_anatomically_valid
from rl4echo.utils.test_metrics import dice, hausdorff

if __name__ == "__main__":
    seg_path = '/data/nnunet_test/out4/inference_raw/'
    gt_path = '/data/nnunet_test/0_gt/'

    preds = []
    gts = []
    spacings = []
    for pred_path in Path(seg_path).rglob("*.nii.gz"):
        filename = pred_path.as_posix().split('/')[-1]

        pred = nib.load(pred_path)
        preds += [pred.get_fdata()[..., 0]]
        spacings += [pred.header['pixdim'][1:3]]
        gt = nib.load(gt_path + filename)
        gts += [gt.get_fdata()]

    preds = np.asarray(preds)
    spacings = np.asarray(spacings)
    gts = np.asarray(gts)

    test_dice = dice(preds, gts, labels=(Label.BG, Label.LV, Label.MYO),
                     exclude_bg=True, all_classes=True)
    test_dice_epi = dice((preds != 0).astype(np.uint8), (gts != 0).astype(np.uint8),
                         labels=(Label.BG, Label.LV), exclude_bg=True, all_classes=False)

    test_hd = hausdorff(preds, gts, labels=(Label.BG, Label.LV, Label.MYO),
                        exclude_bg=True, all_classes=True, voxel_spacing=spacings)
    test_hd_epi = hausdorff((preds != 0).astype(np.uint8), (gts != 0).astype(np.uint8),
                            labels=(Label.BG, Label.LV), exclude_bg=True, all_classes=False,
                            voxel_spacing=spacings)['Hausdorff']
    anat_errors = is_anatomically_valid(preds)

    print(f"Test Dice {test_dice}")
    print(f"Test Dice EPI {test_dice_epi}")
    print(f"Test HD {test_hd}")
    print(f"Test HD  EPI {test_hd_epi}")
    print(f"Test anat errors {anat_errors.sum() / len(gts)}")

