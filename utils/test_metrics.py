from typing import Tuple

import numpy as np
from medpy.metric import dc, hd
from vital.data.config import LabelEnum


def dice(pred: np.ndarray, target: np.ndarray, labels: Tuple[LabelEnum], exclude_bg: bool = True,
         all_classes: bool = False):
    """Compute dice for one sample.

    Args:
        pred: prediction array in categorical form (H, W)
        target: target array in categorical form (H, W)
        labels: List of labels for which to compute the dice.
        exclude_bg: If true, background dice is not considered.

    Returns:
        mean dice
    """
    dices = []
    if len(labels) > 2:
        for label in labels:
            if exclude_bg and label == 0:
                pass
            else:
                pred_mask, gt_mask = np.isin(pred, label), np.isin(target, label)
                dices.append(dc(pred_mask, gt_mask))
        if all_classes:
            dice_dict = {f"dice_{label.name}": dice for label, dice in zip(labels[1:], dices)}
            dice_dict['Dice'] = np.array(dices).mean()
            return dice_dict
        else:
            return np.array(dices).mean()
    else:
        if all_classes:
            return {'Dice': dc(pred.squeeze(), target.squeeze())}
        else:
            return dc(pred.squeeze(), target.squeeze())


def hausdorff(pred: np.ndarray, target: np.ndarray, labels: Tuple[LabelEnum], exclude_bg: bool = True,
         all_classes: bool = False):
    """Compute hausdorff for one sample.

    Args:
        pred: prediction array in categorical form (H, W)
        target: target array in categorical form (H, W)
        labels: List of labels for which to compute the metric.
        exclude_bg: If true, background dice is not considered.

    Returns:
        mean hausdorff
    """
    hausdorff = []
    if len(labels) > 2:
        for label in labels:
            if exclude_bg and label == 0:
                pass
            else:
                pred_mask, gt_mask = np.isin(pred, label), np.isin(target, label)
                hausdorff.append(hd(pred_mask, gt_mask))
        if all_classes:
            hd_dict = {f"hd_{label.name}": dice for label, dice in zip(labels[1:], hausdorff)}
            hd_dict['Hausdorff'] = np.array(hausdorff).mean()
            return hd_dict
        else:
            return np.array(hausdorff).mean()
    else:
        if all_classes:
            return {'Hausdorff': hd(pred.squeeze(), target.squeeze())}
        else:
            return hd(pred.squeeze(), target.squeeze())
