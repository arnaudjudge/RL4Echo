import torch
from vital.vital.metrics.train.functional import differentiable_dice_score


def accuracy(pred, imgs, gt):
    actions = torch.round(pred)
    assert actions.shape == gt.shape, \
        print(f"Actions shape {actions.shape} vs GT shape {gt.shape}")
    simple = (actions == gt).float()
    return simple.mean(dim=(1, 2), keepdim=True)


def dice_score(output, target):
    out = torch.zeros(len(output))
    # output = target
    for i in range(len(output)):
        # sparse = torch.stack((output[i], 1-output[i]*1000), dim=0).squeeze(2)
        # out[i] = 1 - differentiable_dice_score(sparse, target[i], bg=True)

        intersection = torch.sum(target[i, ...] * output[i, ...])
        out[i] = (2. * intersection) / (torch.sum(target[i, ...]) + torch.sum(output[i, ...]))
        intersect = torch.sum(target[i, ...] * output[i, ...])
        union = torch.sum(output[i, ...]) + torch.sum(target[i, ...]) - intersect
        iou = torch.mean(intersect / union)
        out[i] = iou
    return out

