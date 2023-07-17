import torch


def accuracy_reward(pred, gt):
    # actions = torch.round(pred)
    assert pred.shape == gt.shape
    simple = (pred == gt).float()
    return simple.mean(dim=(1, 2), keepdim=True)
