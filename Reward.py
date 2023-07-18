import torch


@torch.no_grad()
def accuracy_reward(pred, gt):
    actions = torch.round(pred)
    assert actions.shape == gt.shape
    simple = (actions == gt).float()
    return simple.mean(dim=(1, 2, 3), keepdim=True)
