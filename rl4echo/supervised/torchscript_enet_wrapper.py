from typing import Tuple, List, Union
import numpy as np
import torch
from lightning import LightningModule
import torch.nn.functional as F
from torch.nn.functional import softmax
from torchvision.transforms.functional import adjust_contrast, rotate
from torch.nn.functional import pad

from rl4echo.supervised.supervised_enet_optimizer import SupervisedEnetOptimizer


from typing import Tuple, List

import torch
import torch.nn.functional as F
from torch import Tensor


class SectorEnetInferenceWrapper(torch.nn.Module):
    def __init__(self, net: torch.nn.Module):
        super().__init__()
        self.net = net
        self.net.eval()

    def undo_pad_centered_to_multiple(
        self,
        x: Tensor,
        pads: List[int],
    ) -> Tensor:
        pad_left = pads[0]
        pad_right = pads[1]
        pad_top = pads[2]
        pad_bottom = pads[3]

        H_end = x.shape[-2] - pad_bottom
        W_end = x.shape[-1] - pad_right

        return x[:, :, pad_top:H_end, pad_left:W_end]

    def pad_centered_to_multiple(
        self,
        x: Tensor,
        multiple: int = 8,
    ) -> Tuple[Tensor, List[int]]:
        H = x.shape[-2]
        W = x.shape[-1]

        new_H = ((H + multiple - 1) // multiple) * multiple
        new_W = ((W + multiple - 1) // multiple) * multiple

        pad_H = new_H - H
        pad_W = new_W - W

        pad_top = pad_H // 2
        pad_bottom = pad_H - pad_top
        pad_left = pad_W // 2
        pad_right = pad_W - pad_left

        pads: List[int] = [pad_left, pad_right, pad_top, pad_bottom]

        x = F.pad(x, pads)

        return x, pads

    def forward(self, x: Tensor) -> Tensor:
        # x.ndim is not supported in TorchScript — use x.dim() instead.
        if x.dim() == 4:
            # dim=(3,) tuple is not supported in TorchScript — use a List[int].
            x = x.mean(dim=[3])

        x = x.unsqueeze(1)  # -> (1, 1, H, W) for conv net

        x, pad_vals = self.pad_centered_to_multiple(x)

        out = torch.round(torch.sigmoid(self.net(x)))



        out = self.undo_pad_centered_to_multiple(out, pad_vals)

        return out


if __name__ == "__main__":

    enet_optimizer = SupervisedEnetOptimizer.load_from_checkpoint(
        "/home/local/USHERBROOKE/juda2901/dev/RL4Echo/rl4echo/test_logs/RL4Echo/8fa6524c78904f998a46d55cd1ab8aa7/checkpoints/epoch=38-step=624.ckpt")

    wrapper = SectorEnetInferenceWrapper(enet_optimizer.net).cpu()
    example_img = torch.rand((1, 487, 480)) # B, C, H, W, T
    print(wrapper(example_img).shape)

    script = torch.jit.script(wrapper)
    # script = torch.jit.optimize_for_inference(script)
    torch.jit.save(script, "/home/local/USHERBROOKE/juda2901/dev/vitalab/Echo-Toolkit/data/model_weights/torchscript_sector_enet.pt")

    print(script(example_img).shape)

    obj = torch.jit.load("/home/local/USHERBROOKE/juda2901/dev/vitalab/Echo-Toolkit/data/model_weights/torchscript_sector_enet.pt")
    print(type(obj))


