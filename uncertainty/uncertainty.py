from typing import Any, Dict, Mapping, Sequence, Union

import pytorch_lightning as pl
import torch
from torch import Tensor
from torch.nn import functional as F
import scipy
from uncertainty.unet import UNet
from vital.metrics.train.metric import DifferentiableDiceCoefficient
from vital.data.camus.config import Label

import numpy as np


def prefix(map: Mapping[str, Any], prefix: str, exclude: Union[str, Sequence[str]] = None) -> Dict[str, Any]:
    """Prepends a prefix to the keys of a mapping with string keys.

    Args:
        map: Mapping with string keys for which to add a prefix to the keys.
        prefix: Prefix to add to the current keys in the mapping.
        exclude: Keys to exclude from the prefix addition. These will remain unchanged in the new mapping.

    Returns:
        Mapping where the keys have been prepended with `prefix`.
    """
    if exclude is None:
        exclude = []
    elif isinstance(exclude, str):
        exclude = [exclude]

    return {f"{prefix}{k}" if k not in exclude else k: v for k, v in map.items()}


class SegmentationUncertainty(pl.LightningModule):
    def __init__(self, input_shape=(1, 256, 256), output_shape=(3, 256, 256), *args, **kwargs):
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.input_shape = input_shape
        self.output_shape = output_shape

        self._dice = DifferentiableDiceCoefficient(include_background=False, reduction="none")

        self.hparams.ce_weight = 0.1
        self.hparams.dice_weight = 1

        self.net = self.configure_model()

        self.output_file = "RESULTS.h5"

    def configure_model(self):
        return UNet(input_shape=self.input_shape, output_shape=self.output_shape)

    def forward(self, x):
        return self.net.forward(x)

    def configure_optimizers(self):
        # add weight decay so predictions are less certain, more randomness?
        return torch.optim.Adam(self.parameters(), lr=0.001, weight_decay=1e-4)

    def _shared_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Dict[str, Tensor]:  # noqa: D102
        x, y = batch['img'], batch['gt']

        # Forward
        y_hat = self.net(x)

        # Segmentation accuracy metrics
        if y_hat.shape[1] == 1:
            ce = F.binary_cross_entropy_with_logits(y_hat.squeeze(), y.type_as(y_hat))
        else:
            ce = F.cross_entropy(y_hat, y)

        dice_values = self._dice(y_hat, y)
        dices = {f"dice/{label}": dice for label, dice in zip([Label.LV, Label.MYO], dice_values)}
        mean_dice = dice_values.mean()

        loss = (self.hparams.ce_weight * ce) + (self.hparams.dice_weight * (1 - mean_dice))

        # Format output
        return {"loss": loss, "ce": ce, "dice": mean_dice, **dices}

    def training_step(self, *args, **kwargs) -> Dict[str, Tensor]:  # noqa: D102
        result = prefix(self._shared_step(*args, **kwargs), "train/")
        self.log_dict(result, prog_bar=False, logger=True, on_step=True, on_epoch=None)
        # Add reference to 'train_loss' under 'loss' keyword, requested by PL to know which metric to optimize
        result["loss"] = result["train/loss"]
        return result

    def validation_step(self, *args, **kwargs) -> Dict[str, Tensor]:  # noqa: D102
        result = prefix(self._shared_step(*args, **kwargs), "val/")
        self.log_dict(result, prog_bar=False, logger=True, on_step=None, on_epoch=True)
        return result

    def on_test_epoch_start(self) -> None:
        pass

    def test_step(self, batch, batch_idx):
        print(batch.keys())
        print(batch["id"])

        exit(0)

    def sample_entropy(self, samples, apply_activation=False):
        """
            samples: (N, T, C, H, W)
        """
        samples = torch.tensor(samples)

        if apply_activation:
            probs = torch.sigmoid(samples) if samples.shape[2] == 1 else F.softmax(samples, dim=2)
        else:
            probs = samples
            # activate_fn = torch.sigmoid if samples.shape[2] == 1 else partial(F.softmax, dim=2)

        # probs = torch.stack(probs, dim=0)
        y_hat = probs.mean(0)

        # from matplotlib import pyplot as plt
        # plt.figure()
        # plt.imshow(y_hat.squeeze())

        if samples.shape[2] == 1:
            y_hat = torch.cat([y_hat, 1 - y_hat], dim=0)
            base = 2
        else:
            base = samples.shape[2]

        uncertainty_map = scipy.stats.entropy(y_hat.cpu().numpy(), axis=1, base=base)

        print('UMAP', uncertainty_map.shape)

        return uncertainty_map
