import pickle
import random
from typing import Dict

import numpy as np
import sklearn
import torch
from lightning import LightningModule
from matplotlib import pyplot as plt
from pytorch_lightning.loggers import CometLogger
from torch import nn, optim
from torch.nn import functional as F
from vital.models.segmentation.unet import UNet

from rl4echo.rewardnet.unet_heads import UNet_multihead
from rl4echo.utils.Metrics import accuracy
from rl4echo.utils.logging_helper import log_sequence, log_video


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean', weight=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.weight = weight

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none') #, weight=self.weight.to("cuda:0"))
        p_t = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - p_t)**self.gamma * ce_loss

        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss


class IQ3DOptimizer(LightningModule):
    def __init__(self, net, loss=nn.CrossEntropyLoss(), save_model_path="last_IQ.ckpt", model_weights=None, **kwargs):
        super().__init__(**kwargs)

        self.net = net
        self.loss = loss
        self.save_model_path = save_model_path

        from torchinfo import summary
        summary(self.net, input_size=(1, 1, 256, 256, 32))

        # class_counts = torch.tensor([48, 256, 248, 43, 5])  # your class counts
        # class_weights = 1.0 / class_counts.float()
        # class_weights = class_weights / class_weights.sum()  # normalize
        # print(f"Using class weights: {class_weights}")
        # # self.loss = nn.CrossEntropyLoss(weight=class_weights.to(self.device))
        # self.loss = FocalLoss(weight=class_weights.to(self.device))

        self.preds = []
        self.targets = []

        if model_weights:
            weights = torch.load(model_weights)
            if weights.get('state_dict', None):
                weights = weights['state_dict']
            self.load_state_dict(weights)

    def forward(self, x):
        out = self.net(x)
        return out

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=1e-3)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "min")
        return {"optimizer": opt, "lr_scheduler": sch, "monitor": "val/loss"}

    def training_step(self, batch, *args, **kwargs) -> Dict:
        x, y = batch['img'], batch['label']

        y_pred = self(x)
        loss = self.loss(y_pred, y)

        acc = (y == y_pred.argmax(dim=1)).type(torch.float)
        logs = {
            'loss': loss,
            'acc': acc.mean()
        }

        self.log_dict(logs, on_epoch=True)
        return logs

    def validation_step(self, batch, batch_idx: int):
        x, y = batch['img'], batch['label']
        y_pred = self(x)
        loss = self.loss(y_pred, y)

        acc = (y == y_pred.argmax(dim=1)).type(torch.float)

        self.log_dict({"val/loss": loss,
                       "val/acc": acc.mean()
                       })

        # log images
        if self.trainer.local_rank == 0:
            idx = random.randint(0, len(x) - 1)  # which image to log
            log_sequence(self.logger, img=x[idx].permute(0, 2, 3, 1), title=f'Image : label {y[idx].item()}, pred {y_pred[idx].argmax(dim=0).item()}',
                         number=batch_idx, epoch=self.current_epoch)

        return {'loss': loss}

    def test_step(self, batch, batch_idx):
        x, y = batch['img'], batch['label']
        label = y
        y_pred = self(x)
        loss = self.loss(y_pred, y)

        print(torch.softmax(y_pred, dim=1))

        acc = (y == y_pred.argmax(dim=1)).type(torch.float)

        self.log_dict({"test/loss": loss,
                       "test/acc": acc.mean()
                       })

        self.preds += [y_pred.argmax(dim=1).item()]
        self.targets += [label.item()]

        if self.trainer.local_rank == 0:
            for i in range(len(x)):
                log_video(self.logger, img=x[i].permute(0, 2, 3, 1),
                          title=f'test_Image_Class - Label {label.item()} - Pred {y_pred.argmax(dim=1).item()}',
                          number=batch_idx * (i + 1), epoch=self.current_epoch)

        return {'loss': loss}

    def on_test_end(self) -> None:
        self.save_model()
        print(np.unique(self.targets))
        cm = sklearn.metrics.confusion_matrix(self.targets, self.preds)

        self.trainer.logger.experiment.log_confusion_matrix(matrix=cm, labels=sorted(np.unique(self.targets)),
                                                            title="Categorical")

        print(cm)
        disp = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(self.targets))
        fig, ax = plt.subplots(figsize=(6, 6))
        disp.plot(ax=ax, cmap='Blues', values_format='d')
        plt.title("Confusion Matrix")
        plt.show()

        binary_targets = (np.asarray(self.targets) >= 2).astype(np.uint8)
        binary_preds = (np.asarray(self.preds) >= 2).astype(np.uint8)
        binary_cm = sklearn.metrics.confusion_matrix(binary_targets, binary_preds)
        print(binary_cm)
        self.trainer.logger.experiment.log_confusion_matrix(matrix=binary_cm, labels=[0, 1], title="Binary",
                                                            file_name="confusion-matrix_binary.json")

    def save_model(self):
        if self.save_model_path:
            sd = self.state_dict()
            torch.save(sd, self.save_model_path)
