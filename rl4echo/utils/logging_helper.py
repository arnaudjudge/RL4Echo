import numpy as np
import cv2
import torch
from matplotlib import pyplot as plt
from lightning.pytorch.loggers import CometLogger
from pytorch_lightning.loggers import TensorBoardLogger


def put_text_to_image(img, text):
    return cv2.putText(img, "{:.3f}".format(text), (0, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (125), 2)


def log_image(logger, img, title, number=0, img_text=None, epoch=0):
    img = img.cpu().numpy()
    if logger is None:
        return
    img = (img.copy() * (255 / max(img.max(), 1))).astype(np.uint8).squeeze(0)
    if img_text:
        img = put_text_to_image(img, img_text)
    if isinstance(logger, TensorBoardLogger):
        logger.experiment.add_image(title, img[None,], global_step=number)
    if isinstance(logger, CometLogger):
        fig = plt.figure()
        plt.imshow(img, cmap='grey')
        plt.axis("off")
        # logger.experiment.log_image(title, img, number)
        logger.experiment.log_figure("{}_{}".format(title, number), fig, step=epoch)
        plt.close()


def log_sequence(logger, img, title, number=0, img_text=None):
    img = img.cpu().numpy()

    # concat images together into bigger image
    img = np.concatenate([img[..., i].transpose((0, 2, 1)) for i in range(min(img.shape[-1], 4))], axis=2)

    if logger is None:
        return
    img = (img.copy() * (255 / max(img.max(), 1))).astype(np.uint8)
    if img_text:
        img = torch.tensor(put_text_to_image(img, img_text))
    logger.experiment.add_image(title, img, number)
    