import os

import numpy as np
import cv2
import torch
from matplotlib import pyplot as plt, animation
from lightning.pytorch.loggers import CometLogger
from pytorch_lightning.loggers import TensorBoardLogger

import logging
logging.getLogger("matplotlib").setLevel(logging.WARNING)


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
        plt.imshow(img, cmap='gray')
        plt.axis("off")
        # logger.experiment.log_image(title, img, number)
        logger.experiment.log_figure("{}_{}".format(title, number), fig, step=epoch)
        plt.close()


def log_sequence(logger, img, title, number=0, img_text=None, epoch=0):
    img = img.cpu().numpy()

    # concat images together into bigger image, maximum 4
    img = np.concatenate([img[..., i].transpose((0, 2, 1)) for i in range(min(img.shape[-1], 4))], axis=2)

    if logger is None:
        return
    img = (img.copy() * (255 / max(img.max(), 1))).astype(np.uint8)
    if img_text:
        img = torch.tensor(put_text_to_image(img, img_text))

    if isinstance(logger, TensorBoardLogger):
        logger.experiment.add_image(title, img, global_step=number)
    if isinstance(logger, CometLogger):
        fig = plt.figure()
        plt.imshow(img.squeeze(0), cmap='gray')
        plt.axis("off")
        logger.experiment.log_figure("{}_{}".format(title, number), fig, step=epoch)
        plt.close()


def log_video(logger, img, title, number=0, img_text=None, epoch=0):
    img = img.cpu().numpy()

    # concat images together into bigger image
    # img = np.concatenate([img[..., i].transpose((0, 2, 1)) for i in range(min(img.shape[-1], 4))], axis=2)
    img = img.transpose((0, 2, 1, 3))
    if logger is None:
        return
    img = (img.copy() * (255 / max(img.max(), 1))).astype(np.uint8)
    if img_text:
        img = put_text_to_image(img, img_text)

    if isinstance(logger, TensorBoardLogger):
        # should have shape N T C H W
        # should have 3 color channels
        img = np.repeat(img[None,].transpose((0, 4, 1, 2, 3)), repeats=3, axis=2)
        logger.experiment.add_video(title, torch.tensor(img), global_step=number)
    if isinstance(logger, CometLogger):
        img = img.squeeze(0)
        fig, ax = plt.subplots()
        im = ax.imshow(img[..., 0], animated=True, cmap='gray')
        ax.axis("off")
        def update(i):
            im.set_array(img[..., i])
            return im,
        animation_fig = animation.FuncAnimation(fig, update, frames=img.shape[-1], interval=100, blit=True,
                                                repeat_delay=10, )
        animation_fig.save("tmp.gif")
        # Log gif file to Comet
        logger.experiment.log_video("tmp.gif", name="{}_{}".format(title, number), overwrite=True, step=epoch)
        os.remove("tmp.gif")
