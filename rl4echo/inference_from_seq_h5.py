import h5py
from pathlib import Path

import numpy as np
import nibabel as nib
import pandas as pd
import skimage.exposure as exp
import torch
from scipy import ndimage
from torchio import Resize, ScalarImage, LabelMap
from tqdm import tqdm

from rl4echo.utils.Metrics import is_anatomically_valid
from vital.models.segmentation.unet import UNet

from matplotlib import pyplot as plt


if __name__ == "__main__":

    resize_down = Resize((256, 256, 1))

    model = UNet(input_shape=(2, 256, 256), output_shape=(1, 256, 256))
    model.load_state_dict(
        torch.load("/data/rl_logs_3d/run_4/2/rewardnet.ckpt"))  # "/data/rl_logs_3d/run_4/3/actor.ckpt"))


    def get_segmentation(img):
        img = torch.tensor(img.astype(np.float32))
        if torch.cuda.is_available():
            model.cuda()
            img = img.cuda()
        out = torch.sigmoid(model(img) / 5.75)
        return out.cpu().detach().numpy()[0]

    with h5py.File('../3dUNC_2dMICCAIRNET.h5', 'a') as f:
        for k in f.keys():
            img = f[k]['img']
            pred = f[k]['pred']

            resize_up = Resize((img.shape[0], img.shape[1], 1))
            label_seq = [] #np.zeros_like(img, dtype=np.uint32)

            for i in range(img.shape[-1]):
                d = img[..., i] / 255
                se = pred[..., i]
                # d = exp.equalize_adapthist(d, clip_limit=0.01)
                d_small = resize_down(ScalarImage(tensor=d[None, ..., None], affine=np.diag([0.37, 0.37, 1, 0])))
                s_small = resize_down(LabelMap(tensor=se[None, ..., None], affine=np.diag([0.37, 0.37, 1, 0])))

                in_ = np.concatenate((d_small.numpy().transpose((0, 3, 1, 2)), s_small.numpy().transpose((0, 3, 1, 2))),
                                     axis=1)

                # segment and post-process
                s = get_segmentation(in_)
                # s = np.argmax(s, axis=0)

                # import matplotlib.pyplot as plt
                # plt.figure()
                # plt.imshow(s[0].T)
                #
                # plt.figure()
                # plt.imshow(se.T)
                #
                # plt.figure()
                # plt.imshow(d.T)
                # plt.show()

                # s = clean(s)

                label = ScalarImage(tensor=s[..., None], affine=d_small.affine)
                label = resize_up(label)

                label_seq += [label.numpy()[0, ..., 0]]

            label_seq = np.asarray(label_seq).transpose(1, 2, 0)
            # if f[k].get('reward_map', None):
            #     del f[k]['reward_map']
            f[k]['reward_map_otherT'] = np.asarray(label_seq)



