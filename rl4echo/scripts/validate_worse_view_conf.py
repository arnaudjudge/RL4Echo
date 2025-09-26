import h5py
import numpy as np
import scipy.ndimage
import skimage
from matplotlib import pyplot as plt, animation
from scipy import ndimage

from vital.data.camus.config import Label
from vital.utils.image.us.measure import EchoMeasure


def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


if __name__ == "__main__":

    with h5py.File("./../../3d_anatomicalreward_worse_view_conf.h5", "r") as h5:
        print(len(h5.keys()))
        for key in h5.keys():

            img = np.array(h5[key]['img']).transpose((2, 1, 0))
            pred = np.array(h5[key]['pred']).transpose((2, 1, 0))
            gt = np.array(h5[key]['gt']).transpose((2, 1, 0))
            reward = np.array(h5[key]['reward_map'][0]).transpose((2, 1, 0))

            f, ax = plt.subplots(1, 3)
            im0 = ax[0].imshow(img[0], animated=True)
            im1 = ax[1].imshow(img[0], animated=True)
            ov1 = ax[1].imshow(pred[0], alpha=0.5, animated=True)

            im2 = ax[2].imshow(reward[0], animated=True)


            anim_running = True

            def onClick(event):
                global anim_running
                if anim_running:
                    animation_fig.event_source.stop()
                    anim_running = False
                else:
                    animation_fig.event_source.start()
                    anim_running = True


            # Animation update function here
            def update(i):
                im0.set_array(img[i])
                im1.set_array(img[i])
                ov1.set_array(pred[i])
                im2.set_array(reward[i])
                ax[0].set_title(i)

                return im1, ov1, im2

            f.canvas.mpl_connect('button_press_event', onClick)
            animation_fig = animation.FuncAnimation(f, update, frames=len(img), interval=300, blit=False,
                                                    repeat_delay=1000, )
            animation_fig.save("animation.gif")
            plt.show()



