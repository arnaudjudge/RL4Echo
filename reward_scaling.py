import numpy as np
import torch
import matplotlib.pyplot as plt


def scale_reward(t):
    y = 1 / (1 + torch.exp(-15 * t + 12))
    y = -torch.threshold(input=-y, threshold=-0.5, value=-1)
    return y


if __name__ == "__main__":

    x = torch.tensor(np.arange(0, 1, 0.01))
    y = scale_reward(x)
    y_np = y.detach().cpu().numpy()

    scat = torch.tensor([0.2])

    plt.figure()
    plt.plot(x, y_np)
    plt.scatter(scat, scale_reward(scat).detach().cpu().numpy())
    plt.show()


