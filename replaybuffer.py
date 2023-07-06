import collections
import numpy as np
import torch
from typing import Tuple

Experience = collections.namedtuple(
    'Experience', field_names=['img', 'mask', 'reward', 'log_prob', 'gt_mask'])


class ReplayBuffer:
    """
    Based partly on and adapted from
    https://www.pytorchlightning.ai/blog/en-lightning-reinforcement-learning-building-a-dqn-with-pytorch-lightning
    """

    def __init__(self, capacity: int) -> None:
        self.buffer = collections.deque(maxlen=capacity)
        self.buffer_priority = collections.deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.buffer)

    def append(self, experience: Experience) -> None:
        """
        Add experience to the buffer
        Args:
            experience: tuple (state, action, reward, log_prob)
        """
        self.buffer.append(experience)
        self.buffer_priority.append(experience[2].mean())

    def sample(self, batch_size: int) -> Tuple:
        #probs = torch.softmax(torch.tensor(self.buffer_priority), dim=0).cpu().numpy()
        indices = np.random.choice(len(self.buffer), batch_size, replace=False) #, p=probs)
        imgs, masks, rewards, log_probs, gt_mask = zip(*[self.buffer[idx] for idx in indices])

        return (np.array(imgs), np.array(masks), np.array(rewards),
                np.array(log_probs), np.array(gt_mask))
