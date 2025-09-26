import time
from typing import Any
import copy

import torch
import torch.nn as nn
from torch import Tensor
from torchvision.transforms.functional import adjust_contrast
import random

from rl4echo.RLmodule_3D import RLmodule3D
import torch.nn.functional as F

from matplotlib import pyplot as plt


# class PlasticLayer(nn.Module):
#     def __init__(self, num_channels):
#         super().__init__()
#         # fast-adapting gain and bias
#         self.gain = nn.Parameter(torch.ones(num_channels, device="cuda:0"))
#         self.bias = nn.Parameter(torch.zeros(num_channels, device="cuda:0"))
#
#     def forward(self, x):
#         return self.gain.view(1, -1, 1, 1, 1) * x + self.bias.view(1, -1, 1, 1, 1)
#
#     def reset(self):
#         nn.init.ones_(self.gain)
#         nn.init.zeros_(self.bias)

class PlasticLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(1.0))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        return self.gain * x + self.bias

    def reset(self):
        self.gain.data.fill_(1.0)
        self.bias.data.fill_(0.0)


def add_plasticity(module):
    """Wrap every Conv3d with a PlasticLayer right after it."""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Conv3d):
            # replace Conv3d -> Sequential(Conv3d, PlasticLayer)
            wrapped = nn.Sequential(
                child,
                PlasticLayer()
            )
            setattr(module, name, wrapped)
        else:
            add_plasticity(child)


def reset_plasticity(module):
    """Reset all PlasticLayers to identity mapping."""
    for m in module.modules():
        if isinstance(m, PlasticLayer):
            m.reset()

def get_plastic_params(model):
    return [p for m in model.modules() if isinstance(m, PlasticLayer)
            for p in [m.gain, m.bias]]

import torch.nn as nn

def get_all_layers(module, layers=None):
    """Recursively collect all leaf layers (modules without children)."""
    if layers is None:
        layers = []
    for child in module.children():
        if len(list(child.children())) == 0:
            layers.append(child)
        else:
            get_all_layers(child, layers)
    return layers

def unfreeze_layers(model, mode="first", n=3):
    """
    Freeze all params, then unfreeze first/last n leaf layers (e.g., convs).
    mode: "first", "last", or "hybrid"
    """
    # Freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # Get all "leaf" layers (like Conv3d, BatchNorm3d, etc.)
    layers = get_all_layers(model)
    # layers = [m for m in model.modules() if isinstance(m, torch.nn.Conv3d)]


    if mode == "first":
        selected = layers[:n]
    elif mode == "last":
        selected = layers[-n:]
    elif mode == "hybrid":
        selected = layers[:n] + layers[-n:]
    else:
        raise ValueError("mode must be 'first', 'last', or 'hybrid'")

    # Unfreeze selected layers
    for layer in selected:
        for param in layer.parameters():
            param.requires_grad = True

    print(f"✅ Unfrozen {mode} {n} layers "
          f"(out of {len(layers)} total). "
          f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    return model

class PPO3D(RLmodule3D):

    def __init__(self,
                 clip_value: float = 0.2,
                 k_steps_per_batch: int = 5,
                 entropy_coeff: float = 0.0,
                 divergence_coeff: float = 0.0,
                 *args: Any,
                 **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        # since optimization is done manually, this flag needs to be set
        self.automatic_optimization = False

    def training_step(self, batch: dict[str, Tensor], nb_batch):
        """
            Defines PPO training steo
            Get actions, log_probs and rewards from current policy
            Calculate and backprop losses for actor and critic K times in loop over same batch
        Args:
            batch: current mini batch of replay data
            nb_batch: batch number

        Returns:
            Training loss and log metrics or None
        """
        opt_net, opt_critic = self.optimizers()
        opt_net.zero_grad()  # do once first if not done initially in loop

        # TODO: REMOVE GT
        b_img, b_gt, b_use_gt = batch['img'].squeeze(0), batch['gt'].squeeze(0), batch['use_gt'].squeeze(0)

        print( batch['image_meta_dict'].get("case_identifier")[0])
        if batch['image_meta_dict'].get("case_identifier")[0] != 'di-30F6-5C3B-EA3A':
            return {'loss': torch.tensor(0, device=self.device) }
        # get actions, log_probs, rewards, etc from pi (stays constant for all steps k)
        prev_actions, prev_log_probs, prev_rewards = self.rollout(b_img, b_gt, b_use_gt, sample=False)
        num_rewards = len(prev_rewards)

        gt = b_gt.cpu().detach().numpy()[0]
        pred = prev_actions.cpu().detach().numpy()[0]

        from vital.utils.image.us.measure import EchoMeasure
        from skimage.measure import find_contours
        import matplotlib.pyplot as plt
        import numpy as np
        from vital.data.camus.config import Label
        from scipy import ndimage
        from scipy.ndimage import gaussian_filter
        from skimage import draw

        plt.figure()
        # gt
        contour = find_contours((gt[..., 0].T == 1).squeeze(), level=0.9)[0]
        plt.plot(contour[:, 1], contour[:, 0], c='#99FF99', zorder=1)
        contour = find_contours((gt[..., 0].T == 2).squeeze(), level=0.9)[0]
        plt.plot(contour[:, 1], contour[:, 0], c='#99FF99', zorder=1, label='Reference')

        # candidate
        contour = find_contours((pred[..., 0].T == 1).squeeze(), level=0.9)[0]
        plt.plot(contour[:, 1], contour[:, 0], c='#FF9999', zorder=1)
        contour = find_contours((pred[..., 0].T == 2).squeeze(), level=0.9)[0]
        plt.plot(contour[:, 1], contour[:, 0], c='#FF9999', zorder=1, label='Candidate')

        i = 0

        lv_points = np.asarray(
            EchoMeasure._endo_base(gt[..., i].T, lv_labels=Label.LV, myo_labels=Label.MYO))

        p = pred[..., i]
        y = np.zeros_like(gt).astype(float)

        lbl, num = ndimage.label(p != 0)
        # Count the number of elements per label
        count = np.bincount(lbl.flat)
        # Select the largest blob
        maxi = np.argmax(count[1:]) + 1
        # Remove the other blobs
        p[lbl != maxi] = 0

        p_points = np.asarray(
            EchoMeasure._endo_base(p.T, lv_labels=Label.LV, myo_labels=Label.MYO))
        a = np.zeros_like(p).astype(float)
        b = np.zeros_like(p).astype(float)

        lv_points = lv_points[np.argsort(lv_points[:, 1])]
        p_points = p_points[np.argsort(p_points[:, 1])]

        d0_sigma = (np.linalg.norm(lv_points[0] - p_points[0]) / a.shape[0] * 200)
        print(d0_sigma)
        d1_sigma = (np.linalg.norm(lv_points[1] - p_points[1]) / b.shape[0] * 200)

        spacing = [0.37, 0.37] #batch['image_meta_dict']['original_spacing'].cpu().numpy()[0, 1:3]

        # larger than 5mm
        # if (np.linalg.norm((lv_points[0] - p_points[0]) * spacing)) > 4:
        rr, cc, val = draw.line_aa(p_points[0, 1], p_points[0, 0], lv_points[0, 1], lv_points[0, 0])
        a[rr, cc] = val
        a = gaussian_filter(a, sigma=d0_sigma)
        a = (a - np.min(a)) / (np.max(a) - np.min(a))
        # if (np.linalg.norm((lv_points[1] - p_points[1]) * spacing)) > 4:
        rr, cc, val = draw.line_aa(p_points[1, 1], p_points[1, 0], lv_points[1, 1], lv_points[1, 0])
        b[rr, cc] = val
        b = gaussian_filter(b, sigma=d1_sigma)
        b = (b - np.min(b)) / (np.max(b) - np.min(b))

        y[..., i] = np.maximum(a, b)

        plt.plot([p_points[0, 1], lv_points[0, 1]],
                 [p_points[0, 0], lv_points[0, 0]],
                 color="yellow", linewidth=3, zorder=2)
        plt.plot([p_points[1, 1], lv_points[1, 1]],
                 [p_points[1, 0], lv_points[1, 0]],
                 color="yellow", linewidth=3, zorder=2)

        plt.scatter(lv_points[..., 1], lv_points[..., 0], c='#009900', zorder=3)
        plt.scatter(p_points[..., 1], p_points[..., 0], c='#990000', zorder=3)

        plt.imshow(1 - y[..., i].T, cmap='gray', vmin=0, vmax=1, aspect='auto', interpolation='none')

        ax = plt.gca()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.legend(loc='upper left')
        plt.show()



        # iterate with pi prime k times
        for k in range(self.hparams.k_steps_per_batch*num_rewards):
            # calculates training loss
            loss, critic_loss, metrics_dict = self.compute_policy_loss((b_img, prev_actions,
                                                                        prev_rewards[k % num_rewards],
                                                                        prev_log_probs, b_gt, b_use_gt))
            self.manual_backward(loss)
            if "32" in self.trainer.precision:
                nn.utils.clip_grad_norm_(self.actor.actor.parameters(), 0.5)
            # TODO: grad accumulation here???
            if k % num_rewards == (num_rewards-1):  # only step when all rewards are done, like a2c with multiple actors
                opt_net.step()
                opt_net.zero_grad()

            # TODO: should this be outside the loop? According to real algo...
            opt_critic.zero_grad()
            self.manual_backward(critic_loss)
            if "32" in self.trainer.precision:
                nn.utils.clip_grad_norm_(self.actor.critic.parameters(), 0.5)
            opt_critic.step()

            logs = {**metrics_dict,
                    **{'loss': loss,
                       'critic_loss': critic_loss,
                       }
                    }

            self.log_dict(logs, prog_bar=True)

    def compute_policy_loss(self, batch, **kwargs):
        """
            Compute unsupervised loss to maximise reward using PPO method.
        Args:
            batch: batch of images, actions, log_probs, rewards and ground truth
            sample: whether to sample from distribution or deterministic approach (mainly for val, test steps)

        Returns:
            mean loss(es) for the batch, metrics dictionary
        """
        b_img, b_actions, b_rewards, b_log_probs, b_gt, b_use_gt = batch

        _, logits, log_probs, entropy, v, old_log_probs = self.actor.evaluate(b_img, b_actions)

        v_deeps = None
        if isinstance(v, list):
            v_deeps = v
            v = v[0]

        log_pi_ratio = (log_probs - old_log_probs)
        with torch.no_grad():
            total_reward = b_rewards - (self.hparams.divergence_coeff * log_pi_ratio)
            # ignore divergence if using ground truth
            # total_reward[b_use_gt, ...] = torch.ones_like(b_rewards)[b_use_gt, ...]

            # assert b_rewards.shape == v.shape
            adv = total_reward - v

        # PPO loss
        # importance ratio
        assert b_log_probs.shape == log_probs.shape
        ratio = (log_probs - b_log_probs).exp()

        # clamp with epsilon value
        clipped = ratio.clamp(1 - self.hparams.clip_value, 1 + self.hparams.clip_value)
        surr_loss = torch.min(adv * ratio, adv * clipped)
        # surr_loss[b_use_gt, ...] = (adv * ratio)[b_use_gt, ...]

        # min trick
        loss = -surr_loss.mean() + (-self.hparams.entropy_coeff * entropy.mean())

        # Critic loss
        if b_rewards.shape != v.shape:  # if critic is resnet, use reward mean instead of pixel-wise
            b_rewards = b_rewards.mean(dim=(1, 2), keepdim=True)

        if v_deeps:
            # deep supervision
            critic_loss = nn.MSELoss()(v_deeps[0], b_rewards)
            for i, v_ in enumerate(v_deeps[1:]):
                downsampled_label = nn.functional.interpolate(b_rewards.unsqueeze(0), v_.shape[1:]).squeeze(0)
                critic_loss += 0.5 ** (i + 1) * nn.MSELoss()(v_, downsampled_label)
            c_norm = 1 / (2 - 2 ** (-len(v_deeps)))
            critic_loss = c_norm * critic_loss
        else:
            critic_loss = nn.MSELoss()(v, b_rewards)

        # metrics dict
        metrics = {
                'v': v.mean(),
                'advantage': adv.mean(),
                'reward': b_rewards.mean(),
                'log_probs': log_probs.mean(),
                'ratio': ratio.mean(),
                'approx_kl_div': log_pi_ratio.mean(),
        }

        return loss, critic_loss, metrics

    def ttoverfit(self, batch_image, num_iter=4, **kwargs):
        """
            Run a few itercvations of optimization to overfit on one test sequence in unsupervised
        Args:
            batch: batch of images
            num_iter: number of iterations of the optimization loop
        Returns:
            None, actor is modified, ready for inference on batch_image alone
        """
        self.train()
        torch.cuda.empty_cache()
        # self.actor.actor.old_net.load_state_dict(copy.deepcopy(self.actor.actor.net.state_dict()))

        # reset_plasticity(self.actor.actor.net)

        # freeze all main weights
        # plastic_params = unfreeze_layers(self.actor.actor.net, n=5)
        for name, module in self.actor.actor.net.named_modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.InstanceNorm3d)):
                for p in module.parameters():
                    p.requires_grad = True
            else:
                for p in module.parameters():
                    p.requires_grad = False

        params_to_optimize = [p for p in self.actor.actor.net.parameters() if p.requires_grad]
        print(len(params_to_optimize))
        # optimizer for only those layers
        opt_net = torch.optim.Adam(params_to_optimize, lr=0.01)


        # # unfreeze only plastic params
        # plastic_params = get_plastic_params(self.actor.actor.net)
        # for p in plastic_params:
        #     p.requires_grad = True

        # for name, p in self.actor.actor.net.named_parameters():
        #     print(name, p.requires_grad)
        # optimizer just for plastic params
        # opt_plastic = torch.optim.Adam(plastic_params, lr=1e-3)

        # opt_net, _ = self.configure_optimizers()

        augmentations = 3
        self.divergence_coeff = 0.01
        self.entropy_coef = 0.1
        with torch.enable_grad():
            batch_image = batch_image
            # split up the input test sequence into smaller chunks
            split_batch_images = list(torch.split(batch_image, 4, dim=-1))
            # make sure last chunk is same size
            if split_batch_images[-1].shape[-1] != 4:
                split_batch_images[-1] = batch_image[..., -4:]

            best_reward = 0
            best_i = 0
            best_params = None
            for i in range(num_iter+1):
                sum_chunk_reward = 0
                # opt_net.zero_grad()
                lowest_frame_reward = 1.0
                for chunk in split_batch_images:
                    chunk = chunk.detach()
                    opt_net.zero_grad()
                    avg_lowest_reward_frame = 0
                    for a in range(augmentations):
                        with torch.no_grad():
                            chunk_def = adjust_contrast(chunk.clone().permute((4, 0, 1, 2, 3)),
                                                        random.uniform(0.4, 0.8)).permute((1, 2, 3, 4, 0))
                            chunk_def += torch.randn(chunk_def.size()).to(
                                next(self.actor.actor.net.parameters()).device) * random.uniform(0.001, 0.01)
                            chunk_def /= chunk_def.max()

                        prev_actions, prev_log_probs, prev_rewards = self.rollout(chunk_def, None, None)
                        # prev_actions = self.actor.act(chunk_def, sample=True)
                        # prev_rewards = self.reward_func(prev_actions, chunk_def, None)
                        # _, _, prev_log_probs, _, _, _ = self.actor.evaluate(chunk_def, prev_action
                        # plt.figure()
                        # plt.imshow(chunk_def[0, ..., 0].T.cpu().numpy(), cmap='gray')
                        # plt.imshow(prev_actions[0, ..., 0].T.cpu().numpy(), alpha=0.3)

                        # plt.figure()
                        # plt.imshow(chunk[0, ..., 0].T.cpu().numpy(), cmap='gray')
                        # plt.show()

                        sum_chunk_reward += prev_rewards[0].mean()
                        lowest_frame_reward = min(prev_rewards[0].mean(axis=(0, 1, 2)).min().item(), lowest_frame_reward)
                        avg_lowest_reward_frame += prev_rewards[0].mean(axis=(0, 1, 2)).min().item() / augmentations

                        if i != 0:
                            loss, _, _ = self.compute_policy_loss((chunk, prev_actions,
                                                                             prev_rewards[0],
                                                                             prev_log_probs, None, None))

                            # loss = loss / len(split_batch_images) / 3
                            loss = loss / augmentations
                            self.manual_backward(loss)
                    # plt.show()
                    lowest_frame_reward = min(avg_lowest_reward_frame, lowest_frame_reward)
                    # update after checking, as current policy was used for calculating the reward
                    if i != 0:
                        if "32" in self.trainer.precision:
                            nn.utils.clip_grad_norm_(self.actor.actor.parameters(), 0.5)
                        opt_net.step()
                print(f"\n{'First, no optimization' if i == 0 else ''}", i, (sum_chunk_reward / len(split_batch_images) / augmentations), lowest_frame_reward)
                if lowest_frame_reward > best_reward: #(sum_chunk_reward / len(split_batch_images)) > best_reward:
                    best_i = i
                    best_reward = lowest_frame_reward #(sum_chunk_reward / len(split_batch_images))
                    best_params = copy.deepcopy(self.actor.actor.net.state_dict())

        self.actor.actor.net.load_state_dict(best_params)
        print(best_i)
        self.eval()