import time
from typing import Any
import copy

import torch
import torch.nn as nn
from torch import Tensor

from rl4echo.RLmodule_3D import RLmodule3D
import torch.nn.functional as F


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
        b_img, b_gt, b_use_gt = batch['img'].squeeze(0), batch['approx_gt'].squeeze(0), batch['use_gt'].squeeze(0)
        print( batch['image_meta_dict'].get("case_identifier")[0])
        # get actions, log_probs, rewards, etc from pi (stays constant for all steps k)
        prev_actions, prev_log_probs, prev_rewards = self.rollout(b_img, b_gt, b_use_gt)
        num_rewards = len(prev_rewards)

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

    def ttoverfit(self, batch_image, num_iter=8, **kwargs):
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

        # add_plasticity(self.actor.actor.net)
        # reset_plasticity(self.actor.actor.net)

        # # freeze all main weights
        # for p in self.actor.actor.net.parameters():
        #     p.requires_grad = False
        #
        # # unfreeze only plastic params
        # plastic_params = get_plastic_params(self.actor.actor.net)
        # for p in plastic_params:
        #     p.requires_grad = True

        # for name, p in self.actor.actor.net.named_parameters():
        #     print(name, p.requires_grad)
        # optimizer just for plastic params
        # opt_plastic = torch.optim.Adam(plastic_params, lr=1e-3)

        self.divergence_coeff = 0.01
        self.entropy_coef = 0.1
        with torch.enable_grad():
            batch_image = batch_image.requires_grad_(True)
            # split up the input test sequence into smaller chunks
            split_batch_images = list(torch.split(batch_image, 4, dim=-1))
            # make sure last chunk is same size
            if split_batch_images[-1].shape[-1] != 4:
                split_batch_images[-1] = batch_image[..., -4:]

            opt_net, _ = self.configure_optimizers()
            best_reward = 0
            best_i = 0
            best_params = None
            for i in range(num_iter+1):
                sum_chunk_reward = 0
                opt_net.zero_grad()
                # opt_plastic.zero_grad()
                lowest_frame_reward = 1.0
                for chunk in split_batch_images:
                    chunk = chunk.detach().requires_grad_(True)
                    prev_actions, prev_log_probs, prev_rewards = self.rollout(chunk, None, None)

                    sum_chunk_reward += prev_rewards[0].mean()
                    lowest_frame_reward = min(prev_rewards[0].mean(axis=(0, 1, 2)).min().item(), lowest_frame_reward)

                    loss, critic_loss, _ = self.compute_policy_loss((chunk, prev_actions,
                                                                     prev_rewards[0],
                                                                     prev_log_probs, None, None))

                    # opt_net.zero_grad()
                    loss = loss / len(split_batch_images)
                    self.manual_backward(loss)

                print("\n", i, (sum_chunk_reward / len(split_batch_images)), lowest_frame_reward)
                if lowest_frame_reward > best_reward: #(sum_chunk_reward / len(split_batch_images)) > best_reward:
                    best_i = i
                    best_reward = lowest_frame_reward #(sum_chunk_reward / len(split_batch_images))
                    best_params = copy.deepcopy(self.actor.actor.net.state_dict())

                # update after checking, as current policy was used for calculating the reward
                if "32" in self.trainer.precision:
                    nn.utils.clip_grad_norm_(self.actor.actor.parameters(), 0.5)
                opt_net.step()
                # opt_plastic.step()
            self.actor.actor.net.load_state_dict(best_params)
            print(best_i)
        self.eval()