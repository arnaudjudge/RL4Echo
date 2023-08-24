import torch
from torch import nn

from actors.Actors import UnetActor, Critic, Actor


class ActorCritic(Actor):
    """
        ActorCritic actor class, evaluates actor and value function approximate
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # override critic value
        self.critic = Critic(self.critic_pretrain_path)

    def evaluate(self, imgs, actions):
        """
            Evaluate images with both actor and critic
        Args:
            imgs: (state) images to evaluate
            actions: segmentation taken over images

        Returns:
            actions (sampled), logits from actor predictions, log_probs, value function estimate from critic
        """
        logits, distribution, old_distribution = self.actor(imgs)
        log_probs = distribution.log_prob(actions)

        if old_distribution:
            old_log_probs = old_distribution.log_prob(actions).detach()
        else:
            old_log_probs = log_probs.detach()

        actions = distribution.sample()
        entropy = distribution.entropy()

        # unsqueeze to match shape to multiply with logprobs
        v = self.critic(imgs).unsqueeze(-1).unsqueeze(-1)

        return actions, logits, log_probs, entropy, v, old_log_probs
