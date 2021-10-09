"""
Based on https://github.com/pranz24/pytorch-soft-actor-critic
"""

import os
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import torchkit.pytorch_utils as ptu
from torchkit.networks import FlattenMlp
from torchkit.continous_actor import DeterministicPolicy, TanhGaussianPolicy


class ModelFreeOffPolicy_MLP(nn.Module):
    """
    standard off-policy Markovian Policy using MLP
    including TD3 and SAC
    NOTE: it can only solve MDP problem, not POMDP
    """

    TD3_name = "td3"
    SAC_name = "sac"

    def __init__(
        self,
        obs_dim,
        action_dim,
        algo,
        dqn_layers,
        policy_layers,
        lr=3e-4,
        gamma=0.99,
        tau=5e-3,
        # td3 params
        exploration_noise=0.1,
        target_noise=0.2,
        target_noise_clip=0.5,
        # sac params
        entropy_alpha=0.2,
        automatic_entropy_tuning=True,
        target_entropy=None,
        alpha_lr=3e-4,
        **kwargs
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau

        assert algo in [self.TD3_name, self.SAC_name]
        self.algo = algo

        # q networks - use two network to mitigate positive bias
        self.qf1 = FlattenMlp(
            input_size=obs_dim + action_dim, output_size=1, hidden_sizes=dqn_layers
        )
        self.qf1_optim = Adam(self.qf1.parameters(), lr=lr)

        self.qf2 = FlattenMlp(
            input_size=obs_dim + action_dim, output_size=1, hidden_sizes=dqn_layers
        )
        self.qf2_optim = Adam(self.qf2.parameters(), lr=lr)

        # target networks
        self.qf1_target = copy.deepcopy(self.qf1)
        self.qf2_target = copy.deepcopy(self.qf2)

        if self.algo == self.TD3_name:
            self.policy = DeterministicPolicy(
                obs_dim=obs_dim, action_dim=action_dim, hidden_sizes=policy_layers
            )
            # NOTE: td3 has a target policy (actor)
            self.policy_target = copy.deepcopy(self.policy)

            self.exploration_noise = exploration_noise
            self.target_noise = target_noise
            self.target_noise_clip = target_noise_clip

        else:  # sac: automatic entropy coefficient tuning
            self.policy = TanhGaussianPolicy(
                obs_dim=obs_dim, action_dim=action_dim, hidden_sizes=policy_layers
            )

            self.automatic_entropy_tuning = automatic_entropy_tuning
            if self.automatic_entropy_tuning:
                if target_entropy is not None:
                    self.target_entropy = float(target_entropy)
                else:
                    self.target_entropy = -float(action_dim)
                self.log_alpha_entropy = torch.zeros(
                    1, requires_grad=True, device=ptu.device
                )
                self.alpha_entropy_optim = Adam([self.log_alpha_entropy], lr=alpha_lr)
                self.alpha_entropy = self.log_alpha_entropy.exp().detach().item()
            else:
                self.alpha_entropy = entropy_alpha

        self.policy_optim = Adam(self.policy.parameters(), lr=lr)

    def forward(self, obs):
        if self.algo == self.TD3_name:
            action = self.policy(obs)
        else:
            action, _, _, _ = self.policy(obs)
        q1, q2 = self.qf1(obs, action), self.qf2(obs, action)
        return action, q1, q2

    def act(
        self, obs, deterministic=False, return_log_prob=False, use_target_policy=False
    ):
        if self.algo == self.TD3_name:
            if use_target_policy:
                mean = self.policy_target(obs)
            else:
                mean = self.policy(obs)
            if deterministic:
                return mean, mean, None, None
            # only use in exploration
            action = (mean + torch.randn_like(mean) * self.exploration_noise).clamp(
                -1, 1
            )  # NOTE
            return action, mean, None, None
        # sac
        action, mean, log_std, log_prob = self.policy(
            obs, deterministic=deterministic, return_log_prob=return_log_prob
        )
        return action, mean, log_std, log_prob

    def update(self, batch):
        obs, next_obs = batch["obs"], batch["obs2"]  # (B, dim)
        action, reward, done = batch["act"], batch["rew"], batch["term"]  # (B, dim)

        # computation of critic loss
        with torch.no_grad():
            if self.algo == self.TD3_name:
                next_action, _, _, _ = self.act(
                    next_obs, deterministic=True, use_target_policy=True
                )
                action_noise = (
                    torch.randn_like(next_action) * self.target_noise
                ).clamp(-self.target_noise_clip, self.target_noise_clip)
                next_action = (next_action + action_noise).clamp(-1, 1)  # NOTE

            else:
                next_action, _, _, next_log_prob = self.act(
                    next_obs, return_log_prob=True
                )

            next_q1 = self.qf1_target(next_obs, next_action)
            next_q2 = self.qf2_target(next_obs, next_action)
            min_next_q_target = torch.min(next_q1, next_q2)

            if self.algo == self.SAC_name:
                min_next_q_target += self.alpha_entropy * (-next_log_prob)

            q_target = reward + (1.0 - done) * self.gamma * min_next_q_target

        q1_pred = self.qf1(obs, action)
        q2_pred = self.qf2(obs, action)

        qf1_loss = F.mse_loss(q1_pred, q_target)  # TD error
        qf2_loss = F.mse_loss(q2_pred, q_target)  # TD error

        # update q networks
        self.qf1_optim.zero_grad()
        self.qf2_optim.zero_grad()
        qf1_loss.backward()
        qf2_loss.backward()
        self.qf1_optim.step()
        self.qf2_optim.step()

        # soft update
        self.soft_target_update()

        # computation of actor loss
        if self.algo == self.TD3_name:
            new_action, _, _, _ = self.act(
                obs, deterministic=True, use_target_policy=False
            )
        else:
            new_action, _, _, log_prob = self.act(obs, return_log_prob=True)
        min_q_new_actions = self._min_q(obs, new_action)

        policy_loss = -min_q_new_actions
        if self.algo == self.SAC_name:
            policy_loss += self.alpha_entropy * log_prob

        policy_loss = policy_loss.mean()

        # update policy network
        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.algo == self.SAC_name and self.automatic_entropy_tuning:
            alpha_entropy_loss = -(
                self.log_alpha_entropy.exp() * (log_prob + self.target_entropy).detach()
            ).mean()

            self.alpha_entropy_optim.zero_grad()
            alpha_entropy_loss.backward()
            self.alpha_entropy_optim.step()

            self.alpha_entropy = self.log_alpha_entropy.exp().detach().item()

        outputs = {
            "qf1_loss": qf1_loss.item(),
            "qf2_loss": qf2_loss.item(),
            "policy_loss": policy_loss.item(),
        }
        if self.algo == self.SAC_name:
            outputs.update(
                {"policy_entropy": -log_prob.mean().item(), "alpha": self.alpha_entropy}
            )
        return outputs

    def _min_q(self, obs, action):
        q1 = self.qf1(obs, action)
        q2 = self.qf2(obs, action)
        min_q = torch.min(q1, q2)
        return min_q

    def soft_target_update(self):
        ptu.soft_update_from_to(self.qf1, self.qf1_target, self.tau)
        ptu.soft_update_from_to(self.qf2, self.qf2_target, self.tau)
        if self.algo == self.TD3_name:
            ptu.soft_update_from_to(self.policy, self.policy_target, self.tau)
