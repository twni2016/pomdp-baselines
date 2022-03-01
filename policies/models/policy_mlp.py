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
from torchkit.actor import DeterministicPolicy, TanhGaussianPolicy, CategoricalPolicy


class ModelFreeOffPolicy_MLP(nn.Module):
    """
    standard off-policy Markovian Policy using MLP
    including TD3 and SAC
    NOTE: it can only solve MDP problem, not POMDP
    """

    TD3_name = "td3"
    SAC_name = "sac"
    SACD_name = "sacd"

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

        assert algo in [self.TD3_name, self.SAC_name, self.SACD_name]
        self.algo = algo

        # q networks - use two network to mitigate positive bias
        if self.algo in [self.TD3_name, self.SAC_name]:
            extra_input_size = action_dim
            output_size = 1
        else:  # sac-discrete
            extra_input_size = 0
            output_size = action_dim

        self.qf1 = FlattenMlp(
            input_size=obs_dim + extra_input_size,
            output_size=output_size,
            hidden_sizes=dqn_layers,
        )
        self.qf1_optim = Adam(self.qf1.parameters(), lr=lr)

        self.qf2 = FlattenMlp(
            input_size=obs_dim + extra_input_size,
            output_size=output_size,
            hidden_sizes=dqn_layers,
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

        elif self.algo == self.SAC_name:
            self.policy = TanhGaussianPolicy(
                obs_dim=obs_dim, action_dim=action_dim, hidden_sizes=policy_layers
            )

        else:  # sac-discrete
            self.policy = CategoricalPolicy(
                obs_dim=obs_dim, action_dim=action_dim, hidden_sizes=policy_layers
            )

        if self.algo in [self.SAC_name, self.SACD_name]:
            self.automatic_entropy_tuning = automatic_entropy_tuning
            if self.automatic_entropy_tuning:
                if target_entropy is not None:
                    if self.algo == self.SAC_name:
                        self.target_entropy = float(target_entropy)
                    else:  # sac-discrete: beta * log(|A|)
                        self.target_entropy = float(target_entropy) * np.log(action_dim)
                else:
                    assert self.algo == self.SAC_name
                    self.target_entropy = -float(action_dim)
                self.log_alpha_entropy = torch.zeros(
                    1, requires_grad=True, device=ptu.device
                )
                self.alpha_entropy_optim = Adam([self.log_alpha_entropy], lr=alpha_lr)
                self.alpha_entropy = self.log_alpha_entropy.exp().detach().item()
            else:
                self.alpha_entropy = entropy_alpha

        self.policy_optim = Adam(self.policy.parameters(), lr=lr)

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
        elif self.algo == self.SAC_name:
            action, mean, log_std, log_prob = self.policy(
                obs, deterministic=deterministic, return_log_prob=return_log_prob
            )
            return action, mean, log_std, log_prob
        else:
            action, prob, log_prob = self.policy(
                obs, deterministic=deterministic, return_log_prob=return_log_prob
            )
            return action, prob, log_prob, None

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
            elif self.algo == self.SAC_name:
                next_action, _, _, next_log_prob = self.act(
                    next_obs, return_log_prob=True
                )
            else:
                _, next_prob, next_log_prob, _ = self.act(
                    next_obs, return_log_prob=True
                )  # (B, A), (B, A)

            if self.algo in [self.TD3_name, self.SAC_name]:
                next_q1 = self.qf1_target(next_obs, next_action)  # (B, 1)
                next_q2 = self.qf2_target(next_obs, next_action)
            else:
                next_q1 = self.qf1_target(next_obs)  # (B, A)
                next_q2 = self.qf2_target(next_obs)

            min_next_q_target = torch.min(next_q1, next_q2)

            if self.algo in [self.SAC_name, self.SACD_name]:
                min_next_q_target += self.alpha_entropy * (-next_log_prob)

            if self.algo == self.SACD_name:  # E_{a'\sim \pi}[Q(s',a')], (B, 1)
                min_next_q_target = (next_prob * min_next_q_target).sum(
                    dim=-1, keepdims=True
                )

            q_target = reward + (1.0 - done) * self.gamma * min_next_q_target

        if self.algo in [self.TD3_name, self.SAC_name]:
            q1_pred = self.qf1(obs, action)
            q2_pred = self.qf2(obs, action)
        else:
            action = action.long()  # (B, 1)
            q1_pred = self.qf1(obs)
            q2_pred = self.qf2(obs)
            q1_pred = q1_pred.gather(dim=-1, index=action)
            q2_pred = q2_pred.gather(dim=-1, index=action)

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
        elif self.algo == self.SAC_name:
            new_action, _, _, log_prob = self.act(obs, return_log_prob=True)
        else:
            _, new_prob, log_prob, _ = self.act(obs, return_log_prob=True)

        if self.algo in [self.TD3_name, self.SAC_name]:
            q1 = self.qf1(obs, new_action)
            q2 = self.qf2(obs, new_action)
        else:
            q1 = self.qf1(obs)
            q2 = self.qf2(obs)

        min_q_new_actions = torch.min(q1, q2)

        policy_loss = -min_q_new_actions
        if self.algo in [self.SAC_name, self.SACD_name]:
            policy_loss += self.alpha_entropy * log_prob

        if self.algo == self.SACD_name:  # E_{a\sim \pi}[Q(s,a)]
            policy_loss = (new_prob * policy_loss).sum(axis=-1, keepdims=True)  # (B,1)

        policy_loss = policy_loss.mean()

        # update policy network
        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.algo in [self.SAC_name, self.SACD_name]:
            if self.algo == self.SACD_name:  # -> negative entropy (B, 1)
                log_prob = (new_prob * log_prob).sum(axis=-1, keepdims=True)

            current_log_prob = log_prob.mean().item()

            if self.automatic_entropy_tuning:
                alpha_entropy_loss = -self.log_alpha_entropy.exp() * (
                    current_log_prob + self.target_entropy
                )

                self.alpha_entropy_optim.zero_grad()
                alpha_entropy_loss.backward()
                self.alpha_entropy_optim.step()

                self.alpha_entropy = self.log_alpha_entropy.exp().item()

        outputs = {
            "qf1_loss": qf1_loss.item(),
            "qf2_loss": qf2_loss.item(),
            "policy_loss": policy_loss.item(),
        }
        if self.algo in [self.SAC_name, self.SACD_name]:
            outputs.update(
                {"policy_entropy": -current_log_prob, "alpha": self.alpha_entropy}
            )
        return outputs

    def soft_target_update(self):
        ptu.soft_update_from_to(self.qf1, self.qf1_target, self.tau)
        ptu.soft_update_from_to(self.qf2, self.qf2_target, self.tau)
        if self.algo == self.TD3_name:
            ptu.soft_update_from_to(self.policy, self.policy_target, self.tau)
