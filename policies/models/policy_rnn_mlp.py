""" 
Experimental code
"""

import torch
import numpy as np
from copy import deepcopy
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import Adam
from utils import helpers as utl
import torchkit.pytorch_utils as ptu
from torchkit.recurrent_critic import Critic_RNN
from torchkit.recurrent_actor import Actor_RNN
from torchkit.actor import DeterministicPolicy, TanhGaussianPolicy, CategoricalPolicy
from utils import logger


class ModelFreeOffPolicy_RNN_MLP(nn.Module):
    """
    Critic uses RNN, while Actor uses MLP.
    In other word, history-dependent Q value function,
        while Markov policy.
    It may be more effective on some special cases of POMDPs,
        where the reward is history-dependent, but Markov actor
        is sufficient to solve the task.
    """

    ARCH = "memory-markov"

    TD3_name = Actor_RNN.TD3_name
    SAC_name = Actor_RNN.SAC_name
    SACD_name = Actor_RNN.SACD_name

    def __init__(
        self,
        obs_dim,
        action_dim,
        encoder,
        algo,
        action_embedding_size,
        state_embedding_size,
        reward_embedding_size,
        rnn_hidden_size,
        dqn_layers,
        policy_layers,
        rnn_num_layers=1,
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
        # pixel obs
        image_encoder_fn=lambda: None,
        **kwargs
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau

        assert algo in [self.TD3_name, self.SAC_name, self.SACD_name]
        self.algo = algo

        # Critics
        self.critic = Critic_RNN(
            obs_dim,
            action_dim,
            encoder,
            algo,
            action_embedding_size,
            state_embedding_size,
            reward_embedding_size,
            rnn_hidden_size,
            dqn_layers,
            rnn_num_layers,
            image_encoder=image_encoder_fn(),  # separate weight
        )

        # target networks
        self.critic_target = deepcopy(self.critic)

        # Actor
        if self.algo == self.TD3_name:
            self.actor = DeterministicPolicy(
                obs_dim=obs_dim,
                action_dim=action_dim,
                hidden_sizes=policy_layers,
                image_encoder=image_encoder_fn(),  # separate weight
            )

            # NOTE: td3 has a target policy (actor)
            self.actor_target = deepcopy(self.actor)
            self.exploration_noise = exploration_noise
            self.target_noise = target_noise
            self.target_noise_clip = target_noise_clip

        elif self.algo == self.SAC_name:
            self.actor = TanhGaussianPolicy(
                obs_dim=obs_dim,
                action_dim=action_dim,
                hidden_sizes=policy_layers,
                image_encoder=image_encoder_fn(),  # separate weight
            )

        else:  # sac-discrete
            self.actor = CategoricalPolicy(
                obs_dim=obs_dim,
                action_dim=action_dim,
                hidden_sizes=policy_layers,
                image_encoder=image_encoder_fn(),  # separate weight
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

        # use separate optimizers
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=lr)

    def act(
        self, obs, deterministic=False, return_log_prob=False, use_target_policy=False
    ):
        if self.algo == self.TD3_name:
            if use_target_policy:
                mean = self.actor_target(obs)
            else:
                mean = self.actor(obs)
            if deterministic:
                return mean, mean, None, None
            # only use in exploration
            action = (mean + torch.randn_like(mean) * self.exploration_noise).clamp(
                -1, 1
            )  # NOTE
            return action, mean, None, None
        elif self.algo == self.SAC_name:
            action, mean, log_std, log_prob = self.actor(
                obs, deterministic=deterministic, return_log_prob=return_log_prob
            )
            return action, mean, log_std, log_prob
        else:
            action, prob, log_prob = self.actor(
                obs, deterministic=deterministic, return_log_prob=return_log_prob
            )
            return action, prob, log_prob, None

    def forward(self, actions, rewards, observs, dones, masks):
        """
        For actions a, rewards r, observs o, dones d: (T+1, B, dim)
                where for each t in [0, T], take action a[t], then receive reward r[t], done d[t], and next obs o[t]
                the hidden state h[t](, c[t]) = RNN(h[t-1](, c[t-1]), a[t], r[t], o[t])
                specially, a[0]=r[0]=d[0]=h[0]=c[0]=0.0, o[0] is the initial obs

        The loss is still on the Q value Q(h[t], a[t]) with real actions taken, i.e. t in [1, T]
                based on Masks (T, B, 1)
        """
        assert (
            actions.dim()
            == rewards.dim()
            == dones.dim()
            == observs.dim()
            == masks.dim()
            == 3
        )
        assert (
            actions.shape[0]
            == rewards.shape[0]
            == dones.shape[0]
            == observs.shape[0]
            == masks.shape[0] + 1
        )
        num_valid = torch.clamp(masks.sum(), min=1.0)  # as denominator of loss

        ### 1. Critic loss
        # Q^tar(h(t+1), pi(h(t+1))) + H[pi(h(t+1))]
        with torch.no_grad():
            # first next_actions from target/current policy, (T+1, B, dim)
            if self.algo == self.TD3_name:
                new_actions, _, _, _ = self.act(
                    observs, deterministic=True, use_target_policy=True
                )
                action_noise = (
                    torch.randn_like(new_actions) * self.target_noise
                ).clamp(-self.target_noise_clip, self.target_noise_clip)
                new_actions = (new_actions + action_noise).clamp(-1, 1)  # NOTE
            elif self.algo == self.SAC_name:
                new_actions, _, _, new_log_probs = self.act(
                    observs, return_log_prob=True
                )
            else:
                _, new_probs, new_log_probs, _ = self.act(observs, return_log_prob=True)

            next_q1, next_q2 = self.critic_target(
                prev_actions=actions,
                rewards=rewards,
                observs=observs,
                current_actions=new_actions
                if self.algo in [self.TD3_name, self.SAC_name]
                else new_probs,
            )  # (T+1, B, 1 or A)

            min_next_q_target = torch.min(next_q1, next_q2)

            if self.algo in [self.SAC_name, self.SACD_name]:
                min_next_q_target += self.alpha_entropy * (
                    -new_log_probs
                )  # (T+1, B, 1 or A)

            if self.algo == self.SACD_name:  # E_{a'\sim \pi}[Q(h',a')], (T+1, B, 1)
                min_next_q_target = (new_probs * min_next_q_target).sum(
                    dim=-1, keepdims=True
                )

            # q_target: (T, B, 1)
            q_target = (
                rewards + (1.0 - dones) * self.gamma * min_next_q_target
            )  # next q
            q_target = q_target[1:]  # (T, B, 1)

        # Q(h(t), a(t)) (T, B, 1)
        q1_pred, q2_pred = self.critic(
            prev_actions=actions,
            rewards=rewards,
            observs=observs,
            current_actions=actions[1:],
        )  # (T, B, 1 or A)

        if self.algo == self.SACD_name:
            stored_actions = actions[1:]  # (T, B, A)
            stored_actions = torch.argmax(
                stored_actions, dim=-1, keepdims=True
            )  # (T, B, 1)
            q1_pred = q1_pred.gather(
                dim=-1, index=stored_actions
            )  # (T, B, A) -> (T, B, 1)
            q2_pred = q2_pred.gather(
                dim=-1, index=stored_actions
            )  # (T, B, A) -> (T, B, 1)

        # masked Bellman error: masks (T,B,1) ignore the invalid error
        # this is not equal to masks * q1_pred, cuz the denominator in mean()
        # 	should depend on masks > 0.0, not a constant B*T
        q1_pred, q2_pred = q1_pred * masks, q2_pred * masks
        q_target = q_target * masks
        qf1_loss = ((q1_pred - q_target) ** 2).sum() / num_valid  # TD error
        qf2_loss = ((q2_pred - q_target) ** 2).sum() / num_valid  # TD error

        self.critic_optimizer.zero_grad()
        (qf1_loss + qf2_loss).backward()
        self.critic_optimizer.step()

        ### 2. Actor loss
        if self.algo == self.TD3_name:
            new_actions, _, _, _ = self.act(
                observs, deterministic=True, use_target_policy=False
            )  # (T+1, B, A)
        elif self.algo == self.SAC_name:
            new_actions, _, _, log_probs = self.act(
                observs, return_log_prob=True
            )  # (T+1, B, A)
        else:
            _, new_probs, log_probs, _ = self.act(
                observs, return_log_prob=True
            )  # (T+1, B, A)

        q1, q2 = self.critic(
            prev_actions=actions,
            rewards=rewards,
            observs=observs,
            current_actions=new_actions
            if self.algo in [self.TD3_name, self.SAC_name]
            else new_probs,
        )  # (T+1, B, 1 or A)
        min_q_new_actions = torch.min(q1, q2)  # (T+1,B,1 or A)

        policy_loss = -min_q_new_actions
        if self.algo in [
            self.SAC_name,
            self.SACD_name,
        ]:  # Q(h(t), pi(h(t))) + H[pi(h(t))]
            policy_loss += self.alpha_entropy * log_probs

        if self.algo == self.SACD_name:  # E_{a\sim \pi}[Q(h,a)]
            policy_loss = (new_probs * policy_loss).sum(
                axis=-1, keepdims=True
            )  # (T+1,B,1)

        policy_loss = policy_loss[:-1]  # (T,B,1) remove the last obs
        # masked policy_loss
        policy_loss = (policy_loss * masks).sum() / num_valid

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        ### 3. soft update
        self.soft_target_update()

        ### 4. update alpha
        if self.algo in [self.SAC_name, self.SACD_name]:
            # extract valid log_probs
            if self.algo == self.SACD_name:  # -> negative entropy (T+1, B, 1)
                log_probs = (new_probs * log_probs).sum(axis=-1, keepdims=True)
            with torch.no_grad():
                current_log_probs = (log_probs[:-1] * masks).sum() / num_valid
                current_log_probs = current_log_probs.item()

            if self.automatic_entropy_tuning:
                alpha_entropy_loss = -self.log_alpha_entropy.exp() * (
                    current_log_probs + self.target_entropy
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
                {"policy_entropy": -current_log_probs, "alpha": self.alpha_entropy}
            )
        return outputs

    def soft_target_update(self):
        ptu.soft_update_from_to(self.critic, self.critic_target, self.tau)
        if self.algo == self.TD3_name:
            ptu.soft_update_from_to(self.actor, self.actor_target, self.tau)

    def report_grad_norm(self):
        # may add qf1, policy, etc.
        return {
            "q_grad_norm": utl.get_grad_norm(self.critic),
            "q_rnn_grad_norm": utl.get_grad_norm(self.critic.rnn),
            "pi_grad_norm": utl.get_grad_norm(self.actor),
        }

    def update(self, batch):
        # all are 3D tensor (T,B,dim)
        actions, rewards, dones = batch["act"], batch["rew"], batch["term"]
        _, batch_size, _ = actions.shape
        if self.algo == self.SACD_name:
            # for discrete action space, convert to one-hot vectors
            actions = F.one_hot(
                actions.squeeze(-1).long(), num_classes=self.action_dim
            ).float()  # (T, B, A)

        masks = batch["mask"]
        obs, next_obs = batch["obs"], batch["obs2"]  # (T, B, dim)

        # extend observs, actions, rewards, dones from len = T to len = T+1
        observs = torch.cat((obs[[0]], next_obs), dim=0)  # (T+1, B, dim)
        actions = torch.cat(
            (ptu.zeros((1, batch_size, self.action_dim)).float(), actions), dim=0
        )  # (T+1, B, dim)
        rewards = torch.cat(
            (ptu.zeros((1, batch_size, 1)).float(), rewards), dim=0
        )  # (T+1, B, dim)
        dones = torch.cat(
            (ptu.zeros((1, batch_size, 1)).float(), dones), dim=0
        )  # (T+1, B, dim)

        return self.forward(actions, rewards, observs, dones, masks)
