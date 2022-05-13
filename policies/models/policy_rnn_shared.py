""" NOTE: Deprecated due to poor performance!!!
"""

import torch
import numpy as np
from copy import deepcopy
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import Adam
from utils import helpers as utl
from torchkit.constant import *
from torchkit.networks import FlattenMlp
from torchkit.actor import DeterministicPolicy, TanhGaussianPolicy
import torchkit.pytorch_utils as ptu
from utils import logger


class ModelFreeOffPolicy_Shared_RNN(nn.Module):
    """
    RNN TD3/SAC (Recurrent Policy) with shared RNN
    the input trajectory include obs,
            and/or action (action_embedding_size != 0),
            and/or reward (reward_embedding_size != 0).
    depends on the task where partially observation is
    """

    ARCH = "memory"

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

        assert algo in [TD3_name, SAC_name]
        self.algo = algo

        ### Build Model
        ## 1. embed action, state, reward (Feed-forward layers first)
        self.state_encoder = utl.FeatureExtractor(obs_dim, state_embedding_size, F.relu)
        self.action_encoder = utl.FeatureExtractor(
            action_dim, action_embedding_size, F.relu
        )
        self.reward_encoder = utl.FeatureExtractor(1, reward_embedding_size, F.relu)

        ## 2. build RNN model
        rnn_input_size = (
            action_embedding_size + state_embedding_size + reward_embedding_size
        )
        self.rnn_hidden_size = rnn_hidden_size

        assert encoder in [LSTM_name, GRU_name]
        self.encoder = encoder
        self.num_layers = 1  # TODO as free param
        self.rnn = RNNs[encoder](
            input_size=rnn_input_size,
            hidden_size=self.rnn_hidden_size,
            num_layers=self.num_layers,
            batch_first=False,
            bias=True,
        )

        for name, param in self.rnn.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param)

        ## 3. build actor-critic
        # build another obs+act branch
        self.current_state_action_encoder = utl.FeatureExtractor(
            obs_dim + action_dim, rnn_input_size, F.relu
        )

        # q-value networks
        self.qf1 = FlattenMlp(
            input_size=self.rnn_hidden_size + rnn_input_size,
            output_size=1,
            hidden_sizes=dqn_layers,
        )
        self.qf2 = FlattenMlp(
            input_size=self.rnn_hidden_size + rnn_input_size,
            output_size=1,
            hidden_sizes=dqn_layers,
        )
        # target networks NOTE: this may be the cause that q-learning is very unstable
        self.qf1_target = deepcopy(self.qf1)
        self.qf2_target = deepcopy(self.qf2)

        # build another obs branch
        self.current_state_encoder = utl.FeatureExtractor(
            obs_dim, state_embedding_size, F.relu
        )

        # policy networks
        if self.algo == TD3_name:
            self.policy = DeterministicPolicy(
                obs_dim=self.rnn_hidden_size + state_embedding_size,
                action_dim=self.action_dim,
                hidden_sizes=policy_layers,
            )
        else:
            self.policy = TanhGaussianPolicy(
                obs_dim=self.rnn_hidden_size + state_embedding_size,
                action_dim=self.action_dim,
                hidden_sizes=policy_layers,
            )
        # use joint optimizer
        #  to avoid forward rnn twice for actor and critic
        #  also exclude q targets
        self.optimizer = Adam(
            [
                *self.state_encoder.parameters(),
                *self.action_encoder.parameters(),
                *self.reward_encoder.parameters(),
                *self.rnn.parameters(),
                *self.current_state_action_encoder.parameters(),
                *self.current_state_encoder.parameters(),
                *self.qf1.parameters(),
                *self.qf2.parameters(),
                *self.policy.parameters(),
            ],
            lr=lr,
        )

        if self.algo == TD3_name:
            # NOTE: td3 has a target policy (actor)
            self.policy_target = deepcopy(self.policy)
            self.exploration_noise = exploration_noise
            self.target_noise = target_noise
            self.target_noise_clip = target_noise_clip

        else:
            ## automatic entropy coefficient tuning
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

        # logger.log(self)

    def get_hidden_states(
        self, prev_actions, rewards, observs, initial_internal_state=None
    ):
        # all the input have the shape of (T+1, B, *)
        # get embedding of initial transition
        input_a = self.action_encoder(prev_actions)
        input_r = self.reward_encoder(rewards)
        input_s = self.state_encoder(observs)
        inputs = torch.cat((input_a, input_r, input_s), dim=-1)

        # feed into RNN: output (T+1, B, hidden_size)
        if initial_internal_state is None:  # initial_internal_state is zeros
            output, _ = self.rnn(inputs)
            return output
        else:  # useful for one-step rollout
            output, current_internal_state = self.rnn(inputs, initial_internal_state)
            return output, current_internal_state

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

        ### 1. get hidden/belief states of the whole/sub trajectories, aligned with observs
        # return the hidden states (T+1, B, dim)
        hidden_states = self.get_hidden_states(
            prev_actions=actions, rewards=rewards, observs=observs
        )

        obs_embeds = self.current_state_encoder(observs)  # (T+1, B, dim)
        joint_policy_embeds = torch.cat(
            (hidden_states, obs_embeds), dim=-1
        )  # (T+1, B, dim)

        ### 2. Critic loss
        # Q^tar(h(t+1), pi(h(t+1))) + H[pi(h(t+1))]
        with torch.no_grad():
            # first next_actions from target/current policy, (T+1, B, dim) including reaction to last obs
            # new_next_actions: (T+1, B, dim), new_next_log_probs: (T+1, B, 1)
            if self.algo == TD3_name:
                new_next_actions = self.policy_target(joint_policy_embeds)
                action_noise = (
                    torch.randn_like(new_next_actions) * self.target_noise
                ).clamp(-self.target_noise_clip, self.target_noise_clip)
                new_next_actions = (new_next_actions + action_noise).clamp(
                    -1, 1
                )  # NOTE
            else:
                new_next_actions, _, _, new_next_log_probs = self.policy(
                    joint_policy_embeds, return_log_prob=True
                )

            obs_act_embeds = self.current_state_action_encoder(
                torch.cat((observs, new_next_actions), dim=-1)
            )  # (T+1, B, dim)
            joint_q_embeds = torch.cat(
                (hidden_states, obs_act_embeds), dim=-1
            )  # (T+1, B, dim)

            next_q1 = self.qf1_target(joint_q_embeds)  # return (T, B, 1)
            next_q2 = self.qf2_target(joint_q_embeds)
            min_next_q_target = torch.min(next_q1, next_q2)

            # min_next_q_target (T+1, B, 1)
            if self.algo == SAC_name:
                min_next_q_target += self.alpha_entropy * (-new_next_log_probs)

            q_target = rewards + (1.0 - dones) * self.gamma * min_next_q_target
            q_target = q_target[1:]  # (T, B, 1)

        # Q(h(t), a(t)) (T, B, 1)
        # current_actions does NOT include last obs's action
        curr_obs_act_embeds = self.current_state_action_encoder(
            torch.cat((observs[:-1], actions[1:]), dim=-1)
        )  # (T, B, dim)
        # 3. joint embeds
        curr_joint_q_embeds = torch.cat(
            (hidden_states[:-1], curr_obs_act_embeds), dim=-1
        )  # (T, B, dim)

        # masked Bellman error: masks (T,B,1) ignore the invalid error
        # this is not equal to masks * q1_pred, cuz the denominator in mean()
        # 	should depend on masks > 0.0, not a constant B*T
        q1_pred = self.qf1(curr_joint_q_embeds)
        q2_pred = self.qf2(curr_joint_q_embeds)
        q1_pred, q2_pred = q1_pred * masks, q2_pred * masks
        q_target = q_target * masks

        qf1_loss = ((q1_pred - q_target) ** 2).sum() / num_valid  # TD error
        qf2_loss = ((q2_pred - q_target) ** 2).sum() / num_valid  # TD error

        ### 3. Actor loss
        # Q(h(t), pi(h(t))) + H[pi(h(t))]
        # new_actions: (T+1, B, dim)
        if self.algo == TD3_name:
            new_actions = self.policy(joint_policy_embeds)
        else:
            new_actions, _, _, new_log_probs = self.policy(
                joint_policy_embeds, return_log_prob=True
            )

        new_obs_act_embeds = self.current_state_action_encoder(
            torch.cat((observs, new_actions), dim=-1)
        )  # (T+1, B, dim)
        new_joint_q_embeds = torch.cat(
            (hidden_states, new_obs_act_embeds), dim=-1
        )  # (T+1, B, dim)

        q1 = self.qf1(new_joint_q_embeds)
        q2 = self.qf2(new_joint_q_embeds)
        min_q_new_actions = torch.min(q1, q2)  # (T+1,B,1)

        policy_loss = -min_q_new_actions
        if self.algo == SAC_name:  # Q(h(t), pi(h(t))) + H[pi(h(t))]
            policy_loss += self.alpha_entropy * new_log_probs

        policy_loss = policy_loss[:-1]  # (T,B,1) remove the last obs
        policy_loss = (policy_loss * masks).sum() / num_valid

        ### 4. update
        total_loss = 0.5 * (qf1_loss + qf2_loss) + policy_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        ### 5. soft update
        self.soft_target_update()

        ### 6. update alpha
        if self.algo == SAC_name:
            # extract valid log_probs
            with torch.no_grad():
                current_log_probs = (new_log_probs[:-1] * masks).sum() / num_valid
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
        if self.algo == SAC_name:
            outputs.update(
                {"policy_entropy": -current_log_probs, "alpha": self.alpha_entropy}
            )
        return outputs

    def soft_target_update(self):
        ptu.soft_update_from_to(self.qf1, self.qf1_target, self.tau)
        ptu.soft_update_from_to(self.qf2, self.qf2_target, self.tau)
        if self.algo == TD3_name:
            ptu.soft_update_from_to(self.policy, self.policy_target, self.tau)

    def report_grad_norm(self):
        return {
            "rnn_grad_norm": utl.get_grad_norm(self.rnn),
            "q_grad_norm": utl.get_grad_norm(self.qf1),
            "pi_grad_norm": utl.get_grad_norm(self.policy),
        }

    def update(self, batch):
        # all are 3D tensor (T,B,dim)
        actions, rewards, dones = batch["act"], batch["rew"], batch["term"]
        _, batch_size, _ = actions.shape
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

    @torch.no_grad()
    def get_initial_info(self):
        # here we assume batch_size = 1

        ## here we set the ndim = 2 for action and reward for compatibility
        prev_action = ptu.zeros((1, self.action_dim)).float()
        reward = ptu.zeros((1, 1)).float()

        hidden_state = ptu.zeros((self.num_layers, 1, self.rnn_hidden_size)).float()
        if self.encoder == GRU_name:
            internal_state = hidden_state
        else:
            cell_state = ptu.zeros((self.num_layers, 1, self.rnn_hidden_size)).float()
            internal_state = (hidden_state, cell_state)

        return prev_action, reward, internal_state

    @torch.no_grad()
    def act(
        self,
        prev_internal_state,
        prev_action,
        reward,
        obs,
        deterministic=False,
        return_log_prob=False,
    ):
        # for evaluation (not training), so no target actor, and T = 1
        # a function that generates action, works like a pytorch module
        prev_action = prev_action.unsqueeze(0)  # (1, B, dim)
        reward = reward.unsqueeze(0)  # (1, B, 1)
        obs = obs.unsqueeze(0)  # (1, B, 1)

        # 1. get hidden state and current internal state
        ## NOTE: in T=1 step rollout (and RNN layers = 1), for GRU they are the same,
        # for LSTM, current_internal_state also includes cell state, i.e.
        # hidden state: (1, B, dim)
        # current_internal_state: (1, B, dim) or ((1, B, dim), (1, B, dim))
        hidden_state, current_internal_state = self.get_hidden_states(
            prev_actions=prev_action,
            rewards=reward,
            observs=obs,
            initial_internal_state=prev_internal_state,
        )
        # 2. another branch for current obs
        curr_embed = self.current_state_encoder(obs)  # (1, B, dim)

        # 3. joint embed
        joint_embeds = torch.cat((hidden_state, curr_embed), dim=-1)  # (1, B, dim)
        if joint_embeds.dim() == 3:
            joint_embeds = joint_embeds.squeeze(0)  # (B, dim)

        # 4. Actor head, generate action tuple
        if self.algo == TD3_name:
            mean = self.policy(joint_embeds)
            if deterministic:
                action_tuple = (mean, mean, None, None)
            else:
                action = (mean + torch.randn_like(mean) * self.exploration_noise).clamp(
                    -1, 1
                )  # NOTE
                action_tuple = (action, mean, None, None)
        else:
            # sac
            action_tuple = self.policy(
                joint_embeds, False, deterministic, return_log_prob
            )

        return action_tuple, current_internal_state
