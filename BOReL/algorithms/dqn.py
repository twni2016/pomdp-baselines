import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from utils import helpers as utl
from torchkit import pytorch_utils as ptu
from torch.distributions import Categorical


class DQN(nn.Module):
    def __init__(
        self,
        q_network,
        lr=None,
        eps_optim=None,
        alpha_optim=None,
        gamma=0.99,
        eps_init=1.0,
        eps_final=0.1,
        exploration_iters=1000,
        tau=5e-3,
    ):
        super().__init__()

        # the network and target network
        self.qf = q_network
        self.target_qf = q_network.copy()

        self.gamma = gamma  # discount factor
        self.eps_init = eps_init  # initial exploration parameter
        self.eps_final = eps_final  # final exploration parameter
        self.exploration_iters = (
            exploration_iters  # num iteration from eps_init to eps_final
        )
        self._set_eps(self.eps_init)  # initialize temperature
        self.tau = tau  # soft target update parameter

        # optimisers
        # self.optimizer = optim.RMSprop(q_network.parameters(), lr=lr, eps=eps_optim, alpha=alpha_optim)
        self.optimizer = optim.Adam(q_network.parameters(), lr=lr)

    def forward(self, obs):
        return self.qf(obs)

    def update(self, obs, action, reward, next_obs, done, **kwargs):
        """
        Inputs are of size (batch, dim). Performs parameters update.
        """
        self.optimizer.zero_grad()
        q_pred = self.forward(obs).gather(
            -1, action.argmax(dim=-1, keepdims=True)
        )  # get q_values at taken actions
        q_target = self.get_q_target(next_obs, reward, done)  # get target update values
        qf_loss = torch.mean((q_pred - q_target) ** 2)  # TD error
        qf_loss.backward()
        self.optimizer.step()

        # soft update of target network
        self.soft_target_update()
        return {"qf_loss": qf_loss.item()}

    def get_q_target(self, next_obs, reward, done):
        next_q_max = self.target_qf(next_obs).detach().max(dim=-1, keepdims=True)[0]
        q_target = reward + (1.0 - done) * self.gamma * next_q_max
        return q_target

    def soft_target_update(self):
        ptu.soft_update_from_to(self.qf, self.target_qf, self.tau)

    def act(self, obs, deterministic=False):
        """
            epsilon-greedy policy based on Q values
        :param obs:
        :param deterministic: whether to sample or take most likely action
        :return: action and its corresponding Q value
        """
        q_values = self.qf(obs)
        if deterministic:
            action = q_values.argmax(dim=-1, keepdims=True)
        else:  # epsilon greedy
            if random.random() <= self.eps:
                action = (
                    ptu.FloatTensor(
                        [
                            random.randrange(q_values.shape[-1])
                            for _ in range(q_values.shape[0])
                        ]
                    )
                    .long()
                    .unsqueeze(dim=-1)
                )
            else:
                action = q_values.argmax(dim=-1, keepdims=True)
        value = q_values.gather(dim=-1, index=action)
        return action, value

    def set_exploration_parameter(self, t):
        """
            set exploration based on linear schedule -- e_t = e_i + min(1, t/T)*(e_f -e_i)
        :param t: iteration
        :return:
        """
        self._set_eps(
            self.eps_init
            + min(1.0, t / self.exploration_iters) * (self.eps_final - self.eps_init)
        )

    def _set_eps(self, eps):
        self.eps = eps

    def train(self, mode=True):
        self.qf.train(mode)
        self.target_qf.train(mode)


class DoubleDQN(DQN):
    def __init__(self, q_network, **kwargs):
        super().__init__(q_network, **kwargs)

    def get_q_target(self, next_obs, reward, done):
        """get update target for q network"""
        # get q values of next obs for every action
        next_q_values = self.qf(next_obs).detach()
        # get optimal actions for next obs
        optimal_next_action = next_q_values.max(1)[1].view(-1, 1)

        # get q values from target network for every action
        next_q_target_values = self.target_qf(next_obs).detach()
        # take the next q value according to the target network at the action
        # which was chosen by the q network
        next_q_target_values = next_q_target_values.gather(
            1, optimal_next_action.type(torch.long)
        )
        targets = (
            reward + (1.0 - done) * self.gamma * next_q_target_values
        )  # r + gamma * q_max
        return targets
