import torch
import torch.nn as nn
from torch.nn import functional as F

from utils import helpers as utl
from torchkit import pytorch_utils as ptu


class RNNEncoder(nn.Module):
    # q(m|s0,a0,r1,s1,...)
    def __init__(
        self,
        # network size
        layers_before_gru=(),
        hidden_size=64,
        layers_after_gru=(),
        task_embedding_size=32,
        # actions, states, rewards
        action_size=2,
        action_embed_size=10,
        state_size=2,
        state_embed_size=10,
        reward_size=1,
        reward_embed_size=5,
        #
        distribution="gaussian",
    ):
        super(RNNEncoder, self).__init__()

        self.task_embedding_size = task_embedding_size
        self.hidden_size = hidden_size

        if distribution == "gaussian":
            self.reparameterise = self._sample_gaussian
        else:
            raise NotImplementedError

        # embed action, state, reward (Feed-forward layers first)
        self.state_encoder = utl.FeatureExtractor(state_size, state_embed_size, F.relu)
        self.action_encoder = utl.FeatureExtractor(
            action_size, action_embed_size, F.relu
        )
        self.reward_encoder = utl.FeatureExtractor(
            reward_size, reward_embed_size, F.relu
        )

        # fully connected layers before the recurrent cell
        curr_input_size = action_embed_size + state_embed_size + reward_embed_size
        self.fc_before_gru = nn.ModuleList([])
        for i in range(len(layers_before_gru)):
            self.fc_before_gru.append(nn.Linear(curr_input_size, layers_before_gru[i]))
            curr_input_size = layers_before_gru[i]

        # recurrent unit
        self.gru = nn.GRU(
            input_size=curr_input_size,
            hidden_size=hidden_size,
            num_layers=1,
        )

        for name, param in self.gru.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param)

        # fully connected layers after the recurrent cell
        curr_input_size = hidden_size
        self.fc_after_gru = nn.ModuleList([])
        for i in range(len(layers_after_gru)):
            self.fc_after_gru.append(nn.Linear(curr_input_size, layers_after_gru[i]))
            curr_input_size = layers_after_gru[i]

        # output layer
        self.fc_mu = nn.Linear(curr_input_size, task_embedding_size)
        self.fc_logvar = nn.Linear(curr_input_size, task_embedding_size)

    def _sample_gaussian(self, mu, logvar, num=None):
        # TODO: not clip logvar?
        if num is None:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            if logvar.shape[0] > 1:
                mu = mu.unsqueeze(0)
                logvar = logvar.unsqueeze(0)
            if logvar.dim() > 2:  # if 3 dims, first must be 1
                assert logvar.shape[0] == 1, "error in dimensions!"
                std = torch.exp(0.5 * logvar).repeat(num, 1, 1)
                eps = torch.randn_like(std)
                mu = mu.repeat(num, 1, 1)
            else:  # (1, embed_size) -> repeat size (num, 1)
                std = torch.exp(0.5 * logvar).repeat(num, 1)  # (num, embed_size)
                eps = torch.randn_like(std)
                mu = mu.repeat(num, 1)
            return eps.mul(std).add_(mu)

    def reset_hidden(self, hidden_state, reset_task):
        if hidden_state.dim() != reset_task.dim():
            if reset_task.dim() == 2:
                reset_task = reset_task.unsqueeze(0)  # (1,1,1)
            elif reset_task.dim() == 1:
                reset_task = reset_task.unsqueeze(0).unsqueeze(2)
        hidden_state = hidden_state * (1 - reset_task)  # (1,1,embed_size)
        return hidden_state

    def prior(self, batch_size, sample=True):
        # we reset the initial hidden state (and thus initial output) as zeros
        # hidden state: (layers=1, B, hidden_size)
        # NOTE: no need to require_grad since we don't optimize the initial h0
        hidden_state = ptu.zeros((1, batch_size, self.hidden_size)).float()

        h = hidden_state  # initial output is exactly initial hidden state
        # forward through fully connected layers after GRU
        for i in range(len(self.fc_after_gru)):
            h = F.relu(self.fc_after_gru[i](h))

        # outputs
        task_mean = self.fc_mu(h)  # (1, B, embed_size)
        task_logvar = self.fc_logvar(h)  # (1, B, embed_size)
        if sample:
            task_sample = self.reparameterise(task_mean, task_logvar)
        else:
            task_sample = task_mean

        return task_sample, task_mean, task_logvar, hidden_state

    def forward(
        self, actions, states, rewards, hidden_state, return_prior=False, sample=True
    ):
        """
        For rollout (one-step one-batch prediction): Actions, states, rewards should be [1, dim]
        hidden state is provided and return prior is False
        return task_embeddings in [1, embed_size] and next hidden state in [1, 1, hidden_size]

        For training vae (total-step full-batch): Actions, states, rewards should be [n, B, dim]
        hidden state is None and return prior is True (so with prior)
        return task_embeddings in [n+1, B, embed_size] and all hidden states in [n+1, B, hidden_size]
        """
        assert (hidden_state is None and return_prior == True) or (
            hidden_state is not None and return_prior == False
        )

        # shape should be: (sequence_len, batch_size, dim)
        if actions.dim() != 3:  # add batch_dim = 1
            actions = actions.unsqueeze(dim=1)
            states = states.unsqueeze(dim=1)
            rewards = rewards.unsqueeze(dim=1)

        if hidden_state is not None:
            # if the sequence_len is one, this will add a dimension at dim 0 (otherwise will be the same)
            hidden_state = hidden_state.reshape((-1, *hidden_state.shape[-2:]))

        if return_prior:
            # if hidden state is none, start with the prior
            prior_sample, prior_mean, prior_logvar, prior_hidden_state = self.prior(
                actions.shape[1], sample
            )
            hidden_state = prior_hidden_state.clone()

        # extract features for states, actions, rewards
        input_a = self.action_encoder(actions)
        input_s = self.state_encoder(states)
        input_r = self.reward_encoder(rewards)
        inputs = torch.cat((input_a, input_s, input_r), dim=-1)

        # forward through fully connected layers before GRU
        for i in range(len(self.fc_before_gru)):
            inputs = F.relu(self.fc_before_gru[i](inputs))

        # GRU cell https://pytorch.org/docs/stable/generated/torch.nn.GRU.html
        # output, h_n = self.gru(input, h_0)
        # let n=seqlen. where input: (n, B, input_dim).
        # h_0 = h_n = (layers=1, B, hidden_size). the hidden state at timestep 0 and n-1
        # output: (n, B, hidden_size). the **hidden states** at last layer for each timestep
        # GRU does not have output gate like LSTM

        output, _ = self.gru(inputs, hidden_state)
        gru_h = output.clone()  # (n, B, hidden_size)
        # gru cell is nonlinear, so we don't need relu after gru

        # forward through fully connected layers after GRU
        for i in range(len(self.fc_after_gru)):
            gru_h = F.relu(self.fc_after_gru[i](gru_h))

        # outputs
        task_mean = self.fc_mu(gru_h)  # (n, B, hidden_size) -> (n, B, embed_size)
        task_logvar = self.fc_logvar(gru_h)
        if sample:
            task_sample = self.reparameterise(task_mean, task_logvar)
        else:
            task_sample = task_mean

        if return_prior:
            task_sample = torch.cat((prior_sample, task_sample))  # (n+1, B, embed_size)
            task_mean = torch.cat((prior_mean, task_mean))
            task_logvar = torch.cat((prior_logvar, task_logvar))
            output = torch.cat((prior_hidden_state, output))  # (n+1, B, hidden_size)

        if task_mean.shape[0] == 1:  # (1, B=1, embed_size) -> (B=1, embed_size)
            task_sample, task_mean, task_logvar = (
                task_sample[0],
                task_mean[0],
                task_logvar[0],
            )

        return task_sample, task_mean, task_logvar, output
