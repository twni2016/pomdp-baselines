import torch
import torch.nn as nn
from torch.nn import functional as F
from utils import helpers as utl
from torchkit.networks import FlattenMlp
import torchkit.pytorch_utils as ptu
from torchkit.recurrent_actor import Actor_RNN


class Critic_RNN(nn.Module):
    TD3_name = Actor_RNN.TD3_name
    SAC_name = Actor_RNN.SAC_name
    SACD_name = Actor_RNN.SACD_name
    LSTM_name = Actor_RNN.LSTM_name
    GRU_name = Actor_RNN.GRU_name
    RNNs = Actor_RNN.RNNs

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
        rnn_num_layers,
        **kwargs
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
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

        assert encoder in self.RNNs
        self.encoder = encoder

        self.rnn = self.RNNs[encoder](
            input_size=rnn_input_size,
            hidden_size=self.rnn_hidden_size,
            num_layers=rnn_num_layers,
            batch_first=False,
            bias=True,
        )

        for name, param in self.rnn.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param)

        if self.algo in [self.TD3_name, self.SAC_name]:
            extra_input_size = action_dim
            output_size = 1
        else:  # sac-discrete
            extra_input_size = 0
            output_size = action_dim

        ## 3. build another obs+act branch
        self.current_state_action_encoder = utl.FeatureExtractor(
            obs_dim + extra_input_size, rnn_input_size, F.relu
        )

        ## 4. build q networks
        self.qf1 = FlattenMlp(
            input_size=self.rnn_hidden_size + rnn_input_size,
            output_size=output_size,
            hidden_sizes=dqn_layers,
        )
        self.qf2 = FlattenMlp(
            input_size=self.rnn_hidden_size + rnn_input_size,
            output_size=output_size,
            hidden_sizes=dqn_layers,
        )

    def get_hidden_states(self, prev_actions, rewards, observs):
        # all the input have the shape of (T+1, B, *)
        # get embedding of initial transition
        input_a = self.action_encoder(prev_actions)
        input_r = self.reward_encoder(rewards)
        input_s = self.state_encoder(observs)
        inputs = torch.cat((input_a, input_r, input_s), dim=-1)

        # feed into RNN: output (T+1, B, hidden_size)
        output, _ = self.rnn(inputs)  # initial hidden state is zeros
        return output

    def forward(self, prev_actions, rewards, observs, current_actions):
        """
        For prev_actions a, rewards r, observs o: (T+1, B, dim)
                a[t] -> r[t], o[t]
        current_actions (or action probs for discrete actions) a': (T or T+1, B, dim)
                o[t] -> a'[t]
        NOTE: there is one timestep misalignment in prev_actions and current_actions
        """
        assert (
            prev_actions.dim()
            == rewards.dim()
            == observs.dim()
            == current_actions.dim()
            == 3
        )
        assert prev_actions.shape[0] == rewards.shape[0] == observs.shape[0]

        ### 1. get hidden/belief states of the whole/sub trajectories, aligned with observs
        # return the hidden states (T+1, B, dim)
        hidden_states = self.get_hidden_states(
            prev_actions=prev_actions, rewards=rewards, observs=observs
        )

        # 2. another branch for state & **current** action
        if current_actions.shape[0] == observs.shape[0]:
            # current_actions include last obs's action, i.e. we have a'[T] in reaction to o[T]
            if self.algo in [self.TD3_name, self.SAC_name]:
                curr_embed = self.current_state_action_encoder(
                    torch.cat((observs, current_actions), dim=-1)
                )  # (T+1, B, dim)
            else:
                curr_embed = self.current_state_action_encoder(observs)  # (T+1, B, dim)
            # 3. joint embeds
            joint_embeds = torch.cat(
                (hidden_states, curr_embed), dim=-1
            )  # (T+1, B, dim)
        else:
            # current_actions does NOT include last obs's action
            if self.algo in [self.TD3_name, self.SAC_name]:
                curr_embed = self.current_state_action_encoder(
                    torch.cat((observs[:-1], current_actions), dim=-1)
                )  # (T, B, dim)
            else:
                curr_embed = self.current_state_action_encoder(
                    observs[:-1]
                )  # (T, B, dim)
            # 3. joint embeds
            joint_embeds = torch.cat(
                (hidden_states[:-1], curr_embed), dim=-1
            )  # (T, B, dim)

        # 4. q value
        q1 = self.qf1(joint_embeds)
        q2 = self.qf2(joint_embeds)

        return q1, q2  # (T or T+1, B, 1 or A)
