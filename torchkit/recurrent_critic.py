import torch
import torch.nn as nn
from torch.nn import functional as F
from utils import helpers as utl
from torchkit.networks import FlattenMlp
from torchkit.constant import *
import torchkit.pytorch_utils as ptu


class Critic_RNN(nn.Module):
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
        image_encoder=None,
        **kwargs
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.algo = algo

        ### Build Model
        ## 1. embed action, state, reward (Feed-forward layers first)

        self.image_encoder = image_encoder
        if self.image_encoder is None:
            self.state_encoder = utl.FeatureExtractor(
                obs_dim, state_embedding_size, F.relu
            )
        else:  # for pixel observation, use external encoder
            assert state_embedding_size == 0
            state_embedding_size = self.image_encoder.embed_size  # reset it

        self.action_encoder = utl.FeatureExtractor(
            action_dim, action_embedding_size, F.relu
        )
        self.reward_encoder = utl.FeatureExtractor(1, reward_embedding_size, F.relu)

        ## 2. build RNN model
        rnn_input_size = (
            action_embedding_size + state_embedding_size + reward_embedding_size
        )
        self.rnn_hidden_size = rnn_hidden_size

        assert encoder in RNNs
        self.encoder = encoder

        self.rnn = RNNs[encoder](
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

        ## 3. build another obs+act branch
        shortcut_embedding_size = rnn_input_size
        if self.algo in [TD3_name, SAC_name] and self.image_encoder is None:
            # for vector-based continuous action problems
            self.current_shortcut_encoder = utl.FeatureExtractor(
                obs_dim + action_dim, shortcut_embedding_size, F.relu
            )
        elif self.algo in [TD3_name, SAC_name] and self.image_encoder is not None:
            # for image-based continuous action problems
            self.current_shortcut_encoder = utl.FeatureExtractor(
                action_dim, shortcut_embedding_size, F.relu
            )
            shortcut_embedding_size += self.image_encoder.embed_size
        elif self.algo == SACD_name and self.image_encoder is None:
            # for vector-based discrete action problems
            self.current_shortcut_encoder = utl.FeatureExtractor(
                obs_dim, shortcut_embedding_size, F.relu
            )
        elif self.algo == SACD_name and self.image_encoder is not None:
            # for image-based discrete action problems
            shortcut_embedding_size = self.image_encoder.embed_size
        else:
            raise NotImplementedError

        ## 4. build q networks
        if self.algo in [TD3_name, SAC_name]:
            output_size = 1
        else:  # sac-discrete
            output_size = action_dim
        self.qf1 = FlattenMlp(
            input_size=self.rnn_hidden_size + shortcut_embedding_size,
            output_size=output_size,
            hidden_sizes=dqn_layers,
        )
        self.qf2 = FlattenMlp(
            input_size=self.rnn_hidden_size + shortcut_embedding_size,
            output_size=output_size,
            hidden_sizes=dqn_layers,
        )

    def _get_obs_embedding(self, observs):
        if self.image_encoder is None:  # vector obs
            return self.state_encoder(observs)
        else:  # pixel obs
            return self.image_encoder(observs)

    def _get_shortcut_obs_act_embedding(self, observs, current_actions):
        if self.algo in [TD3_name, SAC_name] and self.image_encoder is None:
            # for vector-based continuous action problems
            return self.current_shortcut_encoder(
                torch.cat([observs, current_actions], dim=-1)
            )
        elif self.algo in [TD3_name, SAC_name] and self.image_encoder is not None:
            # for image-based continuous action problems
            return torch.cat(
                [
                    self.image_encoder(observs),
                    self.current_shortcut_encoder(current_actions),
                ],
                dim=-1,
            )
        elif self.algo == SACD_name and self.image_encoder is None:
            # for vector-based discrete action problems (not using actions)
            return self.current_shortcut_encoder(observs)
        elif self.algo == SACD_name and self.image_encoder is not None:
            # for image-based discrete action problems (not using actions)
            return self.image_encoder(observs)
        else:
            raise NotImplementedError

    def get_hidden_states(self, prev_actions, rewards, observs):
        # all the input have the shape of (T+1, B, *)
        # get embedding of initial transition
        input_a = self.action_encoder(prev_actions)
        input_r = self.reward_encoder(rewards)
        input_s = self._get_obs_embedding(observs)
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
            curr_embed = self._get_shortcut_obs_act_embedding(
                observs, current_actions
            )  # (T+1, B, dim)
            # 3. joint embeds
            joint_embeds = torch.cat(
                (hidden_states, curr_embed), dim=-1
            )  # (T+1, B, dim)
        else:
            # current_actions does NOT include last obs's action
            curr_embed = self._get_shortcut_obs_act_embedding(
                observs[:-1], current_actions
            )  # (T, B, dim)
            # 3. joint embeds
            joint_embeds = torch.cat(
                (hidden_states[:-1], curr_embed), dim=-1
            )  # (T, B, dim)

        # 4. q value
        q1 = self.qf1(joint_embeds)
        q2 = self.qf2(joint_embeds)

        return q1, q2  # (T or T+1, B, 1 or A)
