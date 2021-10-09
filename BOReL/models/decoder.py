import torch
import torch.nn as nn
from torch.nn import functional as F

from utils import helpers as utl
from torchkit import pytorch_utils as ptu


class StateTransitionDecoder(nn.Module):
    def __init__(
        self,
        task_embedding_size,
        layers,
        #
        action_size,
        action_embed_size,
        state_size,
        state_embed_size,
        pred_type="deterministic",
    ):
        super(StateTransitionDecoder, self).__init__()

        assert pred_type == "deterministic"
        self.pred_type = pred_type
        self.state_encoder = utl.FeatureExtractor(state_size, state_embed_size, F.relu)
        self.action_encoder = utl.FeatureExtractor(
            action_size, action_embed_size, F.relu
        )

        curr_input_size = task_embedding_size + state_embed_size + action_embed_size
        self.fc_layers = nn.ModuleList([])
        for i in range(len(layers)):
            self.fc_layers.append(nn.Linear(curr_input_size, layers[i]))
            curr_input_size = layers[i]

        # output layer
        if pred_type == "gaussian":
            self.fc_out = nn.Linear(curr_input_size, 2 * state_size)
        else:
            self.fc_out = nn.Linear(curr_input_size, state_size)

    def forward(self, task_embedding, state, action):

        ha = self.action_encoder(action)
        hs = self.state_encoder(state)
        h = torch.cat((task_embedding, hs, ha), dim=-1)

        for i in range(len(self.fc_layers)):
            h = F.relu(self.fc_layers[i](h))

        return self.fc_out(h)


class RewardDecoder(nn.Module):
    def __init__(
        self,
        layers,
        task_embedding_size,
        action_size,
        action_embed_size,
        state_size,
        state_embed_size,
        pred_type="deterministic",
        input_prev_state=False,
        input_action=False,
    ):
        super(RewardDecoder, self).__init__()

        assert pred_type == "deterministic"
        self.pred_type = pred_type
        self.input_prev_state = input_prev_state
        self.input_action = input_action

        # get state as input and predict reward prob
        self.state_encoder = utl.FeatureExtractor(state_size, state_embed_size, F.relu)
        self.action_encoder = utl.FeatureExtractor(
            action_size, action_embed_size, F.relu
        )
        curr_input_size = task_embedding_size + state_embed_size
        if input_prev_state:
            curr_input_size += state_embed_size
        if input_action:
            curr_input_size += action_embed_size
        self.fc_layers = nn.ModuleList([])
        for i in range(len(layers)):
            self.fc_layers.append(nn.Linear(curr_input_size, layers[i]))
            curr_input_size = layers[i]

        if pred_type == "gaussian":
            self.fc_out = nn.Linear(curr_input_size, 2)
        else:  # deterministic
            self.fc_out = nn.Linear(curr_input_size, 1)

    def forward(self, task_embedding, next_state, prev_state=None, action=None):
        """
        r(s',a,s;m), where a and s are optional cuz env emits reward which often
        only depends on next state s'.
        all inputs are 2D or 3D (*, dim)
        """
        # task_embedding = task_embedding.reshape((-1, task_embedding.shape[-1]))
        # next_state = next_state.reshape((-1, next_state.shape[-1]))

        # first embedding layers
        hns = self.state_encoder(next_state)
        h = torch.cat((task_embedding, hns), dim=-1)
        if self.input_action:
            # action = action.reshape((-1, action.shape[-1]))
            ha = self.action_encoder(action)
            h = torch.cat((h, ha), dim=-1)
        if self.input_prev_state:
            # prev_state = prev_state.reshape((-1, prev_state.shape[-1]))
            hps = self.state_encoder(prev_state)
            h = torch.cat((h, hps), dim=-1)

        # concat to a 3-layer MLP
        for i in range(len(self.fc_layers)):
            h = F.relu(self.fc_layers[i](h))

        p_x = self.fc_out(h)  # (B, 1 or 2)

        return p_x
