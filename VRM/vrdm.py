import torch
from torch.autograd import Variable
import numpy as np
import torch.nn as nn
from torch import distributions as dis
from torchkit import pytorch_utils as ptu
from utils import logger

EPS = 1e-6  # Avoid NaN (prevents division by zero or log of zero)
# CAP the standard deviation of the actor
LOG_STD_MAX = 2
LOG_STD_MIN = -20
REG = 1e-3  # regularization of the actor
SIG_MIN = 1e-3


class VRM(nn.Module):
    def __init__(
        self,
        input_size,
        action_size,
        rnn_type="mtlstm",
        d_layers=[256],
        z_layers=[64],
        taus=[
            1.0,
        ],
        decode_layers=[128, 128],
        x_phi_layers=[
            128,
        ],
        posterior_layers=[
            128,
        ],
        prior_layers=[
            128,
        ],
        lr_st=8e-4,
        predict_done=False,
        optimizer="adam",
        feedforward_actfun_rnn=nn.Tanh,
        sig_scale="auto",
    ):
        """
        Variational Multi-Layer RNN model with Action Feedback, using soft actor-critic for reinforcement learning.
        :param input_size: int, size of input vector.
        :param action_size: int, size of action vector.
        :param rnn_type: string, can be 'mtrnn' or 'gru' or 'lstm', indicating the type of RNN used.
        :param d_layers: 1-D int array, indicating how many hidden neurons (d) in each layer. e.g. [256] is one-layer LSTM with 256 units.
        :param z_layers: 1-D int array, indicating how many hidden variable neurons (z) in each layer.
        :param taus: 1-D int array, indicating timescales in each layer. e.g. [1.0] is the normal one.
        :param decode_layers: 1-D int array, indicating layer-sizes of decoding layers, empty array means direct linear connection
        :param x_phi_layers: 1-D int array, indicating layer-sizes of feature extracting layers, empty array means direct linear connection
        :param posterior_layers: 1-D int array, indicating layer-sizes of posterior layers, empty array means direct linear connection
        :param prior_layers: 1-D int array, indicating layer-sizes of prior layers, empty array means direct linear connection
        :param lr_st: learning rate of state transition model (for ELBO)
        :param optimizer: optimizer for state transition model, 'adam' or 'rmsprop'
        :param predict_done: boolean, whether the model predict "done"
        :param sig_scale: sigma value of all the stochastic variables in the model, 'auto' by default, but can be set to certain fixed value
        """
        super(VRM, self).__init__()

        if len(d_layers) != len(taus):
            raise ValueError(
                "Length of hidden layer size and timescales should be the same."
            )

        # Network layer parameters
        self.input_size = input_size
        self.action_size = action_size

        self.d_layers = d_layers
        self.z_layers = z_layers
        self.taus = taus
        self.rnn_type = rnn_type
        self.n_levels = len(d_layers)
        self.decode_layers = decode_layers
        self.x_phi_layers = x_phi_layers  # feature-extracting transformations
        self.prior_layers = prior_layers
        self.posterior_layers = posterior_layers
        self.action_feedback = True
        self.batch = True
        self.predict_done = predict_done
        self.sig_scale = sig_scale

        # feature-extracting transformations
        self.x2phi = nn.ModuleList()
        last_layer_size = self.input_size
        for layer_size in self.x_phi_layers:
            self.x2phi.append(nn.Linear(last_layer_size, layer_size, bias=True))
            last_layer_size = layer_size
            self.x2phi.append(feedforward_actfun_rnn())
        self.x2phi.append(nn.Linear(last_layer_size, self.x_phi_layers[-1], bias=True))

        self.f_x2phi = nn.Sequential(*self.x2phi)

        # input encoding layers
        self.xphi2h0 = nn.Linear(self.x_phi_layers[-1], self.d_layers[0], bias=True)

        if self.action_feedback:
            self.f_daphi2mu_q = nn.ModuleList()
            if isinstance(self.sig_scale, float):
                self.f_daphi2sig_q = lambda x: ptu.tensor(
                    self.sig_scale, dtype=torch.float32
                )
                self.f_da2sig_p = lambda x: ptu.tensor(
                    self.sig_scale, dtype=torch.float32
                )
            else:
                self.f_daphi2sig_q = nn.ModuleList()
                self.f_da2sig_p = nn.ModuleList()
            self.f_da2mu_p = nn.ModuleList()

        else:
            self.f_dphi2mu_q = nn.ModuleList()
            if isinstance(self.sig_scale, float):
                self.f_dphi2sig_q = lambda x: ptu.tensor(
                    self.sig_scale, dtype=torch.float32
                )
                self.f_d2sig_p = lambda x: ptu.tensor(
                    self.sig_scale, dtype=torch.float32
                )
            else:
                self.f_dphi2sig_q = nn.ModuleList()
                self.f_d2sig_p = nn.ModuleList()
            self.f_d2mu_p = nn.ModuleList()

        for lev in range(self.n_levels):

            if self.action_feedback:
                daphi2mu_q = nn.ModuleList()
                daphi2sig_q = nn.ModuleList()
                last_layer_size = (
                    self.d_layers[lev] + self.action_size + self.x_phi_layers[-1]
                )
                for layer_size in self.posterior_layers:
                    daphi2mu_q.append(nn.Linear(last_layer_size, layer_size, bias=True))
                    daphi2mu_q.append(feedforward_actfun_rnn())
                    daphi2sig_q.append(
                        nn.Linear(last_layer_size, layer_size, bias=True)
                    )
                    daphi2sig_q.append(feedforward_actfun_rnn())
                    last_layer_size = layer_size
                daphi2mu_q.append(
                    nn.Linear(last_layer_size, self.z_layers[lev], bias=True)
                )
                daphi2sig_q.append(
                    nn.Linear(last_layer_size, self.z_layers[lev], bias=True)
                )
                daphi2sig_q.append(nn.Softplus())

                self.f_daphi2mu_q.append(nn.Sequential(*daphi2mu_q))
                if not isinstance(self.sig_scale, float):
                    self.f_daphi2sig_q.append(nn.Sequential(*daphi2sig_q))

                da2mu_p = nn.ModuleList()
                da2sig_p = nn.ModuleList()
                last_layer_size = self.d_layers[lev] + self.action_size
                for layer_size in self.prior_layers:
                    da2mu_p.append(nn.Linear(last_layer_size, layer_size, bias=True))
                    da2mu_p.append(feedforward_actfun_rnn())
                    da2sig_p.append(nn.Linear(last_layer_size, layer_size, bias=True))
                    da2sig_p.append(feedforward_actfun_rnn())
                    last_layer_size = layer_size
                da2mu_p.append(
                    nn.Linear(last_layer_size, self.z_layers[lev], bias=True)
                )
                da2sig_p.append(
                    nn.Linear(last_layer_size, self.z_layers[lev], bias=True)
                )
                da2sig_p.append(nn.Softplus())

                self.f_da2mu_p.append(nn.Sequential(*da2mu_p))
                if not isinstance(self.sig_scale, float):
                    self.f_da2sig_p.append(nn.Sequential(*da2sig_p))

            else:
                dphi2mu_q = nn.ModuleList()
                dphi2sig_q = nn.ModuleList()
                last_layer_size = self.d_layers[lev] + self.x_phi_layers[-1]
                for layer_size in self.posterior_layers:
                    dphi2mu_q.append(nn.Linear(last_layer_size, layer_size, bias=True))
                    dphi2mu_q.append(feedforward_actfun_rnn())
                    dphi2sig_q.append(nn.Linear(last_layer_size, layer_size, bias=True))
                    dphi2sig_q.append(feedforward_actfun_rnn())
                    last_layer_size = layer_size
                dphi2mu_q.append(
                    nn.Linear(last_layer_size, self.z_layers[lev], bias=True)
                )
                dphi2sig_q.append(
                    nn.Linear(last_layer_size, self.z_layers[lev], bias=True)
                )
                dphi2sig_q.append(nn.Softplus())

                self.f_dphi2mu_q.append(nn.Sequential(*dphi2mu_q))
                if not isinstance(self.sig_scale, float):
                    self.f_dphi2sig_q.append(nn.Sequential(*dphi2sig_q))

                d2mu_p = nn.ModuleList()
                d2sig_p = nn.ModuleList()
                last_layer_size = self.d_layers[lev]
                for layer_size in self.prior_layers:
                    d2mu_p.append(nn.Linear(last_layer_size, layer_size, bias=True))
                    d2mu_p.append(feedforward_actfun_rnn())
                    d2sig_p.append(nn.Linear(last_layer_size, layer_size, bias=True))
                    d2sig_p.append(feedforward_actfun_rnn())
                    last_layer_size = layer_size
                d2mu_p.append(nn.Linear(last_layer_size, self.z_layers[lev], bias=True))
                d2sig_p.append(
                    nn.Linear(last_layer_size, self.z_layers[lev], bias=True)
                )
                d2sig_p.append(nn.Softplus())

                self.f_d2mu_p.append(nn.Sequential(*d2mu_p))
                if not isinstance(self.sig_scale, float):
                    self.f_d2sig_p.append(nn.Sequential(*d2sig_p))

        # recurrent connections
        if self.rnn_type == "mtrnn":
            self.z2h = nn.ModuleList()
            self.d2h = nn.ModuleDict()
            for l in range(self.n_levels):
                self.z2h.append(nn.Linear(self.z_layers[l], self.d_layers[l]))

                m = nn.Linear(
                    d_layers[l], d_layers[l], bias=True
                )  # link from current level
                self.d2h["{}to{}".format(l, l)] = m
                if l > 0:  # not lowest level, link from one level lower
                    m = nn.Linear(d_layers[l - 1], d_layers[l], bias=True)
                    self.d2h["{}to{}".format(l - 1, l)] = m
                if (
                    l < self.n_levels - 1
                ):  # not highest level, link from one level lower
                    m = nn.Linear(d_layers[l + 1], d_layers[l], bias=True)
                    self.d2h["{}to{}".format(l + 1, l)] = m

        elif self.rnn_type == "mtgru":
            raise NotImplementedError

        elif self.rnn_type == "mtlstm":
            self.rnn_levels = nn.ModuleList()
            for l in range(self.n_levels):
                if l == 0:  # lowest level
                    if self.n_levels == 1:
                        rnn_input_size = self.x_phi_layers[-1] + self.z_layers[l]
                    else:
                        rnn_input_size = (
                            self.x_phi_layers[-1]
                            + self.d_layers[l + 1]
                            + self.z_layers[l]
                        )

                elif (
                    l == self.n_levels - 1
                ):  # not highest level, link from one level lower
                    rnn_input_size = self.d_layers[l - 1] + self.z_layers[l]

                else:
                    rnn_input_size = (
                        self.d_layers[l - 1] + self.d_layers[l + 1] + self.z_layers[l]
                    )

                self.rnn_levels.append(nn.LSTMCell(rnn_input_size, self.d_layers[l]))

        else:
            raise ValueError("rnn_type must be 'mtrnn' or 'mtlstm'")

        # output decoding layers
        self.dz2mux = nn.ModuleList()
        self.dz2sigx = nn.ModuleList()

        last_layer_size = self.d_layers[0] + self.z_layers[0]
        for layer_size in self.decode_layers:
            self.dz2mux.append(nn.Linear(last_layer_size, layer_size, bias=True))
            self.dz2mux.append(feedforward_actfun_rnn())
            self.dz2sigx.append(nn.Linear(last_layer_size, layer_size, bias=True))
            self.dz2sigx.append(feedforward_actfun_rnn())
            last_layer_size = layer_size
        self.dz2mux.append(nn.Linear(last_layer_size, self.input_size, bias=True))
        self.dz2sigx.append(nn.Linear(last_layer_size, self.input_size, bias=True))
        self.dz2sigx.append(nn.Softplus())

        self.f_dz2mux = nn.Sequential(*self.dz2mux)
        if isinstance(self.sig_scale, float):
            self.f_dz2sigx = lambda x: ptu.tensor(self.sig_scale, dtype=torch.float32)
        else:
            self.f_dz2sigx = nn.Sequential(*self.dz2sigx)

        # predict done
        if self.predict_done:
            self.done_hidden_size = 128
            self.dz2logdone = nn.Sequential(
                nn.Linear(
                    self.d_layers[0] + self.z_layers[0],
                    self.done_hidden_size,
                    bias=True,
                ),
                nn.ReLU(),
                nn.Linear(self.done_hidden_size, 2, bias=True),
                nn.LogSoftmax(),
            )
            self.optimizer_done = torch.optim.Adam(
                self.dz2logdone.parameters(), lr=lr_st
            )

        # optimizer
        if optimizer == "rmsprop":
            self.optimizer_st = torch.optim.RMSprop(
                self.parameters(), lr=lr_st, alpha=0.99
            )
        elif optimizer == "adam":
            self.optimizer_st = torch.optim.Adam(self.parameters(), lr=lr_st)

    def rnn(self, prev_h_levels, prev_d_levels, new_z_levels, x_phi):
        new_h_levels = []
        new_d_levels = []

        if self.rnn_type == "mtrnn":

            for l in range(self.n_levels):

                new_h = (1.0 - 1.0 / self.taus[l]) * prev_h_levels[l]

                new_h += (1.0 / self.taus[l]) * self.d2h["{}to{}".format(l, l)](
                    prev_d_levels[l]
                )
                if l > 0:
                    new_h += (1.0 / self.taus[l]) * self.d2h["{}to{}".format(l - 1, l)](
                        prev_d_levels[l - 1]
                    )
                if l < self.n_levels - 1:
                    new_h += (1.0 / self.taus[l]) * self.d2h["{}to{}".format(l + 1, l)](
                        prev_d_levels[l + 1]
                    )

                new_h += (1.0 / self.taus[l]) * self.z2h[l](new_z_levels[l])

                ## encode input
                if l == 0:
                    new_h += (1.0 / self.taus[l]) * self.xphi2h0(x_phi)

                new_h_levels.append(new_h)
                new_d_levels.append(torch.tanh(new_h))

        elif self.rnn_type == "mtgru":  # gru or lstm

            raise NotImplementedError

        elif self.rnn_type == "mtlstm":  # gru or lstm

            for l in range(self.n_levels):
                if l == 0:  # lowest level
                    if self.n_levels == 1:
                        rnn_input = x_phi
                    else:
                        rnn_input = torch.cat((x_phi, prev_d_levels[l + 1]), dim=-1)

                elif (
                    l == self.n_levels - 1
                ):  # not highest level, link from one level lower
                    rnn_input = prev_d_levels[l - 1]

                else:
                    rnn_input = torch.cat(
                        (prev_d_levels[l - 1], prev_d_levels[l + 1]), dim=-1
                    )

                last = torch.cat((rnn_input, new_z_levels[l]), dim=-1)

                new_d, new_h = self.rnn_levels[l](
                    last, (prev_d_levels[l], prev_h_levels[l])
                )

                # dilated LSTM
                mask_new = (
                    torch.rand_like(new_h, dtype=torch.float32) - 1 / self.taus[l]
                )
                mask_new = (1.0 - torch.sign(mask_new)) / 2.0

                mask_old = ptu.ones_like(new_h, dtype=torch.float32) - mask_new

                new_d = mask_new * new_d + mask_old * prev_d_levels[l]
                new_h = mask_new * new_h + mask_old * prev_h_levels[l]

                new_h_levels.append(new_h)
                new_d_levels.append(new_d)

        return new_h_levels, new_d_levels

    def sample_z(self, mu, sig):
        # Using reparameterization trick to sample from a gaussian
        if isinstance(sig, torch.Tensor):
            eps = Variable(ptu.randn_like(mu))
        else:
            eps = ptu.randn_like(mu)
        return mu + sig * eps

    def forward_generative(self, prev_h_levels, prev_d_levels, a_prev):
        # one-step generation

        # prior
        if self.action_feedback:
            a_prev = a_prev.view(prev_h_levels[0].size()[0], -1)
            mu_levels = [
                self.f_da2mu_p[l](torch.cat((prev_d_levels[l], a_prev), dim=-1))
                for l in range(self.n_levels)
            ]
            if isinstance(self.sig_scale, float):
                sig_levels = [
                    ptu.tensor(self.sig_scale, dtype=torch.float32)
                    for l in range(self.n_levels)
                ]
            else:
                sig_levels = [
                    self.f_da2sig_p[l](torch.cat((prev_d_levels[l], a_prev), dim=-1))
                    for l in range(self.n_levels)
                ]
        else:
            mu_levels = [
                self.f_d2mu_p[l](prev_d_levels[l]) for l in range(self.n_levels)
            ]
            if isinstance(self.sig_scale, float):
                sig_levels = [
                    ptu.tensor(self.sig_scale, dtype=torch.float32)
                    for l in range(self.n_levels)
                ]
            else:
                sig_levels = [
                    self.f_d2sig_p[l](prev_d_levels[l]) for l in range(self.n_levels)
                ]

        new_z_p_levels = [
            self.sample_z(mu_levels[l], sig_levels[l]) for l in range(self.n_levels)
        ]

        # pred x

        d0_prev = prev_d_levels[0]
        z0_new = new_z_p_levels[0]

        last = torch.cat((d0_prev, z0_new), dim=-1)
        mux = self.f_dz2mux(last)

        last = torch.cat((d0_prev, z0_new), dim=-1)
        sigx = self.f_dz2sigx(last) + SIG_MIN

        x_pred = self.sample_z(mux, sigx)

        # feature extraction
        x_phi = self.f_x2phi(x_pred)

        new_h_levels, new_d_levels = self.rnn(
            prev_h_levels, prev_d_levels, new_z_p_levels, x_phi
        )

        return (
            x_pred,
            new_h_levels,
            new_d_levels,
            new_z_p_levels,
            mu_levels,
            sig_levels,
            mux,
            sigx,
        )

    def forward_inference(self, prev_h_levels, prev_d_levels, x_obs, a_prev_obs):

        # feature extraction
        last = x_obs.view(prev_h_levels[0].size()[0], -1)
        x_phi = self.f_x2phi(last)

        a_prev_obs = a_prev_obs.view(prev_h_levels[0].size()[0], -1)

        # posterior

        if self.action_feedback:
            mu_levels = [
                self.f_daphi2mu_q[l](
                    torch.cat((prev_d_levels[l], a_prev_obs, x_phi), dim=-1)
                )
                for l in range(self.n_levels)
            ]
            if isinstance(self.sig_scale, float):
                sig_levels = [
                    ptu.tensor(self.sig_scale, dtype=torch.float32)
                    for l in range(self.n_levels)
                ]
            else:
                sig_levels = [
                    self.f_daphi2sig_q[l](
                        torch.cat((prev_d_levels[l], a_prev_obs, x_phi), dim=-1)
                    )
                    for l in range(self.n_levels)
                ]
        else:
            mu_levels = [
                self.f_dphi2mu_q[l](torch.cat((prev_d_levels[l], x_phi), dim=-1))
                for l in range(self.n_levels)
            ]
            if isinstance(self.sig_scale, float):
                sig_levels = [
                    ptu.tensor(self.sig_scale, dtype=torch.float32)
                    for l in range(self.n_levels)
                ]
            else:
                sig_levels = [
                    self.f_dphi2sig_q[l](torch.cat((prev_d_levels[l], x_phi), dim=-1))
                    for l in range(self.n_levels)
                ]

        new_z_q_levels = [
            self.sample_z(mu_levels[l], sig_levels[l]) for l in range(self.n_levels)
        ]

        new_h_levels, new_d_levels = self.rnn(
            prev_h_levels, prev_d_levels, new_z_q_levels, x_phi
        )

        return new_h_levels, new_d_levels, new_z_q_levels, mu_levels, sig_levels

    def train_st(
        self,
        x_obs,
        a_obs,
        h_levels_0=None,
        d_levels_0=None,
        h_0_detach=True,
        validity=None,
        done_obs=None,
        seq_len=64,
    ):
        """
        train the VRNN model using observations x_obs and executed actions a_obs.
        :param x_obs: observations, pytorch tensor, size = batch_size by num_steps by dim_obs.
        :param a_obs: executed actions, pytorch tensor, size = batch_size by num_steps by dim_action.
        :param h_levels_0: initial hidden states of the RNN, list of  pytorch tensors, each level size = batch_size by dim_h.
        :param d_levels_0: initial outputs of the RNN,  list pytorch tensors, each level size = batch_size by dim_h.
        :param h_0_detach: whether initial states are detached in training, boolean, if True, initial states is not trainable.
        :param validity: validity matrix for padding, pytorch tensor (elements are 1 or 0), size = batch_size by num_steps. if validity=None, there is no need for padding.
        :param seq_len: length of sequences used for BPTT
        :return: loss value and h_levels_init, d_levels_init (will be different from inputed one if h_0_detach=False)
        """

        ### shorten x, r .. by using v
        if not validity is None:
            v = validity.cpu().numpy().reshape([x_obs.size()[0], x_obs.size()[1]])
            stps = np.sum(v, axis=1)
            max_stp = int(np.max(stps))

            x_obs = x_obs[:, :max_stp]
            a_obs = a_obs[:, :max_stp]

            if not done_obs is None:
                done_obs = done_obs[:, :max_stp]

            validity = validity[:, :max_stp].reshape([x_obs.size()[0], x_obs.size()[1]])

        batch_size = x_obs.size()[0]

        if validity is None:  # no need for padding
            validity = ptu.ones([x_obs.size()[0], x_obs.size()[1]], requires_grad=False)

        if h_levels_0 is None:
            h_levels_0 = self.init_hidden_zeros(batch_size=batch_size)
        elif isinstance(h_levels_0[0], np.ndarray):
            h_levels_0 = [ptu.from_numpy(h_0) for h_0 in h_levels_0]

        if d_levels_0 is None:
            d_levels_0 = self.init_hidden_zeros(batch_size=batch_size)
        elif isinstance(d_levels_0[0], np.ndarray):
            d_levels_0 = [ptu.from_numpy(d_0) for d_0 in d_levels_0]

        if h_0_detach:
            h_levels_init = [h_0.detach() for h_0 in h_levels_0]
            d_levels_init = [d_0.detach() for d_0 in d_levels_0]
            h_levels = h_levels_init
            d_levels = d_levels_init
        else:
            h_levels_init = [h_0 for h_0 in h_levels_0]
            d_levels_init = [d_0 for d_0 in d_levels_0]
            h_levels = h_levels_init
            d_levels = d_levels_init

        x_obs = x_obs.data
        a_obs = a_obs.data
        if not done_obs is None:
            done_obs = done_obs.data

        # sample minibatch of minibatch_size x seq_len
        stps_burnin = 64
        x_sampled = ptu.zeros(
            [x_obs.size()[0], seq_len, x_obs.size()[-1]], dtype=torch.float32
        )
        a_sampled = ptu.zeros(
            [a_obs.size()[0], seq_len, a_obs.size()[-1]], dtype=torch.float32
        )
        v_sampled = ptu.zeros([validity.size()[0], seq_len], dtype=torch.float32)

        for b in range(x_obs.size()[0]):
            v = validity.cpu().numpy().reshape([x_obs.size()[0], x_obs.size()[1]])
            stps = np.sum(v[b], axis=0).astype(int)
            start_index = np.random.randint(-seq_len + 1, stps - 1)

            for tmp, TMP in zip(
                (x_sampled, a_sampled, v_sampled), (x_obs, a_obs, validity)
            ):

                if start_index < 0 and start_index + seq_len > stps:
                    tmp[b, :stps] = TMP[b, :stps]

                elif start_index < 0:
                    tmp[b, : (start_index + seq_len)] = TMP[
                        b, : (start_index + seq_len)
                    ]

                elif start_index + seq_len > stps:
                    tmp[b, : (stps - start_index)] = TMP[b, start_index:stps]

                else:
                    tmp[b] = TMP[b, start_index : (start_index + seq_len)]

            h_levels_b = [h_level[b : b + 1] for h_level in h_levels]
            d_levels_b = [d_level[b : b + 1] for d_level in d_levels]

            if start_index < 1:
                pass
            else:
                x_tmp = x_obs[
                    b : b + 1, max(0, start_index - stps_burnin) : start_index
                ]
                a_tmp = a_obs[
                    b : b + 1, max(0, start_index - stps_burnin) : start_index
                ]

                for t_burnin in range(x_tmp.size()[1]):

                    h_levels_b, d_levels_b, _, _, _ = self.forward_inference(
                        h_levels_b, d_levels_b, x_tmp[:, t_burnin], a_tmp[:, t_burnin]
                    )

                for lev in range(self.n_levels):
                    h_levels[lev][b] = h_levels_b[lev][0].data
                    d_levels[lev][b] = d_levels_b[lev][0].data
        KL = 0

        h_series_levels = [[] for l in range(self.n_levels)]
        d_series_levels = [[] for l in range(self.n_levels)]
        z_p_series_levels = [[] for l in range(self.n_levels)]
        sig_p_series_levels = [[] for l in range(self.n_levels)]
        sig_q_series_levels = [[] for l in range(self.n_levels)]
        mu_p_series_levels = [[] for l in range(self.n_levels)]
        mu_q_series_levels = [[] for l in range(self.n_levels)]
        mux_pred_series = []
        sigx_pred_series = []

        for stp in range(seq_len):

            curr_x_obs = x_sampled[:, stp]
            prev_a_obs = a_sampled[:, stp]

            a_prev = prev_a_obs if self.action_feedback else None

            if not isinstance(self.sig_scale, float):
                (
                    x_pred,
                    _,
                    _,
                    z_p_levels,
                    mu_p_levels,
                    sig_p_levels,
                    mux_pred,
                    sigx_pred,
                ) = self.forward_generative(h_levels, d_levels, a_prev)
                (
                    h_levels,
                    d_levels,
                    z_q_levels,
                    mu_q_levels,
                    sig_q_levels,
                ) = self.forward_inference(h_levels, d_levels, curr_x_obs, prev_a_obs)
            else:
                (
                    x_pred,
                    _,
                    _,
                    z_p_levels,
                    mu_p_levels,
                    sig_p_levels,
                    _,
                    _,
                ) = self.forward_generative(h_levels, d_levels, a_prev)
                (
                    h_levels,
                    d_levels_new,
                    z_q_levels,
                    mu_q_levels,
                    sig_q_levels,
                ) = self.forward_inference(h_levels, d_levels, curr_x_obs, prev_a_obs)

                last = torch.cat((d_levels[0], z_q_levels[0]), dim=-1)
                mux_pred = self.f_dz2mux(last)
                sigx_pred = self.f_dz2sigx(last) + SIG_MIN

                d_levels = d_levels_new

            # KL divergence term

            for l in range(self.n_levels):
                h_series_levels[l].append(h_levels[l])
                d_series_levels[l].append(d_levels[l])
                z_p_series_levels[l].append(z_p_levels[l])
                mu_p_series_levels[l].append(mu_p_levels[l])
                sig_p_series_levels[l].append(sig_p_levels[l])
                mu_q_series_levels[l].append(mu_q_levels[l])
                sig_q_series_levels[l].append(sig_q_levels[l])

            mux_pred_series.append(mux_pred)
            if not isinstance(self.sig_scale, float):
                sigx_pred_series.append(sigx_pred)

        if not isinstance(self.sig_scale, float):
            sig_p_tensor_levels = [
                torch.stack(sig_p_series_levels[l], dim=1) for l in range(self.n_levels)
            ]
            sig_q_tensor_levels = [
                torch.stack(sig_q_series_levels[l], dim=1) for l in range(self.n_levels)
            ]
            sigx_pred_tensor = torch.stack(sigx_pred_series, dim=1)

        mu_p_tensor_levels = [
            torch.stack(mu_p_series_levels[l], dim=1) for l in range(self.n_levels)
        ]
        mu_q_tensor_levels = [
            torch.stack(mu_q_series_levels[l], dim=1) for l in range(self.n_levels)
        ]
        mux_pred_tensor = torch.stack(mux_pred_series, dim=1)

        if not isinstance(self.sig_scale, float):
            for l in range(self.n_levels):
                KL += (
                    torch.mean(
                        torch.mean(
                            torch.log(sig_p_tensor_levels[l])
                            - torch.log(sig_q_tensor_levels[l])
                            + (
                                (mu_p_tensor_levels[l] - mu_q_tensor_levels[l]).pow(2)
                                + sig_q_tensor_levels[l].pow(2)
                            )
                            / (2.0 * sig_p_tensor_levels[l].pow(2))
                            - 0.5,
                            dim=-1,
                        )
                        * v_sampled
                    )
                    / self.n_levels
                )

            # log likelihood term
            Log = torch.mean(
                torch.mean(
                    -torch.pow(mux_pred_tensor - x_sampled, 2)
                    / torch.pow(sigx_pred_tensor, 2)
                    / 2
                    - torch.log(sigx_pred_tensor * 2.5066),
                    dim=-1,
                )
                * v_sampled
            )
            elbo = -KL + Log
            loss = -elbo
        else:
            loss = torch.mean((mux_pred_tensor - x_sampled).pow(2))

        self.optimizer_st.zero_grad()
        loss.backward()

        if self.predict_done:
            self.optimizer_done.zero_grad()

            d_tensor_0 = torch.stack(d_series_levels[0], dim=1).detach().data
            z_p_tensor_0 = torch.stack(z_p_series_levels[0], dim=1).detach().data
            logdone_tensor = self.dz2logdone(
                torch.cat((d_tensor_0, z_p_tensor_0), dim=-1)
            )

            loss_done = -torch.mean(torch.sum(done_obs * logdone_tensor, dim=-1))

            loss_done.backward()
            self.optimizer_done.step()

        self.optimizer_st.step()

        return loss.cpu().item(), h_levels_init, d_levels_init

    def init_hidden_zeros(self, batch_size=1):

        h_levels = [ptu.zeros((batch_size, d_size)) for d_size in self.d_layers]

        return h_levels


class VRDM(nn.Module):
    def __init__(
        self,
        fim: VRM,
        klm: VRM,
        lr_rl=3e-4,
        gamma=0.99,
        feedforward_actfun_sac=nn.ReLU,
        beta_h="auto",
        policy_layers=[256, 256],
        value_layers=[256, 256],
    ):
        """
        :param fim: the first-impression model
        :param klm: the keep learning model
        :param lr_rl: learning rate for RL
        :param gamma: discount factor
        :param beta_h: entropy coefficient, can also be a fixed float value
        :param policy_layers: 1-D int array, indicating layer-sizes of policy layers, empty array means direct linear connection
        :param value_layers: 1-D int array, indicating layer-sizes of V and Q layers, empty array means direct linear connection
        """
        super(VRDM, self).__init__()

        self.gamma = gamma
        self.a_prev = None

        self.fim = fim
        self.klm = klm
        self.action_size = self.fim.action_size
        self.input_size = self.fim.input_size

        self.include_obs = True
        self.policy_layers = policy_layers
        self.value_layers = value_layers

        self.d_layers = []
        for lev in range(self.fim.n_levels):
            self.d_layers.append(self.fim.d_layers[lev] + self.klm.d_layers[lev])
        self.n_levels = len(self.d_layers)

        self.forward_inference_fim = self.fim.forward_inference
        self.forward_inference_klm = self.klm.forward_inference

        self.h_levels = self.init_hidden_zeros(batch_size=1)
        self.d_levels = self.init_hidden_zeros(batch_size=1)

        self.beta_h = beta_h
        self.target_entropy = -np.float32(self.action_size)

        if isinstance(self.beta_h, str) and self.beta_h.startswith("auto"):
            self.log_beta_h = torch.zeros(1, requires_grad=True, device=ptu.device)
            # self.beta_h = torch.exp(self.log_beta_h)

        # policy network
        self.d2mua = nn.ModuleList()
        last_layer_size = (
            self.d_layers[0]
            if not self.include_obs
            else self.d_layers[0] + self.input_size
        )
        for layer_size in self.policy_layers:
            self.d2mua.append(nn.Linear(last_layer_size, layer_size, bias=True))
            last_layer_size = layer_size
            self.d2mua.append(feedforward_actfun_sac())
        self.d2mua.append(nn.Linear(last_layer_size, self.action_size, bias=True))
        # self.d2mua.append(nn.Tanh())

        self.f_d2mua = nn.Sequential(*self.d2mua)

        self.d2log_siga = nn.ModuleList()
        last_layer_size = (
            self.d_layers[0]
            if not self.include_obs
            else self.d_layers[0] + self.input_size
        )
        for layer_size in self.policy_layers:
            self.d2log_siga.append(nn.Linear(last_layer_size, layer_size, bias=True))
            last_layer_size = layer_size
            self.d2log_siga.append(feedforward_actfun_sac())
        self.d2log_siga.append(nn.Linear(last_layer_size, self.action_size, bias=True))

        self.f_d2log_siga = nn.Sequential(*self.d2log_siga)

        # V network
        self.d2v = nn.ModuleList()
        last_layer_size = (
            self.d_layers[0]
            if not self.include_obs
            else self.d_layers[0] + self.input_size
        )
        for layer_size in self.value_layers:
            self.d2v.append(nn.Linear(last_layer_size, layer_size, bias=True))
            last_layer_size = layer_size
            self.d2v.append(feedforward_actfun_sac())
        self.d2v.append(nn.Linear(last_layer_size, 1, bias=True))

        self.f_d2v = nn.Sequential(*self.d2v)

        # Q networks (double q-learning)
        self.da2q1 = nn.ModuleList()
        last_layer_size = (
            self.d_layers[0] + self.action_size
            if not self.include_obs
            else self.d_layers[0] + self.input_size + self.action_size
        )
        for layer_size in self.value_layers:
            self.da2q1.append(nn.Linear(last_layer_size, layer_size, bias=True))
            last_layer_size = layer_size
            self.da2q1.append(feedforward_actfun_sac())
        self.da2q1.append(nn.Linear(last_layer_size, 1, bias=True))

        self.f_da2q1 = nn.Sequential(*self.da2q1)

        self.da2q2 = nn.ModuleList()
        last_layer_size = (
            self.d_layers[0] + self.action_size
            if not self.include_obs
            else self.d_layers[0] + self.input_size + self.action_size
        )
        for layer_size in self.value_layers:
            self.da2q2.append(nn.Linear(last_layer_size, layer_size, bias=True))
            last_layer_size = layer_size
            self.da2q2.append(feedforward_actfun_sac())
        self.da2q2.append(nn.Linear(last_layer_size, 1, bias=True))

        self.f_da2q2 = nn.Sequential(*self.da2q2)

        # target V network
        self.d2v_tar = nn.ModuleList()
        last_layer_size = (
            self.d_layers[0]
            if not self.include_obs
            else self.d_layers[0] + self.input_size
        )
        for layer_size in self.value_layers:
            self.d2v_tar.append(nn.Linear(last_layer_size, layer_size, bias=True))
            last_layer_size = layer_size
            self.d2v_tar.append(feedforward_actfun_sac())
        self.d2v_tar.append(nn.Linear(last_layer_size, 1, bias=True))

        self.f_d2v_tar = nn.Sequential(*self.d2v_tar)

        # synchronizing target V network and V network
        state_dict_tar = self.f_d2v_tar.state_dict()
        state_dict = self.f_d2v.state_dict()
        for key in list(self.f_d2v.state_dict().keys()):
            state_dict_tar[key] = state_dict[key]
        self.f_d2v_tar.load_state_dict(state_dict_tar)

        # p = prior (generative model), q = posterier (inference model)

        self.optimizer_a = torch.optim.Adam(
            [*self.f_d2mua.parameters(), *self.f_d2log_siga.parameters()], lr=lr_rl
        )
        self.optimizer_v = torch.optim.Adam(
            [
                *self.f_da2q1.parameters(),
                *self.f_da2q2.parameters(),
                *self.f_d2v.parameters(),
            ],
            lr=lr_rl,
        )
        self.optimizer_e = torch.optim.Adam(
            [self.log_beta_h], lr=lr_rl
        )  # optimizer for beta_h

        self.mse_loss = nn.MSELoss()

    def sample_z(self, mu, sig):
        # Using reparameterization trick to sample from a gaussian
        eps = Variable(torch.randn_like(mu))
        return mu + sig * eps

    def sample_action(self, d0_prev, x_prev, detach=False):
        # output action
        if not self.include_obs:
            s = d0_prev
        else:
            s = torch.cat((d0_prev, x_prev), dim=-1)

        mua = self.f_d2mua(s)

        siga = torch.exp(self.f_d2log_siga(s).clamp(LOG_STD_MIN, LOG_STD_MAX))

        if detach:
            return (
                torch.tanh(self.sample_z(mua, siga).detach()),
                mua.detach(),
                siga.detach(),
            )
        else:
            return torch.tanh(self.sample_z(mua, siga).detach()), mua, siga

    def preprocess_sac(self, x_obs, r_obs, a_obs, d_obs=None, v_obs=None, seq_len=64):

        ### shorten x, r .. by using v
        if not v_obs is None:
            v = v_obs.cpu().numpy().reshape([x_obs.size()[0], x_obs.size()[1]])
            stps = np.sum(v, axis=1)
            max_stp = int(np.max(stps))

            x_obs = x_obs[:, :max_stp]
            a_obs = a_obs[:, :max_stp]
            r_obs = r_obs[:, :max_stp]
            d_obs = d_obs[:, :max_stp]
            v_obs = v_obs[:, :max_stp]

        batch_size = x_obs.size()[0]
        start_indices = np.zeros(x_obs.size()[0], dtype=int)

        for b in range(x_obs.size()[0]):
            v = v_obs.cpu().numpy().reshape([x_obs.size()[0], x_obs.size()[1]])
            stps = np.sum(v[b], axis=0).astype(int)
            start_indices[b] = np.random.randint(-seq_len + 1, stps - 1)

        x_obs = x_obs.data
        a_obs = a_obs.data
        r_obs = r_obs.data
        d_obs = d_obs.data
        v_obs = v_obs.data

        # initialize hidden states
        h_levels_0 = self.init_hidden_zeros(batch_size=batch_size)
        d_levels_0 = self.init_hidden_zeros(batch_size=batch_size)

        h_levels = [h_0.detach() for h_0 in h_levels_0]
        d_levels = [d_0.detach() for d_0 in d_levels_0]

        h_levels_fim = []
        d_levels_fim = []
        h_levels_klm = []
        d_levels_klm = []

        for lev in range(self.n_levels):
            h_levels_fim.append(h_levels[lev][:, : self.fim.d_layers[lev]])
            d_levels_fim.append(d_levels[lev][:, : self.fim.d_layers[lev]])

            h_levels_klm.append(h_levels[lev][:, -self.klm.d_layers[lev] :])
            d_levels_klm.append(d_levels[lev][:, -self.klm.d_layers[lev] :])
        # ========================= FIM =========================

        # h_series_levels = [[] for l in range(self.n_levels)]
        d_series_levels_fim = [[] for l in range(self.n_levels)]

        stps_burnin = 64

        x_sampled = ptu.zeros(
            [x_obs.size()[0], seq_len + 1, x_obs.size()[-1]], dtype=torch.float32
        )  # +1 for SP
        a_sampled = ptu.zeros(
            [a_obs.size()[0], seq_len + 1, a_obs.size()[-1]], dtype=torch.float32
        )

        for b in range(x_obs.size()[0]):  # for each trajectory
            v = v_obs.cpu().numpy().reshape([x_obs.size()[0], x_obs.size()[1]])
            stps = np.sum(v[b], axis=0).astype(int)  # real length
            start_index = start_indices[b]

            for tmp, TMP in zip((x_sampled, a_sampled), (x_obs, a_obs)):
                # select the [max(0, start_idx), min(start_idx+seq_len+1, stps)] sub-traj

                sub_seq_start = max(0, start_index)
                sub_seq_end = min(start_index + seq_len + 1, stps)
                tmp[b, : sub_seq_end - sub_seq_start] = TMP[
                    b, sub_seq_start:sub_seq_end
                ]

            h_levels_b_fim = [h_level[b : b + 1] for h_level in h_levels_fim]
            d_levels_b_fim = [d_level[b : b + 1] for d_level in d_levels_fim]

            if start_index < 1:
                pass
            else:
                x_tmp = x_obs[
                    b : b + 1, max(0, start_index - stps_burnin) : start_index
                ]
                a_tmp = a_obs[
                    b : b + 1, max(0, start_index - stps_burnin) : start_index
                ]

                for t_burnin in range(x_tmp.size()[0]):
                    x_tmp_t = x_tmp[:, t_burnin]
                    a_tmp_t = a_tmp[:, t_burnin] if self.fim.action_feedback else None
                    (
                        h_levels_b_fim,
                        d_levels_b_fim,
                        _,
                        _,
                        _,
                    ) = self.forward_inference_fim(
                        h_levels_b_fim, d_levels_b_fim, x_tmp_t, a_tmp_t
                    )

                for lev in range(self.n_levels):
                    h_levels_fim[lev][b] = h_levels_b_fim[lev][0].data
                    d_levels_fim[lev][b] = d_levels_b_fim[lev][0].data

        for stp in range(seq_len + 1):
            curr_x_obs = x_sampled[:, stp]
            prev_a_obs = a_sampled[:, stp] if self.fim.action_feedback else None
            h_levels_fim, d_levels_fim, _, _, _ = self.forward_inference_fim(
                h_levels_fim, d_levels_fim, curr_x_obs, prev_a_obs
            )

            for l in range(self.n_levels):
                d_series_levels_fim[l].append(d_levels_fim[l].detach())

        d_low_tensor_fim = torch.stack(d_series_levels_fim[0], dim=1).detach().data

        S_sampled_fim = d_low_tensor_fim[:, :-1, :]
        SP_sampled_fim = d_low_tensor_fim[:, 1:, :]

        # ========================= END - FIM =========================

        # ========================= KLM =========================
        d_series_levels_klm = [[] for l in range(self.n_levels)]

        stps_burnin = 64

        for b in range(x_obs.size()[0]):
            v = v_obs.cpu().numpy().reshape([x_obs.size()[0], x_obs.size()[1]])
            start_index = start_indices[b]

            h_levels_b_klm = [h_level[b : b + 1] for h_level in h_levels_klm]
            d_levels_b_klm = [d_level[b : b + 1] for d_level in d_levels_klm]

            if start_index < 1:
                pass
            else:
                x_tmp = x_obs[
                    b : b + 1, max(0, start_index - stps_burnin) : start_index
                ]
                a_tmp = a_obs[
                    b : b + 1, max(0, start_index - stps_burnin) : start_index
                ]

                for t_burnin in range(x_tmp.size()[0]):
                    x_tmp_t = x_tmp[:, t_burnin]
                    a_tmp_t = a_tmp[:, t_burnin] if self.klm.action_feedback else None
                    (
                        h_levels_b_klm,
                        d_levels_b_klm,
                        _,
                        _,
                        _,
                    ) = self.forward_inference_klm(
                        h_levels_b_klm, d_levels_b_klm, x_tmp_t, a_tmp_t
                    )

                for lev in range(self.n_levels):
                    h_levels_klm[lev][b] = h_levels_b_klm[lev][0].data
                    d_levels_klm[lev][b] = d_levels_b_klm[lev][0].data

        for stp in range(seq_len + 1):
            curr_x_obs = x_sampled[:, stp]
            prev_a_obs = a_sampled[:, stp] if self.klm.action_feedback else None
            h_levels_klm, d_levels_klm, _, _, _ = self.forward_inference_klm(
                h_levels_klm, d_levels_klm, curr_x_obs, prev_a_obs
            )

            for l in range(self.n_levels):
                d_series_levels_klm[l].append(d_levels_klm[l].detach())

        d_low_tensor_klm = torch.stack(d_series_levels_klm[0], dim=1).detach().data

        S_sampled_klm = d_low_tensor_klm[:, :-1, :]
        SP_sampled_klm = d_low_tensor_klm[:, 1:, :]
        # ========================= END - KLM =========================

        if self.include_obs:
            S_sampled = torch.cat(
                (S_sampled_fim, S_sampled_klm, x_sampled[:, :-1, :]), dim=-1
            )
            SP_sampled = torch.cat(
                (SP_sampled_fim, SP_sampled_klm, x_sampled[:, 1:, :]), dim=-1
            )
        else:
            S_sampled = torch.cat((S_sampled_fim, S_sampled_klm), dim=-1)
            SP_sampled = torch.cat((SP_sampled_fim, SP_sampled_klm), dim=-1)

        A = a_obs
        R = r_obs

        if d_obs is None:
            D = ptu.zeros_like(R, dtype=torch.float32)
        else:
            D = d_obs

        if v_obs is None:  # no need for padding
            V = ptu.ones_like(R, requires_grad=False, dtype=torch.float32)
        else:
            V = v_obs

        A_sampled = ptu.zeros(
            [A.size()[0], seq_len + 1, A.size()[-1]], dtype=torch.float32
        )
        D_sampled = ptu.zeros([D.size()[0], seq_len + 1, 1], dtype=torch.float32)
        R_sampled = ptu.zeros([R.size()[0], seq_len + 1, 1], dtype=torch.float32)
        V_sampled = ptu.zeros([V.size()[0], seq_len + 1, 1], dtype=torch.float32)

        for b in range(A.size()[0]):
            v = v_obs.cpu().numpy().reshape([A.size()[0], A.size()[1]])
            stps = np.sum(v[b], axis=0).astype(int)
            start_index = start_indices[b]

            # sampled_indices = np.arange(start_index, start_index + seq_len)

            for tmp, TMP in zip(
                (A_sampled, D_sampled, R_sampled, V_sampled), (A, D, R, V)
            ):
                # select the [max(0, start_idx), min(start_idx+seq_len+1, stps)] sub-traj

                sub_seq_start = max(0, start_index)
                sub_seq_end = min(start_index + seq_len + 1, stps)
                tmp[b, : sub_seq_end - sub_seq_start] = TMP[
                    b, sub_seq_start:sub_seq_end
                ]

        R_sampled = R_sampled[:, 1:, :].data
        A_sampled = A_sampled[:, 1:, :].data
        D_sampled = D_sampled[:, 1:, :].data
        V_sampled = V_sampled[:, 1:, :].data

        return S_sampled, SP_sampled, A_sampled, R_sampled, D_sampled, V_sampled

    def train_rl_sac_(
        self, S_sampled, SP_sampled, A_sampled, R_sampled, D_sampled, V_sampled
    ):

        gamma = self.gamma

        if isinstance(self.beta_h, str):
            beta_h = torch.exp(self.log_beta_h).data
        else:
            beta_h = self.beta_h

        mua_tensor = self.f_d2mua(S_sampled)
        siga_tensor = torch.exp(
            self.f_d2log_siga(S_sampled).clamp(LOG_STD_MIN, LOG_STD_MAX)
        )
        v_tensor = self.f_d2v(S_sampled)
        vp_tensor = self.f_d2v_tar(SP_sampled)
        q_tensor_1 = self.f_da2q1(torch.cat((S_sampled, A_sampled), dim=-1))
        q_tensor_2 = self.f_da2q2(torch.cat((S_sampled, A_sampled), dim=-1))

        # Using Torch API for computing log_prob
        # --------- loss_v ------------
        mu_prob = dis.Normal(mua_tensor, siga_tensor)

        sampled_u = mu_prob.sample()
        sampled_a = torch.tanh(sampled_u)

        log_pi_exp = torch.sum(
            mu_prob.log_prob(sampled_u), dim=-1, keepdim=True
        ) - torch.sum(torch.log(1 - sampled_a.pow(2) + EPS), dim=-1, keepdim=True)

        sampled_q = torch.min(
            self.f_da2q1(torch.cat((S_sampled, sampled_a), dim=-1)).data,
            self.f_da2q2(torch.cat((S_sampled, sampled_a), dim=-1)).data,
        )
        q_exp = sampled_q

        v_tar = (q_exp - beta_h * log_pi_exp.data).detach().data

        loss_v = 0.5 * self.mse_loss(v_tensor * V_sampled, v_tar * V_sampled)

        loss_q = 0.5 * self.mse_loss(
            q_tensor_1 * V_sampled,
            (R_sampled + (1 - D_sampled) * gamma * vp_tensor.detach().data) * V_sampled,
        ) + 0.5 * self.mse_loss(
            q_tensor_2 * V_sampled,
            (R_sampled + (1 - D_sampled) * gamma * vp_tensor.detach().data) * V_sampled,
        )

        loss_critic = loss_v + loss_q

        # ----------- loss_a ---------------
        mu_prob = dis.Normal(mua_tensor, siga_tensor)

        sampled_u = mu_prob.rsample()
        sampled_a = torch.tanh(sampled_u)

        log_pi_exp = torch.sum(
            mu_prob.log_prob(sampled_u).clamp(-20, 10), dim=-1, keepdim=True
        ) - torch.sum(torch.log(1 - sampled_a.pow(2) + EPS), dim=-1, keepdim=True)

        loss_a = torch.mean(
            beta_h * log_pi_exp * V_sampled
            - torch.min(
                self.f_da2q1(torch.cat((S_sampled, sampled_a), dim=-1)),
                self.f_da2q2(torch.cat((S_sampled, sampled_a), dim=-1)),
            )
            * V_sampled
        ) + REG / 2 * (
            torch.mean(
                (
                    siga_tensor
                    * V_sampled.repeat_interleave(siga_tensor.size()[-1], dim=-1)
                ).pow(2)
            )
            + torch.mean(
                (
                    mua_tensor
                    * V_sampled.repeat_interleave(mua_tensor.size()[-1], dim=-1)
                ).pow(2)
            )
        )

        self.optimizer_v.zero_grad()
        self.optimizer_a.zero_grad()
        (loss_critic + loss_a).backward()  # avoid multiple backward
        self.optimizer_v.step()
        self.optimizer_a.step()
        # --------------------------------------------------------------------------

        # update entropy coefficient if required
        if isinstance(beta_h, torch.Tensor):
            self.optimizer_e.zero_grad()

            loss_e = -torch.mean(
                self.log_beta_h * (log_pi_exp + self.target_entropy).data
            )
            loss_e.backward()
            self.optimizer_e.step()

        # update target V network
        state_dict_tar = self.f_d2v_tar.state_dict()
        state_dict = self.f_d2v.state_dict()
        for key in list(self.f_d2v.state_dict().keys()):
            state_dict_tar[key] = 0.995 * state_dict_tar[key] + 0.005 * state_dict[key]
            # state_dict_tar[key] = 0 * state_dict_tar[key] + 1 * state_dict[key]
        # self.f_d2v_tar.load_state_dict(state_dict)
        self.f_d2v_tar.load_state_dict(state_dict_tar)

        return loss_v.item(), loss_a.item(), loss_q.item()

    def init_hidden_zeros(self, batch_size=1):

        h_levels = [ptu.zeros((batch_size, d_size)) for d_size in self.d_layers]

        return h_levels

    def detach_states(self, states):
        states = [s.detach() for s in states]
        return states

    def init_episode(self, x_0=None, h_levels_0=None, d_levels_0=None):
        if h_levels_0 is None:
            self.h_levels = self.init_hidden_zeros(batch_size=1)
        else:
            self.h_levels = [ptu.from_numpy(h0) for h0 in h_levels_0]

        if d_levels_0 is None:
            self.d_levels = self.init_hidden_zeros(batch_size=1)
        else:
            self.d_levels = [ptu.from_numpy(d0) for d0 in d_levels_0]

        if x_0 is None:
            x_obs_0 = None
        else:
            x_obs_0 = ptu.from_numpy(x_0).view(1, -1)

        a, _, _ = self.sample_action(self.d_levels[0], x_obs_0, detach=True)

        self.a_prev = a

        return a.cpu().numpy()

    def select(self, s, r_prev, action_return="normal"):
        r_prev = np.array([r_prev]).reshape([-1]).astype(np.float32)
        s = np.array(s).reshape([-1]).astype(np.float32)
        x_obs = torch.cat((ptu.from_numpy(s), ptu.from_numpy(r_prev))).view([1, -1])

        self.h_levels_fim = []
        self.d_levels_fim = []
        self.h_levels_klm = []
        self.d_levels_klm = []
        for lev in range(self.fim.n_levels):
            self.h_levels_fim.append(self.h_levels[lev][:, : self.fim.d_layers[lev]])
            self.d_levels_fim.append(self.d_levels[lev][:, : self.fim.d_layers[lev]])

            self.h_levels_klm.append(self.h_levels[lev][:, -self.klm.d_layers[lev] :])
            self.d_levels_klm.append(self.d_levels[lev][:, -self.klm.d_layers[lev] :])

        self.h_levels_fim, self.d_levels_fim, _, _, _ = self.forward_inference_fim(
            self.h_levels_fim, self.d_levels_fim, x_obs, self.a_prev
        )
        self.h_levels_klm, self.d_levels_klm, _, _, _ = self.forward_inference_klm(
            self.h_levels_klm, self.d_levels_klm, x_obs, self.a_prev
        )
        for lev in range(self.n_levels):
            self.h_levels[lev] = torch.cat(
                (self.h_levels_fim[lev], self.h_levels_klm[lev]), dim=-1
            )
            self.d_levels[lev] = torch.cat(
                (self.d_levels_fim[lev], self.d_levels_klm[lev]), dim=-1
            )

        a, mua, siga = self.sample_action(self.d_levels[0], x_obs, detach=True)

        self.a_prev = a

        if action_return == "mean":
            return torch.tanh(mua).cpu().numpy()
        else:
            return a.cpu().numpy()

    def learn_st(
        self,
        train_fim: bool,
        train_klm: bool,
        SP,
        A,
        R,
        D=None,
        V=None,
        H0=None,
        D0=None,
        times=1,
        minibatch_size=4,
        seq_len=64,
    ):  # learning from the data of this episode

        if D is None:
            D = np.zeros_like(R, dtype=np.float32)
        if V is None:
            V = np.ones_like(R, dtype=np.float32)

        for xt in range(times):
            weights = np.sum(V, axis=-1) + 2 * seq_len - 2
            e_samples = np.random.choice(
                SP.shape[0], minibatch_size, p=weights / weights.sum()
            )

            sp = SP[e_samples]
            a = A[e_samples]
            r = R[e_samples]
            d = D[e_samples]
            v = V[e_samples]

            if not H0 is None:
                h0 = [hl[e_samples] for hl in H0]
            else:
                h0 = None

            if not D0 is None:
                d0 = [dl[e_samples] for dl in D0]
            else:
                d0 = None

            r_obs = ptu.from_numpy(r.reshape([r.shape[0], r.shape[1], 1]))
            x_obs = torch.cat((ptu.from_numpy(sp), r_obs), dim=-1)

            a_obs = ptu.from_numpy(a)

            ####
            d_obs = ptu.from_numpy(d.reshape([r.shape[0], r.shape[1], 1]))
            ####

            v_obs = ptu.from_numpy(v.reshape([r.shape[0], r.shape[1], 1]))

            if train_fim:
                loss, h_levels_init, d_levels_init = self.fim.train_st(
                    x_obs,
                    a_obs,
                    h_levels_0=h0,
                    validity=v_obs,
                    d_levels_0=d0,
                    h_0_detach=False,
                    done_obs=d_obs,
                    seq_len=seq_len,
                )

            if train_klm:
                loss, h_levels_init, d_levels_init = self.klm.train_st(
                    x_obs,
                    a_obs,
                    h_levels_0=h0,
                    validity=v_obs,
                    d_levels_0=d0,
                    h_0_detach=False,
                    done_obs=d_obs,
                    seq_len=seq_len,
                )

        if not H0 is None:
            for l in range(len(H0)):
                H0[l][e_samples, :] = h_levels_init[l].cpu().detach().numpy()
                D0[l][e_samples, :] = d_levels_init[l].cpu().detach().numpy()

        return H0, D0, loss

    def learn_rl_sac(
        self, SP, A, R, D=None, V=None, times=1, minibatch_size=4, seq_len=64
    ):

        if D is None:
            D = np.zeros_like(R, dtype=np.float32)
        if V is None:
            V = np.ones_like(R, dtype=np.float32)

        for _ in range(times):

            # sample the trajectory weighted by their length
            weights = np.sum(V, axis=-1) + 2 * seq_len - 2
            e_samples = np.random.choice(
                SP.shape[0], minibatch_size, p=weights / weights.sum()
            )

            sp = SP[e_samples]  # (B, T, dim)
            a = A[e_samples]
            r = R[e_samples]
            d = D[e_samples]
            v = V[e_samples]

            r_obs = ptu.from_numpy(r.reshape([r.shape[0], r.shape[1], 1]))
            x_obs = torch.cat((ptu.from_numpy(sp), r_obs), dim=-1)  # (B, T, S+1)

            a_obs = ptu.from_numpy(a)  # (B, T, A)

            d_obs = ptu.from_numpy(d.reshape([r.shape[0], r.shape[1], 1]))  # (B, T, 1)
            v_obs = ptu.from_numpy(v.reshape([r.shape[0], r.shape[1], 1]))

            (
                S_sampled,
                SP_sampled,
                A_sampled,
                R_sampled,
                D_sampled,
                V_sampled,
            ) = self.preprocess_sac(
                x_obs, r_obs, a_obs, d_obs=d_obs, v_obs=v_obs, seq_len=seq_len
            )

            loss_v, loss_a, loss_q = self.train_rl_sac_(
                S_sampled=S_sampled,
                SP_sampled=SP_sampled,
                A_sampled=A_sampled,
                R_sampled=R_sampled,
                D_sampled=D_sampled,
                V_sampled=V_sampled,
            )

        return loss_v, loss_a, loss_q
