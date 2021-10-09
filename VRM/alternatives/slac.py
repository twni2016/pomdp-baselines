import torch
from torch.autograd import Variable
import numpy as np
from copy import deepcopy
import torch.nn as nn
from torch import distributions as dis

EPS = 1e-6  # Avoid NaN (prevents division by zero or log of zero)
# CAP the standard deviation of the actor
LOG_STD_MAX = 2
LOG_STD_MIN = -20
REG = 1e-3  # regularization of the actor


SIG_MIN = 1e-2  #


class SLAC(nn.Module):
    def __init__(
        self,
        input_size,
        action_size,
        d_layer=256,
        z_layer=32,
        lr_st=1e-4,
        lr_rl=3e-4,
        seq_len=8,
        beta_h="auto_1.0",
        model_act_fn=nn.ReLU,
        sigx="auto",
    ):
        """
        Variational Multi-Layer RNN model with Action Feedback, using soft actor-critic for reinforcement learning.
        :param input_size: int, size of input vector.
        :param action_size: int, size of action vector.
        :param d_layer: int, indicating how many hidden neurons (d) in each layer. (z^2 in the original SLAC paper)
        :param z_layer: int, indicating how many hidden variable neurons (z) in each layer. (z^1 in the original SLAC paper)
        :param beta_h: entropy coefficient (see https://spinningup.openai.com/en/latest/algorithms/sac.html)
        :param seq_len: sequence length
        :param sigx: standard deviation of observation prediction, can be a float or 'auto" (parameterized).
        """

        super(SLAC, self).__init__()

        # Network layer parameters
        self.input_size = input_size
        self.action_size = action_size
        self.d_layer = d_layer
        self.z_layer = z_layer
        self.batch = True
        self.sigx_value = sigx
        self.seq_len = seq_len
        self.device = torch.device("cpu")
        self.beta_h = beta_h
        self.hidden_size = 256  # used in the original paper
        self.target_entropy = -np.float32(action_size)

        if isinstance(self.beta_h, str) and self.beta_h.startswith("auto"):
            # Default initial value of beta_h when learned
            init_value = 1.0
            if "_" in self.beta_h:
                init_value = float(self.beta_h.split("_")[1])
                assert (
                    init_value > 0.0
                ), "The initial value of beta_h must be greater than 0"

            self.log_beta_h = torch.tensor(
                np.log(init_value).astype(np.float32), requires_grad=True
            )
        else:
            self.beta_h = np.float32(beta_h)
        # policy network
        self.f_s2mua = nn.Sequential(
            nn.Linear(input_size * seq_len, self.hidden_size, bias=True),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size, bias=True),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.action_size, bias=True),
        )

        self.f_s2log_siga = nn.Sequential(
            nn.Linear(input_size * seq_len, self.hidden_size, bias=True),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size, bias=True),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.action_size, bias=True),
        )

        # V network
        self.f_d2v = nn.Sequential(
            nn.Linear(self.d_layer, self.hidden_size, bias=True),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size, bias=True),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1, bias=True),
        )

        # Q networks (double q-learning)
        self.f_da2q1 = nn.Sequential(
            nn.Linear(self.d_layer + self.action_size, self.hidden_size, bias=True),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size, bias=True),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1, bias=True),
        )

        self.f_da2q2 = nn.Sequential(
            nn.Linear(self.d_layer + self.action_size, self.hidden_size, bias=True),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size, bias=True),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1, bias=True),
        )

        # target V network
        self.f_d2v_tar = nn.Sequential(
            nn.Linear(self.d_layer, self.hidden_size, bias=True),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size, bias=True),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1, bias=True),
        )

        # synchronizing target V network and V network
        state_dict_tar = self.f_d2v_tar.state_dict()
        state_dict = self.f_d2v.state_dict()
        for key in list(self.f_d2v.state_dict().keys()):
            state_dict_tar[key] = state_dict[key]
        self.f_d2v_tar.load_state_dict(state_dict_tar)

        # p = prior (generative model), q = posterier (inference model)
        # generative models
        self.f_da2muz_p = nn.Sequential(
            nn.Linear(self.d_layer + self.action_size, self.hidden_size, bias=True),
            model_act_fn(),
            nn.Linear(self.hidden_size, self.hidden_size, bias=True),
            model_act_fn(),
            nn.Linear(self.hidden_size, self.z_layer, bias=True),
        )

        self.f_da2sigz_p = nn.Sequential(
            nn.Linear(self.d_layer + self.action_size, self.hidden_size, bias=True),
            model_act_fn(),
            nn.Linear(self.hidden_size, self.hidden_size, bias=True),
            model_act_fn(),
            nn.Linear(self.hidden_size, self.z_layer, bias=True),
            nn.Softplus(),
        )

        self.f_zda2mud = nn.Sequential(
            nn.Linear(
                self.d_layer + self.action_size + self.z_layer,
                self.hidden_size,
                bias=True,
            ),
            model_act_fn(),
            nn.Linear(self.hidden_size, self.hidden_size, bias=True),
            model_act_fn(),
            nn.Linear(self.hidden_size, self.d_layer, bias=True),
        )

        self.f_zda2sigd = nn.Sequential(
            nn.Linear(
                self.d_layer + self.action_size + self.z_layer,
                self.hidden_size,
                bias=True,
            ),
            model_act_fn(),
            nn.Linear(self.hidden_size, self.hidden_size, bias=True),
            model_act_fn(),
            nn.Linear(self.hidden_size, self.d_layer, bias=True),
            nn.Softplus(),
        )

        self.f_z2mud_begin = nn.Sequential(
            nn.Linear(self.z_layer, self.hidden_size, bias=True),
            model_act_fn(),
            nn.Linear(self.hidden_size, self.hidden_size, bias=True),
            model_act_fn(),
            nn.Linear(self.hidden_size, self.d_layer, bias=True),
        )

        self.f_z2sigd_begin = nn.Sequential(
            nn.Linear(self.z_layer, self.hidden_size, bias=True),
            model_act_fn(),
            nn.Linear(self.hidden_size, self.hidden_size, bias=True),
            model_act_fn(),
            nn.Linear(self.hidden_size, self.d_layer, bias=True),
            nn.Softplus(),
        )

        self.f_zd2mux = nn.Sequential(
            nn.Linear(self.d_layer + self.z_layer, self.hidden_size, bias=True),
            model_act_fn(),
            nn.Linear(self.hidden_size, self.hidden_size, bias=True),
            model_act_fn(),
            nn.Linear(self.hidden_size, self.input_size, bias=True),
        )
        if sigx is "auto":
            self.f_zd2sigx = nn.Sequential(
                nn.Linear(self.d_layer + self.z_layer, self.hidden_size, bias=True),
                model_act_fn(),
                nn.Linear(self.hidden_size, self.hidden_size, bias=True),
                model_act_fn(),
                nn.Linear(self.hidden_size, self.input_size, bias=True),
                nn.Softplus(),
            )
        else:
            self.sigx = torch.tensor(float(sigx))  # according to the paper

        self.f_zda2mur = nn.Sequential(
            nn.Linear(
                2 * self.d_layer + 2 * self.z_layer + self.action_size,
                self.hidden_size,
                bias=True,
            ),
            model_act_fn(),
            nn.Linear(self.hidden_size, self.hidden_size, bias=True),
            model_act_fn(),
            nn.Linear(self.hidden_size, 1, bias=True),
        )

        self.f_zda2sigr = nn.Sequential(
            nn.Linear(
                2 * self.d_layer + 2 * self.z_layer + self.action_size,
                self.hidden_size,
                bias=True,
            ),
            model_act_fn(),
            nn.Linear(self.hidden_size, self.hidden_size, bias=True),
            model_act_fn(),
            nn.Linear(self.hidden_size, 1, bias=True),
            nn.Softplus(),
        )

        # inference models
        self.f_dxa2muz_q = nn.Sequential(
            nn.Linear(
                self.d_layer + self.input_size + self.action_size,
                self.hidden_size,
                bias=True,
            ),
            model_act_fn(),
            nn.Linear(self.hidden_size, self.hidden_size, bias=True),
            model_act_fn(),
            nn.Linear(self.hidden_size, self.z_layer, bias=True),
        )

        self.f_dxa2sigz_q = nn.Sequential(
            nn.Linear(
                self.d_layer + self.input_size + self.action_size,
                self.hidden_size,
                bias=True,
            ),
            model_act_fn(),
            nn.Linear(self.hidden_size, self.hidden_size, bias=True),
            model_act_fn(),
            nn.Linear(self.hidden_size, self.z_layer, bias=True),
            nn.Softplus(),
        )

        self.f_x2muz_q_begin = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size, bias=True),
            model_act_fn(),
            nn.Linear(self.hidden_size, self.hidden_size, bias=True),
            model_act_fn(),
            nn.Linear(self.hidden_size, self.z_layer, bias=True),
        )

        self.f_x2sigz_q_begin = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size, bias=True),
            model_act_fn(),
            nn.Linear(self.hidden_size, self.hidden_size, bias=True),
            model_act_fn(),
            nn.Linear(self.hidden_size, self.z_layer, bias=True),
            nn.Softplus(),
        )

        self.optimizer_st = torch.optim.Adam(self.parameters(), lr=lr_st)
        self.optimizer_a = torch.optim.Adam(
            [*self.f_s2mua.parameters(), *self.f_s2log_siga.parameters()], lr=lr_rl
        )
        self.optimizer_v = torch.optim.Adam(
            [
                *self.f_da2q1.parameters(),
                *self.f_da2q2.parameters(),
                *self.f_d2v.parameters(),
            ],
            lr=lr_rl,
        )
        if isinstance(self.log_beta_h, torch.Tensor):
            self.optimizer_e = torch.optim.Adam(
                [self.log_beta_h], lr=lr_rl
            )  # optimizer for beta_h

        self.mse_loss = nn.MSELoss()
        self.to(device=self.device)

    def sample_z(self, mu, sig):
        # Using reparameterization trick to sample from a gaussian
        eps = Variable(torch.randn_like(mu))
        return mu + sig * eps

    def train_st(self, x_obs, a_obs, r_obs, validity=None):
        """
        train the VRNN model using observations x_obs and executed actions a_obs.
        :param x_obs: observations, pytorch tensor, size = batch_size by (num_steps+1) by dim_obs.
        :param r_obs: rewards, pytorch tensor, size = batch_size by num_steps by 1.
        :param a_obs: executed actions, pytorch tensor, size = batch_size by num_steps by dim_action.
        :param validity: validity matrix for padding, pytorch tensor (elements are 1 or 0), size = batch_size by num_steps. if validity=None, there is no need for padding.
        :return: loss value
        """
        seq_len = self.seq_len

        if isinstance(x_obs, np.ndarray):
            x_obs = torch.from_numpy(x_obs)

        if isinstance(a_obs, np.ndarray):
            a_obs = torch.from_numpy(a_obs)

        if isinstance(r_obs, np.ndarray):
            r_obs = torch.from_numpy(r_obs)

        if isinstance(validity, np.ndarray):
            validity = torch.from_numpy(validity)

        if not validity is None:
            v = validity.cpu().numpy().reshape([x_obs.size()[0], x_obs.size()[1]])
            stps = np.sum(v, axis=1)
            max_stp = int(np.max(stps))

            x_obs = x_obs[:, :max_stp]
            a_obs = a_obs[:, :max_stp]
            r_obs = r_obs[:, :max_stp]

            validity = validity[:, :max_stp].reshape([x_obs.size()[0], x_obs.size()[1]])

        batch_size = x_obs.size()[0]

        if validity is None:  # no need for padding
            validity = torch.ones(
                [x_obs.size()[0], x_obs.size()[1]], requires_grad=False
            )

        x_obs = x_obs.data
        a_obs = a_obs.data
        r_obs = r_obs.data
        validity = validity.data

        # sample a minibatch of batch_size x seq_len
        x_sampled = torch.zeros(
            [x_obs.size()[0], seq_len, x_obs.size()[-1]], dtype=torch.float32
        )
        a_sampled = torch.zeros(
            [a_obs.size()[0], seq_len, a_obs.size()[-1]], dtype=torch.float32
        )
        v_sampled = torch.zeros([validity.size()[0], seq_len], dtype=torch.float32)
        r_sampled = torch.zeros([r_obs.size()[0], seq_len, 1], dtype=torch.float32)

        for b in range(x_obs.size()[0]):
            v = validity.cpu().numpy().reshape([x_obs.size()[0], x_obs.size()[1]])
            stps = np.sum(v[b], axis=0).astype(int)
            start_index = np.random.randint(-seq_len + 1, stps - 1)

            for tmp, TMP in zip(
                (x_sampled, a_sampled, v_sampled, r_sampled),
                (x_obs, a_obs, validity, r_obs),
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

        sigz_p_series = []
        sigz_q_series = []
        muz_p_series = []
        muz_q_series = []
        mux_pred_series = []
        sigx_pred_series = []
        mur_prev_pred_series = []
        sigr_prev_pred_series = []

        for stp in range(seq_len):

            curr_x_obs = x_sampled[:, stp]
            prev_a_obs = a_sampled[:, stp]

            if stp == 0:

                muz_p = torch.zeros([batch_size, self.z_layer], dtype=torch.float32)
                sigz_p = torch.ones([batch_size, self.z_layer], dtype=torch.float32)

                z_p = self.sample_z(muz_p, sigz_p).detach()  # first step no gradient

                muz_q = self.f_x2muz_q_begin(curr_x_obs)
                sigz_q = self.f_x2sigz_q_begin(curr_x_obs)

                z_q = self.sample_z(muz_q, sigz_q)

                mud = self.f_z2mud_begin(z_q)
                sigd = self.f_z2sigd_begin(z_q)

                d = self.sample_z(mud, sigd)

                mux_pred = self.f_zd2mux(torch.cat([z_p, d], dim=-1))
                if self.sigx_value == "auto":
                    sigx_pred = self.f_zd2sigx(torch.cat([z_p, d], dim=-1)) + SIG_MIN

                    # no r_pred at the first step
            else:
                muz_p = self.f_da2muz_p(torch.cat([prev_d, prev_a_obs], dim=-1))
                sigz_p = self.f_da2sigz_p(torch.cat([prev_d, prev_a_obs], dim=-1))

                z_p = self.sample_z(muz_p, sigz_p)

                muz_q = self.f_dxa2muz_q(
                    torch.cat([prev_d, curr_x_obs, prev_a_obs], dim=-1)
                )
                sigz_q = self.f_dxa2sigz_q(
                    torch.cat([prev_d, curr_x_obs, prev_a_obs], dim=-1)
                )

                z_q = self.sample_z(muz_q, sigz_q)

                mud = self.f_zda2mud(torch.cat([z_q, prev_d, prev_a_obs], dim=-1))
                sigd = self.f_zda2sigd(torch.cat([z_q, prev_d, prev_a_obs], dim=-1))

                d = self.sample_z(mud, sigd)

                mux_pred = self.f_zd2mux(torch.cat([z_p, d], dim=-1))
                if self.sigx_value == "auto":
                    sigx_pred = self.f_zd2sigx(torch.cat([z_p, d], dim=-1)) + SIG_MIN

                mur_prev_pred = self.f_zda2mur(
                    torch.cat([z_p, prev_z_q, d, prev_d, prev_a_obs], dim=-1)
                )
                sigr_prev_pred = (
                    self.f_zda2sigr(
                        torch.cat([z_p, prev_z_q, d, prev_d, prev_a_obs], dim=-1)
                    )
                    + SIG_MIN
                )

                mur_prev_pred_series.append(mur_prev_pred)
                sigr_prev_pred_series.append(sigr_prev_pred)

            prev_z_q = z_q
            prev_d = d

            muz_p_series.append(muz_p)
            sigz_p_series.append(sigz_p)

            muz_q_series.append(muz_q)
            sigz_q_series.append(sigz_q)

            mux_pred_series.append(mux_pred)
            if self.sigx_value == "auto":
                sigx_pred_series.append(sigx_pred)

        sig_p_tensor = torch.stack(sigz_p_series, dim=1)
        sig_q_tensor = torch.stack(sigz_q_series, dim=1)
        mu_p_tensor = torch.stack(muz_p_series, dim=1)
        mu_q_tensor = torch.stack(muz_q_series, dim=1)

        mux_pred_tensor = torch.stack(mux_pred_series, dim=1)
        if self.sigx_value == "auto":
            sigx_pred_tensor = torch.stack(sigx_pred_series, dim=1)

        mur_prev_pred_tensor = torch.stack(mur_prev_pred_series, dim=1)
        sigr_prev_pred_tensor = torch.stack(sigr_prev_pred_series, dim=1)

        KL = torch.mean(
            torch.mean(
                torch.log(sig_p_tensor)
                - torch.log(sig_q_tensor)
                + ((mu_p_tensor - mu_q_tensor).pow(2) + sig_q_tensor.pow(2))
                / (2.0 * sig_p_tensor.pow(2))
                - 0.5,
                dim=-1,
            )
            * v_sampled
        )

        # log likelihood term
        if not self.sigx_value == "auto":
            Log = torch.mean(
                torch.sum(
                    -torch.pow(mux_pred_tensor[:, :, :-1] - x_sampled[:, :, :-1], 2)
                    / torch.pow(self.sigx, 2)
                    / 2
                    - torch.log(self.sigx * 2.5066),
                    dim=-1,
                )
                * v_sampled
            )  # except reward
        else:
            Log = torch.mean(
                torch.sum(
                    -torch.pow(mux_pred_tensor[:, :, :-1] - x_sampled[:, :, :-1], 2)
                    / torch.pow(sigx_pred_tensor[:, :, :-1], 2)
                    / 2
                    - torch.log(sigx_pred_tensor[:, :, :-1] * 2.5066),
                    dim=-1,
                )
                * v_sampled
            )  # except reward

        Log += (
            torch.mean(
                torch.sum(
                    -torch.pow(mur_prev_pred_tensor - r_sampled[:, :-1], 2)
                    / torch.pow(sigr_prev_pred_tensor, 2)
                    / 2
                    - torch.log(sigr_prev_pred_tensor * 2.5066),
                    dim=-1,
                )
                * v_sampled[:, :-1]
            )
            * (seq_len - 1.0)
            / seq_len
        )  # reward

        Log /= self.input_size

        elbo = -KL + Log
        loss = -elbo

        self.optimizer_st.zero_grad()
        loss.backward()

        self.optimizer_st.step()

        # if np.random.rand() < 0.005:
        #     print(loss.cpu().item())

        return loss.cpu().item()

    def train_rl_sac(
        self,
        x_obs,
        s_0,
        r_obs,
        a_obs,
        gamma,
        d_obs=None,
        validity=None,
        equal_pad=True,
        reward_scale=1,
        beta_h="auto",
        computation="explicit",
        grad_clip=False,
    ):

        seq_len = self.seq_len

        if isinstance(x_obs, np.ndarray):
            x_obs = torch.from_numpy(x_obs)

        if isinstance(s_0, np.ndarray):
            s_0 = torch.from_numpy(s_0)

        if isinstance(a_obs, np.ndarray):
            a_obs = torch.from_numpy(a_obs)

        if isinstance(r_obs, np.ndarray):
            r_obs = torch.from_numpy(r_obs)

        if isinstance(validity, np.ndarray):
            validity = torch.from_numpy(validity)

        if isinstance(d_obs, np.ndarray):
            d_obs = torch.from_numpy(d_obs)

        if isinstance(self.beta_h, str):
            beta_h = torch.exp(self.log_beta_h)
        else:
            beta_h = self.beta_h

        if not validity is None:
            v = validity.cpu().numpy().reshape([x_obs.size()[0], x_obs.size()[1]])
            stps = np.sum(v, axis=1)
            max_stp = int(np.max(stps))

            x_obs = x_obs[:, :max_stp]
            a_obs = a_obs[:, :max_stp]
            r_obs = r_obs[:, :max_stp]
            d_obs = d_obs[:, :max_stp]
            validity = validity[:, :max_stp]
        else:
            max_stp = x_obs.size()[1]

        xs_obs = torch.zeros(
            [x_obs.size()[0], max_stp, x_obs.size()[-1] * seq_len]
        )  # input for the policy function (include reward)
        xs_obs[:, -1, -(x_obs.size()[-1]) : -1] = s_0

        if equal_pad:
            for tau in range(seq_len - 1):
                xs_obs[
                    :, -1, (x_obs.size()[-1]) * tau : (x_obs.size()[-1]) * (tau + 1)
                ] = xs_obs[:, -1, -(x_obs.size()[-1]) :]

        for tau in range(max_stp):
            xs_obs[:, tau, : -(x_obs.size()[-1])] = xs_obs[
                :, tau - 1, (x_obs.size()[-1]) :
            ]
            xs_obs[:, tau, -(x_obs.size()[-1]) :] = x_obs[:, tau, :]

        x_obs = x_obs.data.to(device=self.device)
        a_obs = a_obs.data.to(device=self.device)
        r_obs = r_obs.data.to(device=self.device)
        V = validity.data.to(device=self.device)

        d_series = []

        start_indices = np.zeros(x_obs.size()[0], dtype=int)

        x_sampled = torch.zeros(
            [x_obs.size()[0], seq_len + 1, x_obs.size()[-1]], dtype=torch.float32
        )  # +1 for SP
        a_sampled = torch.zeros(
            [a_obs.size()[0], seq_len + 1, a_obs.size()[-1]], dtype=torch.float32
        )
        XS_sampled = torch.zeros(
            [x_obs.size()[0], seq_len + 1, seq_len * x_obs.size()[-1]],
            dtype=torch.float32,
        )

        for b in range(x_obs.size()[0]):
            v = V.cpu().numpy().reshape([x_obs.size()[0], x_obs.size()[1]])
            stps = np.sum(v[b], axis=0).astype(int)
            start_indices[b] = np.random.randint(-seq_len + 1, stps - 1)
            start_index = start_indices[b]

            for tmp, TMP in zip(
                (x_sampled, a_sampled, XS_sampled), (x_obs, a_obs, xs_obs)
            ):

                if start_index < 0 and start_index + seq_len + 1 > stps:
                    tmp[b, :stps] = TMP[b, :stps]

                elif start_index < 0:
                    tmp[b, : (start_index + seq_len + 1)] = TMP[
                        b, : (start_index + seq_len + 1)
                    ]

                elif start_index + seq_len + 1 > stps:
                    tmp[b, : (stps - start_index)] = TMP[b, start_index:stps]

                else:
                    tmp[b] = TMP[b, start_index : (start_index + seq_len + 1)]

        # import time
        # t_start = time.time()

        for stp in range(seq_len + 1):
            curr_x_obs = x_sampled[:, stp]
            prev_a_obs = a_sampled[:, stp]

            if stp == 0:

                muz_q = self.f_x2muz_q_begin(curr_x_obs)
                sigz_q = self.f_x2sigz_q_begin(curr_x_obs)

                z_q = self.sample_z(muz_q, sigz_q)

                mud = self.f_z2mud_begin(z_q)
                sigd = self.f_z2sigd_begin(z_q)

                d = self.sample_z(mud, sigd)

            else:
                muz_q = self.f_dxa2muz_q(
                    torch.cat([prev_d, curr_x_obs, prev_a_obs], dim=-1)
                )
                sigz_q = self.f_dxa2sigz_q(
                    torch.cat([prev_d, curr_x_obs, prev_a_obs], dim=-1)
                )

                z_q = self.sample_z(muz_q, sigz_q)

                mud = self.f_zda2mud(torch.cat([z_q, prev_d, prev_a_obs], dim=-1))
                sigd = self.f_zda2sigd(torch.cat([z_q, prev_d, prev_a_obs], dim=-1))

                d = self.sample_z(mud, sigd)

            prev_d = d

            d_series.append(d)

        d_tensor = torch.stack(d_series, dim=1).detach().data

        XS_sampled = XS_sampled[:, :-1, :]
        S_sampled = d_tensor[:, :-1, :]
        SP_sampled = d_tensor[:, 1:, :]

        A = a_obs[:, 1:, :]
        R = r_obs[:, 1:, :]

        if d_obs is None:
            D = torch.zeros_like(R, requires_grad=False, dtype=torch.float32)
        else:
            D = d_obs[:, 1:, :]

        if validity is None:  # no need for padding
            V = torch.ones_like(R, requires_grad=False, dtype=torch.float32)
        else:
            V = V[:, 1:, :]

        A_sampled = torch.zeros(
            [A.size()[0], seq_len, A.size()[-1]], dtype=torch.float32
        )
        D_sampled = torch.zeros(
            [D.size()[0], seq_len, D.size()[-1]], dtype=torch.float32
        )
        R_sampled = torch.zeros(
            [R.size()[0], seq_len, R.size()[-1]], dtype=torch.float32
        )
        V_sampled = torch.zeros(
            [V.size()[0], seq_len, V.size()[-1]], dtype=torch.float32
        )

        for b in range(A.size()[0]):
            v = V.cpu().numpy().reshape([A.size()[0], A.size()[1]])
            stps = np.sum(v[b], axis=0).astype(int)
            start_index = start_indices[b]

            for tmp, TMP in zip(
                (A_sampled, D_sampled, R_sampled, V_sampled), (A, D, R, V)
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

        mua_tensor = self.f_s2mua(XS_sampled)
        siga_tensor = torch.exp(
            self.f_s2log_siga(XS_sampled).clamp(LOG_STD_MIN, LOG_STD_MAX)
        )
        v_tensor = self.f_d2v(S_sampled)
        vp_tensor = self.f_d2v_tar(SP_sampled).data
        q_tensor_1 = self.f_da2q1(torch.cat((S_sampled, A_sampled), dim=-1))
        q_tensor_2 = self.f_da2q2(torch.cat((S_sampled, A_sampled), dim=-1))

        # ------ explicit computing---------------
        if computation == "explicit":
            # ------ loss_v ---------------
            sampled_u = self.sample_z(mua_tensor.data, siga_tensor.data).data
            sampled_a = torch.tanh(sampled_u)

            sampled_q = torch.min(
                self.f_da2q1(torch.cat((S_sampled, sampled_a), dim=-1)).data,
                self.f_da2q2(torch.cat((S_sampled, sampled_a), dim=-1)).data,
            )

            q_exp = sampled_q
            log_pi_exp = torch.sum(
                -(mua_tensor.data - sampled_u.data).pow(2)
                / (siga_tensor.data.pow(2))
                / 2
                - torch.log(siga_tensor.data * torch.tensor(2.5066)),
                dim=-1,
                keepdim=True,
            )
            log_pi_exp -= torch.sum(
                torch.log(1.0 - sampled_a.pow(2) + EPS), dim=-1, keepdim=True
            )

            v_tar = (q_exp - beta_h * log_pi_exp.data).detach().data

            loss_v = 0.5 * self.mse_loss(v_tensor * V_sampled, v_tar * V_sampled)

            loss_q = 0.5 * self.mse_loss(
                q_tensor_1 * V_sampled,
                (
                    reward_scale * R_sampled
                    + (1 - D_sampled) * gamma * vp_tensor.detach().data
                )
                * V_sampled,
            ) + 0.5 * self.mse_loss(
                q_tensor_2 * V_sampled,
                (
                    reward_scale * R_sampled
                    + (1 - D_sampled) * gamma * vp_tensor.detach().data
                )
                * V_sampled,
            )

            loss_critic = loss_v + loss_q

            # -------- loss_a ---------------

            sampled_u = Variable(
                self.sample_z(mua_tensor.data, siga_tensor.data), requires_grad=True
            )
            sampled_a = torch.tanh(sampled_u)

            Q_tmp = torch.min(
                self.f_da2q1(torch.cat((S_sampled, torch.tanh(sampled_u)), dim=-1)),
                self.f_da2q2(torch.cat((S_sampled, torch.tanh(sampled_u)), dim=-1)),
            )
            Q_tmp.backward(torch.ones_like(Q_tmp))

            PQPU = sampled_u.grad.data  # \frac{\partial Q}{\partial a}

            eps = (sampled_u.data - mua_tensor.data) / (
                siga_tensor.data
            )  # action noise quantity
            a = sampled_a.data  # action quantity

            grad_mua = (beta_h * (2 * a) - PQPU).data * V_sampled.repeat_interleave(
                a.size()[-1], dim=-1
            ) + REG * mua_tensor * V_sampled.repeat_interleave(a.size()[-1], dim=-1)
            grad_siga = (
                -beta_h / (siga_tensor.data + EPS) + 2 * beta_h * a * eps - PQPU * eps
            ).data * V_sampled.repeat_interleave(
                a.size()[-1], dim=-1
            ) + REG * siga_tensor * V_sampled.repeat_interleave(
                a.size()[-1], dim=-1
            )

            # if np.random.rand() < 0.002:
            #     print("grad_mua = ", end="")
            #     print(grad_mua)
            #     print("grad_siga = ", end="")
            #     print(grad_siga)

            self.optimizer_v.zero_grad()
            loss_critic.backward()
            if grad_clip:
                nn.utils.clip_grad_norm_(
                    [
                        *self.f_d2v.parameters(),
                        *self.f_da2q1.parameters(),
                        *self.f_da2q2.parameters(),
                    ],
                    1.0,
                )
            self.optimizer_v.step()

            self.optimizer_a.zero_grad()
            mua_tensor.backward(
                grad_mua / torch.ones_like(mua_tensor, dtype=torch.float32).sum()
            )
            siga_tensor.backward(
                grad_siga / torch.ones_like(siga_tensor, dtype=torch.float32).sum()
            )
            if grad_clip:
                nn.utils.clip_grad_value_(
                    [*self.f_d2vara.parameters(), *self.f_d2mua.parameters()], 1.0
                )
            self.optimizer_a.step()

        # Using Torch API for computing log_prob
        # ------ implicit computing---------------

        elif computation == "implicit":
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
                (
                    reward_scale * R_sampled
                    + (1 - D_sampled) * gamma * vp_tensor.detach().data
                )
                * V_sampled,
            ) + 0.5 * self.mse_loss(
                q_tensor_2 * V_sampled,
                (
                    reward_scale * R_sampled
                    + (1 - D_sampled) * gamma * vp_tensor.detach().data
                )
                * V_sampled,
            )

            loss_critic = loss_v + loss_q

            # ----------- loss_a ---------------
            mu_prob = dis.Normal(mua_tensor, siga_tensor)

            sampled_u = mu_prob.rsample()
            sampled_a = torch.tanh(sampled_u)

            log_pi_exp = torch.sum(
                mu_prob.log_prob(sampled_u), dim=-1, keepdim=True
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
            loss_critic.backward()
            if grad_clip:
                nn.utils.clip_grad_norm_(
                    [
                        *self.f_d2v.parameters(),
                        *self.f_da2q1.parameters(),
                        *self.f_da2q2.parameters(),
                    ],
                    1.0,
                )
            self.optimizer_v.step()

            self.optimizer_a.zero_grad()
            loss_a.backward()
            if grad_clip:
                nn.utils.clip_grad_value_(
                    [*self.f_d2vara.parameters(), *self.f_d2mua.parameters()], 1.0
                )
            self.optimizer_a.step()
        # --------------------------------------------------------------------------

        # update entropy coefficient if required
        if isinstance(self.beta_h, str):
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

        self.f_d2v_tar.load_state_dict(state_dict_tar)

        if computation == "implicit":
            return loss_v.item(), loss_a.item(), loss_q.item()
        elif computation == "explicit":
            return (
                loss_v.item(),
                torch.mean(grad_mua).item() + torch.mean(grad_siga).item(),
                loss_q.item(),
            )

    def select(self, SS, action_return="normal"):
        # output action

        if isinstance(SS, np.ndarray):
            SS = (
                torch.from_numpy(SS.astype(np.float32))
                .view(1, self.seq_len * self.input_size)
                .to(device=self.device)
            )

        mua = self.f_s2mua(SS)
        siga = torch.exp(self.f_s2log_siga(SS).clamp(LOG_STD_MIN, LOG_STD_MAX))

        # if np.random.rand() < 0.005:
        #     print("mua = ", end="")
        #     print(mua)
        #     print("siga = ", end="")
        #     print(siga)
        #     print("beta_h = ", end="")
        #     print(torch.exp(self.log_beta_h).cpu().data.numpy())

        a = np.tanh(self.sample_z(mua, siga).cpu().detach()).numpy()

        if action_return == "mean":
            return np.tanh(mua.cpu().detach().numpy())
        else:
            return a
