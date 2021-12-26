import torch
from torch.autograd import Variable
import numpy as np
from torch import distributions as dis
from copy import deepcopy
import torch.nn as nn

EPS = 1e-6  # Avoid NaN (prevents division by zero or log of zero)
# CAP the standard deviation of the actor
LOG_STD_MAX = 2
LOG_STD_MIN = -20
REG = 1e-3  # regularization of the actor


class SAC(nn.Module):
    def __init__(
        self,
        input_size,
        action_size,
        gamma=0.99,
        reward_scale=1,
        beta_h="auto_1.0",
        policy_layers=[256, 256],
        value_layers=[256, 256],
        lr=3e-4,
        act_fn=nn.ReLU,
    ):

        super(SAC, self).__init__()

        self.input_size = input_size
        self.action_size = action_size
        self.reward_scale = reward_scale
        self.beta_h = beta_h
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
            # Force conversion to float
            # this will throw an error if a malformed string (different from 'auto')
            # is passed
            self.beta_h = float(self.beta_h)

        self.gamma = gamma

        self.policy_layers = policy_layers
        self.value_layers = value_layers

        # policy network
        self.s2mua = nn.ModuleList()
        last_layer_size = self.input_size
        for layer_size in self.policy_layers:
            self.s2mua.append(nn.Linear(last_layer_size, layer_size, bias=True))
            last_layer_size = layer_size
            self.s2mua.append(act_fn())
        self.s2mua.append(nn.Linear(last_layer_size, self.action_size, bias=True))

        self.f_s2mua = nn.Sequential(*self.s2mua)

        self.s2log_siga = nn.ModuleList()
        last_layer_size = self.input_size
        for layer_size in self.policy_layers:
            self.s2log_siga.append(nn.Linear(last_layer_size, layer_size, bias=True))
            last_layer_size = layer_size
            self.s2log_siga.append(act_fn())
        self.s2log_siga.append(nn.Linear(last_layer_size, self.action_size, bias=True))

        self.f_s2log_siga = nn.Sequential(*self.s2log_siga)

        # V network
        self.s2v = nn.ModuleList()
        last_layer_size = self.input_size
        for layer_size in self.value_layers:
            self.s2v.append(nn.Linear(last_layer_size, layer_size, bias=True))
            last_layer_size = layer_size
            self.s2v.append(act_fn())
        self.s2v.append(nn.Linear(last_layer_size, 1, bias=True))

        self.f_s2v = nn.Sequential(*self.s2v)

        # Q network 1
        self.sa2q1 = nn.ModuleList()
        last_layer_size = self.input_size + self.action_size
        for layer_size in self.value_layers:
            self.sa2q1.append(nn.Linear(last_layer_size, layer_size, bias=True))
            last_layer_size = layer_size
            self.sa2q1.append(act_fn())
        self.sa2q1.append(nn.Linear(last_layer_size, 1, bias=True))

        self.f_sa2q1 = nn.Sequential(*self.sa2q1)

        # Q network 2
        self.sa2q2 = nn.ModuleList()
        last_layer_size = self.input_size + self.action_size
        for layer_size in self.value_layers:
            self.sa2q2.append(nn.Linear(last_layer_size, layer_size, bias=True))
            last_layer_size = layer_size
            self.sa2q2.append(act_fn())
        self.sa2q2.append(nn.Linear(last_layer_size, 1, bias=True))

        self.f_sa2q2 = nn.Sequential(*self.sa2q2)

        # target V network
        self.s2v_tar = nn.ModuleList()
        last_layer_size = self.input_size
        for layer_size in self.value_layers:
            self.s2v_tar.append(nn.Linear(last_layer_size, layer_size, bias=True))
            last_layer_size = layer_size
            self.s2v_tar.append(act_fn())
        self.s2v_tar.append(nn.Linear(last_layer_size, 1, bias=True))

        self.f_s2v_tar = nn.Sequential(*self.s2v_tar)

        # synchronizing target V network and V network
        state_dict_tar = self.f_s2v_tar.state_dict()
        state_dict = self.f_s2v.state_dict()
        for key in list(self.f_s2v.state_dict().keys()):
            state_dict_tar[key] = state_dict[key]
        self.f_s2v_tar.load_state_dict(state_dict_tar)

        self.optimizer_a = torch.optim.Adam(
            [*self.f_s2mua.parameters(), *self.f_s2log_siga.parameters()], lr=lr
        )
        self.optimizer_v = torch.optim.Adam(
            [
                *self.f_s2v.parameters(),
                *self.f_sa2q1.parameters(),
                *self.f_sa2q2.parameters(),
            ],
            lr=lr,
        )
        if isinstance(self.beta_h, str):
            self.optimizer_e = torch.optim.Adam(
                [self.log_beta_h], lr=lr
            )  # optimizer for beta_h

        self.mse_loss = nn.MSELoss()

    def get_v(self, S):
        pass

    def get_q(self, S, A):
        pass

    def sample_z(self, mu, sig):
        # Using reparameterization trick to sample from a gaussian
        eps = Variable(torch.randn_like(mu))
        return mu + sig * eps

    def select(self, S, action_return="normal"):
        # output action

        if isinstance(S, np.ndarray):
            S = torch.from_numpy(S.astype(np.float32))

        mua = self.f_s2mua(S)
        siga = torch.exp(self.f_s2log_siga(S))

        a = np.tanh(self.sample_z(mua, siga).cpu().detach()).numpy()

        if action_return == "mean":
            return np.tanh(mua.cpu().detach().numpy())
        else:
            return a

    def sample_action(self, S, detach=False):
        # output action

        mua = self.f_s2mua(S)
        siga = torch.exp(self.f_s2log_siga(S).clamp(LOG_STD_MIN, LOG_STD_MAX))

        if detach:
            return (
                torch.tanh(self.sample_z(mua, siga)).cpu().detach(),
                mua.cpu().detach(),
                siga.cpu().detach(),
            )
        else:
            return torch.tanh(self.sample_z(mua, siga)), mua, siga

    def learn(self, S, SP, R, A, D, V, computation="explicit", grad_clip=False):

        if isinstance(S, np.ndarray):
            S = torch.from_numpy(S)

        if isinstance(SP, np.ndarray):
            SP = torch.from_numpy(SP)

        if isinstance(R, np.ndarray):
            R = torch.from_numpy(R)

        if isinstance(A, np.ndarray):
            A = torch.from_numpy(A)

        if isinstance(D, np.ndarray):
            D = torch.from_numpy(D)

        if isinstance(V, np.ndarray):
            V = torch.from_numpy(V)

        gamma = self.gamma
        reward_scale = self.reward_scale
        if isinstance(self.beta_h, str):
            beta_h = torch.exp(self.log_beta_h).data
        else:
            beta_h = np.float32(self.beta_h)

        S = S.data  # shape: batch_size x num_steps x n_neurons

        mua_tensor = self.f_s2mua(S)
        siga_tensor = torch.exp(self.f_s2log_siga(S).clamp(LOG_STD_MIN, LOG_STD_MAX))
        v_tensor = self.f_s2v(S)
        vp_tensor = self.f_s2v_tar(SP)
        q_tensor_1 = self.f_sa2q1(torch.cat((S, A), dim=-1))
        q_tensor_2 = self.f_sa2q2(torch.cat((S, A), dim=-1))

        # ------ explicit computing---------------
        if computation == "explicit":
            # ------ loss_v ---------------
            sampled_u = self.sample_z(mua_tensor.data, siga_tensor.data).data
            sampled_a = torch.tanh(sampled_u)

            sampled_q = torch.min(
                self.f_sa2q1(torch.cat((S, sampled_a), dim=-1)).data,
                self.f_sa2q2(torch.cat((S, sampled_a), dim=-1)).data,
            )

            q_exp = sampled_q

            log_pi_exp = torch.sum(
                -(mua_tensor.data - sampled_u.data).pow(2)
                / (siga_tensor.data.pow(2))
                / 2
                - torch.log(siga_tensor.data * torch.tensor(2.5066)).clamp(
                    LOG_STD_MIN, LOG_STD_MAX
                ),
                dim=-1,
                keepdim=True,
            )
            log_pi_exp -= torch.sum(
                torch.log(1.0 - sampled_a.pow(2) + EPS), dim=-1, keepdim=True
            )

            v_tar = (q_exp - beta_h * log_pi_exp.data).detach().data

            loss_v = 0.5 * self.mse_loss(v_tensor * V, v_tar * V)

            loss_q = 0.5 * self.mse_loss(
                q_tensor_1 * V,
                (reward_scale * R + (1 - D) * gamma * vp_tensor.detach().data) * V,
            ) + 0.5 * self.mse_loss(
                q_tensor_2 * V,
                (reward_scale * R + (1 - D) * gamma * vp_tensor.detach().data) * V,
            )

            loss_critic = loss_v + loss_q

            # -------- loss_a ---------------

            sampled_u = Variable(
                self.sample_z(mua_tensor.data, siga_tensor.data), requires_grad=True
            )
            sampled_a = torch.tanh(sampled_u)

            Q_tmp = torch.min(
                self.f_sa2q1(torch.cat((S, torch.tanh(sampled_u)), dim=-1)),
                self.f_sa2q2(torch.cat((S, torch.tanh(sampled_u)), dim=-1)),
            )
            Q_tmp.backward(torch.ones_like(Q_tmp))

            PQPU = sampled_u.grad.data  # \frac{\partial Q}{\partial a}

            eps = (sampled_u.data - mua_tensor.data) / (
                siga_tensor.data
            )  # action noise quantity
            a = sampled_a.data  # action quantity

            grad_mua = (beta_h * (2 * a) - PQPU).data * V.repeat_interleave(
                a.size()[-1], dim=-1
            ) + REG * mua_tensor * V.repeat_interleave(a.size()[-1], dim=-1)
            grad_siga = (
                -beta_h / (siga_tensor.data + EPS) + 2 * beta_h * a * eps - PQPU * eps
            ).data * V.repeat_interleave(
                a.size()[-1], dim=-1
            ) + REG * siga_tensor * V.repeat_interleave(
                a.size()[-1], dim=-1
            )

            self.optimizer_v.zero_grad()
            loss_critic.backward()
            if grad_clip:
                nn.utils.clip_grad_norm_(
                    [
                        *self.f_s2v.parameters(),
                        *self.f_sa2q1.parameters(),
                        *self.f_sa2q2.parameters(),
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
                    [*self.f_s2log_siga.parameters(), *self.f_s2mua.parameters()], 1.0
                )
            self.optimizer_a.step()

        # Using Torch API for computing log_prob
        # ------ implicit computing---------------

        elif computation == "implicit":
            # --------- loss_v ------------
            mu_prob = dis.Normal(mua_tensor, siga_tensor)

            sampled_u = mu_prob.sample()
            sampled_a = torch.tanh(sampled_u)

            log_pi_exp = (
                torch.sum(
                    mu_prob.log_prob(sampled_u).clamp(LOG_STD_MIN, LOG_STD_MAX),
                    dim=-1,
                    keepdim=True,
                )
                - torch.sum(torch.log(1 - sampled_a.pow(2) + EPS), dim=-1, keepdim=True)
            )

            sampled_q = torch.min(
                self.f_sa2q1(torch.cat((S, sampled_a), dim=-1)).data,
                self.f_sa2q2(torch.cat((S, sampled_a), dim=-1)).data,
            )
            q_exp = sampled_q

            v_tar = (q_exp - beta_h * log_pi_exp.data).detach().data

            loss_v = 0.5 * self.mse_loss(v_tensor * V, v_tar * V)

            loss_q = 0.5 * self.mse_loss(
                q_tensor_1 * V,
                (reward_scale * R + (1 - D) * gamma * vp_tensor.detach().data) * V,
            ) + 0.5 * self.mse_loss(
                q_tensor_2 * V,
                (reward_scale * R + (1 - D) * gamma * vp_tensor.detach().data) * V,
            )

            loss_critic = loss_v + loss_q

            # ----------- loss_a ---------------
            mu_prob = dis.Normal(mua_tensor, siga_tensor)

            sampled_u = mu_prob.rsample()
            sampled_a = torch.tanh(sampled_u)

            log_pi_exp = (
                torch.sum(
                    mu_prob.log_prob(sampled_u).clamp(LOG_STD_MIN, LOG_STD_MAX),
                    dim=-1,
                    keepdim=True,
                )
                - torch.sum(torch.log(1 - sampled_a.pow(2) + EPS), dim=-1, keepdim=True)
            )

            loss_a = torch.mean(
                beta_h * log_pi_exp * V
                - torch.min(
                    self.f_sa2q1(torch.cat((S, sampled_a), dim=-1)),
                    self.f_sa2q2(torch.cat((S, sampled_a), dim=-1)),
                )
                * V
            ) + REG / 2 * (
                torch.mean(
                    (
                        siga_tensor
                        * V.repeat_interleave(siga_tensor.size()[-1], dim=-1)
                    ).pow(2)
                )
                + torch.mean(
                    (
                        mua_tensor * V.repeat_interleave(mua_tensor.size()[-1], dim=-1)
                    ).pow(2)
                )
            )

            self.optimizer_v.zero_grad()
            loss_critic.backward()
            if grad_clip:
                nn.utils.clip_grad_norm_(
                    [
                        *self.f_s2v.parameters(),
                        *self.f_sa2q1.parameters(),
                        *self.f_sa2q2.parameters(),
                    ],
                    1.0,
                )
            self.optimizer_v.step()

            self.optimizer_a.zero_grad()
            loss_a.backward()
            if grad_clip:
                nn.utils.clip_grad_value_(
                    [*self.f_s2log_siga.parameters(), *self.f_s2mua.parameters()], 1.0
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
        state_dict_tar = self.f_s2v_tar.state_dict()
        state_dict = self.f_s2v.state_dict()
        for key in list(self.f_s2v.state_dict().keys()):
            state_dict_tar[key] = 0.995 * state_dict_tar[key] + 0.005 * state_dict[key]
        self.f_s2v_tar.load_state_dict(state_dict_tar)

        if computation == "implicit":
            return loss_v.item(), loss_a.item(), loss_a.item(), loss_q.item()
        elif computation == "explicit":
            return (
                loss_v.item(),
                torch.mean(grad_mua).item(),
                torch.mean(grad_siga).item(),
                loss_q.item(),
            )


class SACRNN(nn.Module):
    def __init__(
        self,
        input_size,
        action_size,
        gamma=0.99,
        reward_scale=1,
        beta_h="auto_1.0",
        shared_layers=256,
        output_layers=256,
        lr=3e-4,
    ):
        super(SACRNN, self).__init__()

        self.input_size = input_size
        self.action_size = action_size
        self.reward_scale = reward_scale
        self.beta_h = beta_h
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
            # Force conversion to float
            # this will throw an error if a malformed string (different from 'auto')
            # is passed
            self.beta_h = float(self.beta_h)

        self.gamma = gamma

        self.shared_layers = shared_layers
        self.output_layers = output_layers

        # shared lstm
        self.rnn = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.shared_layers,
            num_layers=1,
            batch_first=False,
            bias=True,
        )

        self.out_s2mua = nn.Sequential(
            nn.Linear(self.shared_layers, self.output_layers),
            nn.ReLU(),
            nn.Linear(self.output_layers, self.output_layers),
            nn.ReLU(),
            nn.Linear(self.output_layers, self.action_size),
        )

        self.out_s2log_siga = nn.Sequential(
            nn.Linear(self.shared_layers, self.output_layers),
            nn.ReLU(),
            nn.Linear(self.output_layers, self.output_layers),
            nn.ReLU(),
            nn.Linear(self.output_layers, self.action_size),
        )

        self.out_s2v = nn.Sequential(
            nn.Linear(self.shared_layers, self.output_layers),
            nn.ReLU(),
            nn.Linear(self.output_layers, self.output_layers),
            nn.ReLU(),
            nn.Linear(self.output_layers, 1),
        )

        self.out_sa2q1 = nn.Sequential(
            nn.Linear(self.shared_layers + self.action_size, self.output_layers),
            nn.ReLU(),
            nn.Linear(self.output_layers, self.output_layers),
            nn.ReLU(),
            nn.Linear(self.output_layers, 1),
        )

        self.out_sa2q2 = nn.Sequential(
            nn.Linear(self.shared_layers + self.action_size, self.output_layers),
            nn.ReLU(),
            nn.Linear(self.output_layers, self.output_layers),
            nn.ReLU(),
            nn.Linear(self.output_layers, 1),
        )

        self.out_s2v_tar = nn.Sequential(
            nn.Linear(self.shared_layers, self.output_layers),
            nn.ReLU(),
            nn.Linear(self.output_layers, self.output_layers),
            nn.ReLU(),
            nn.Linear(self.output_layers, 1),
        )

        # synchronizing target V network and V network

        state_dict_tar = self.out_s2v_tar.state_dict()
        state_dict = self.out_s2v.state_dict()
        for key in list(self.out_s2v.state_dict().keys()):
            state_dict_tar[key] = state_dict[key]
        self.out_s2v_tar.load_state_dict(state_dict_tar)

        self.optimizer_a = torch.optim.Adam(self.parameters(), lr=lr)
        self.optimizer_v = torch.optim.Adam(self.parameters(), lr=lr)
        if isinstance(self.beta_h, str):
            self.optimizer_e = torch.optim.Adam(
                [self.log_beta_h], lr=lr
            )  # optimizer for beta_h

        self.mse_loss = nn.MSELoss()

        # hidden states of the networks
        self.hc = (None, None)

    def get_v(self, S):
        pass

    def get_q(self, S, A):
        pass

    def sample_z(self, mu, sig):
        # Using reparameterization trick to sample from a gaussian
        eps = Variable(torch.randn_like(mu))
        return mu + sig * eps

    def init_episode(self, S0, action_return="normal"):

        if isinstance(S0, np.ndarray):
            S = torch.from_numpy(S0.astype(np.float32)).view(1, -1, self.input_size)

        output, self.hc = self.rnn(S)

        mua = self.out_s2mua(output)
        siga = torch.exp(self.out_s2log_siga(output).clamp(LOG_STD_MIN, LOG_STD_MAX))

        a = np.tanh(self.sample_z(mua, siga).cpu().detach()).numpy()

        if action_return == "mean":
            return np.tanh(mua.cpu().detach().numpy())
        else:
            return a

    def select(self, S, action_return="normal"):
        # output action

        if isinstance(S, np.ndarray):
            S = torch.from_numpy(S.astype(np.float32)).view(1, -1, self.input_size)

        output, self.hc = self.rnn(S, self.hc)

        mua = self.out_s2mua(output)
        siga = torch.exp(self.out_s2log_siga(output).clamp(LOG_STD_MIN, LOG_STD_MAX))

        a = np.tanh(self.sample_z(mua, siga).cpu().detach()).numpy()

        if action_return == "mean":
            return np.tanh(mua.cpu().detach().numpy())
        else:
            return a

    def learn(
        self, S, SP, R, A, D, V, seq_len=64, computation="explicit", grad_clip=False
    ):

        if isinstance(S, np.ndarray):
            S = torch.from_numpy(S)

        if isinstance(SP, np.ndarray):
            SP = torch.from_numpy(SP)

        if isinstance(R, np.ndarray):
            R = torch.from_numpy(R)

        if isinstance(A, np.ndarray):
            A = torch.from_numpy(A)

        if isinstance(D, np.ndarray):
            D = torch.from_numpy(D)

        if isinstance(V, np.ndarray):
            V = torch.from_numpy(V)

        S_sampled = torch.zeros(
            [S.size()[0], seq_len, S.size()[-1]], dtype=torch.float32
        )
        SP_sampled = torch.zeros(
            [SP.size()[0], seq_len, SP.size()[-1]], dtype=torch.float32
        )
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

        stps_burnin = 40

        for b in range(S.size()[0]):
            v = V.cpu().numpy().reshape([S.size()[0], S.size()[1]])
            stps = np.sum(v[b], axis=0).astype(int)
            start_index = np.random.randint(
                -seq_len + 1, stps - 1
            )  # TODO: why sample from negatives?

            for tmp, TMP in zip(
                (S_sampled, A_sampled, D_sampled, R_sampled, V_sampled, SP_sampled),
                (S, A, D, R, V, SP),
            ):
                # select the [max(0, start_idx), min(start_idx+seq_len, stps)] sub-traj

                sub_seq_start = max(0, start_index)
                sub_seq_end = min(start_index + seq_len, stps)
                tmp[b, : sub_seq_end - sub_seq_start] = TMP[
                    b, sub_seq_start:sub_seq_end
                ]

            init_hc = (
                torch.zeros([1, S.size()[0], self.shared_layers], dtype=torch.float32),
                torch.zeros([1, S.size()[0], self.shared_layers], dtype=torch.float32),
            )

            if start_index < 1:
                pass
            else:  # use the previous stps_burnin sub-traj before selected sub-traj as burn-in
                _, hcs = self.rnn(
                    S[
                        b : b + 1, max(0, start_index - stps_burnin) : start_index
                    ].transpose(0, 1)
                )
                init_hc[0][:, b, :] = hcs[0][:, 0, :]
                init_hc[1][:, b, :] = hcs[1][:, 0, :]

        S_sampled = S_sampled.transpose(
            0, 1
        ).data  # new shape: num_steps x batch_size x n_neurons
        SP_sampled = SP_sampled.transpose(0, 1).data
        A_sampled = A_sampled.transpose(0, 1).data
        R_sampled = R_sampled.transpose(0, 1).data
        V_sampled = V_sampled.transpose(0, 1).data
        D_sampled = D_sampled.transpose(0, 1).data

        gamma = self.gamma
        reward_scale = self.reward_scale

        if isinstance(self.beta_h, str):
            beta_h = torch.exp(self.log_beta_h).data
        else:
            beta_h = np.float32(self.beta_h)

        SS_sampled = torch.cat((S_sampled[0:1], SP_sampled), dim=0)
        output, _ = self.rnn(
            SS_sampled, init_hc
        )  # NOTE: no reward or action input ot RNN!

        q_tensor_1 = self.out_sa2q1(torch.cat((output[:-1], A_sampled), dim=-1))
        q_tensor_2 = self.out_sa2q2(torch.cat((output[:-1], A_sampled), dim=-1))
        mua_tensor = self.out_s2mua(output[:-1])
        siga_tensor = torch.exp(
            self.out_s2log_siga(output[:-1]).clamp(LOG_STD_MIN, LOG_STD_MAX)
        )
        v_tensor = self.out_s2v(output[:-1])
        vp_tensor = self.out_s2v_tar(output[1:])

        # ------ explicit computing---------------
        if computation == "explicit":
            # ------ loss_v ---------------
            sampled_u = self.sample_z(mua_tensor.data, siga_tensor.data).data
            sampled_a = torch.tanh(sampled_u)

            output, _ = self.rnn(S_sampled)

            sampled_q1 = self.out_sa2q1(torch.cat((output, sampled_a), dim=-1)).data
            sampled_q2 = self.out_sa2q2(torch.cat((output, sampled_a), dim=-1)).data

            sampled_q = torch.min(sampled_q1, sampled_q2)

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

            output, _ = self.rnn(S_sampled)

            sampled_q1 = self.out_sa2q1(torch.cat((output, sampled_a), dim=-1))
            sampled_q2 = self.out_sa2q2(torch.cat((output, sampled_a), dim=-1))

            Q_tmp = torch.min(sampled_q1, sampled_q2)

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

            self.optimizer_v.zero_grad()
            loss_critic.backward(retain_graph=True)
            if grad_clip:
                nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            self.optimizer_v.step()

            self.optimizer_a.zero_grad()
            mua_tensor.backward(
                grad_mua / torch.ones_like(mua_tensor, dtype=torch.float32).sum(),
                retain_graph=True,
            )
            siga_tensor.backward(
                grad_siga / torch.ones_like(siga_tensor, dtype=torch.float32).sum()
            )
            if grad_clip:
                nn.utils.clip_grad_value_(self.parameters(), 1.0)
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

            output, _ = self.rnn(S_sampled)

            sampled_q1 = self.out_sa2q1(torch.cat((output, sampled_a), dim=-1)).data
            sampled_q2 = self.out_sa2q2(torch.cat((output, sampled_a), dim=-1)).data

            sampled_q = torch.min(sampled_q1, sampled_q2)
            q_exp = sampled_q

            v_tar = (q_exp - beta_h * log_pi_exp.data).detach().data

            loss_v = 0.5 * self.mse_loss(v_tensor * V_sampled, v_tar * V_sampled)
            # NOTE: this masked method uses constant denominator, which is inaccurate
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

            output, _ = self.rnn(S_sampled)

            sampled_q1 = self.out_sa2q1(torch.cat((output, sampled_a), dim=-1))
            sampled_q2 = self.out_sa2q2(torch.cat((output, sampled_a), dim=-1))

            log_pi_exp = torch.sum(
                mu_prob.log_prob(sampled_u), dim=-1, keepdim=True
            ) - torch.sum(torch.log(1 - sampled_a.pow(2) + EPS), dim=-1, keepdim=True)

            loss_a = torch.mean(
                beta_h * log_pi_exp * V_sampled
                - torch.min(sampled_q1, sampled_q2) * V_sampled
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

            total_loss = loss_critic + loss_a

            self.optimizer_v.zero_grad()
            total_loss.backward()
            if grad_clip:
                nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                nn.utils.clip_grad_value_(self.parameters(), 1.0)
            self.optimizer_v.step()

        # --------------------------------------------------------------------------

        # update entropy coefficient if required
        if isinstance(self.beta_h, str):
            self.optimizer_e.zero_grad()
            # NOTE: log_pi_exp does not masked! it still have invalid items
            loss_e = -torch.mean(
                self.log_beta_h * (log_pi_exp + self.target_entropy).data
            )
            loss_e.backward()
            self.optimizer_e.step()

        # update target V network

        state_dict_tar = self.out_s2v_tar.state_dict()
        state_dict = self.out_s2v.state_dict()
        for key in list(self.out_s2v.state_dict().keys()):
            state_dict_tar[key] = 0.995 * state_dict_tar[key] + 0.005 * state_dict[key]
        self.out_s2v_tar.load_state_dict(state_dict_tar)

        if computation == "implicit":
            return loss_v.item(), loss_a.item(), loss_a.item(), loss_q.item()
        elif computation == "explicit":
            return (
                loss_v.item(),
                torch.mean(grad_mua).item(),
                torch.mean(grad_siga).item(),
                loss_q.item(),
            )
