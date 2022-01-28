import gym
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from utils import helpers as utl
from torchkit import pytorch_utils as ptu
from models.decoder import StateTransitionDecoder, RewardDecoder, TaskDecoder
from models.encoder import RNNEncoder


class VAE:
    def __init__(self, args):
        self.args = args

        self.initialize_encoder()
        self.initialize_decoder()
        self.initialize_optimizer()

    def initialize_encoder(self):
        # initialize RNN encoder -- self.encoder
        self.encoder = RNNEncoder(
            layers_before_gru=self.args.layers_before_aggregator,
            hidden_size=self.args.aggregator_hidden_size,
            layers_after_gru=self.args.layers_after_aggregator,
            task_embedding_size=self.args.task_embedding_size,
            action_size=self.args.action_dim,
            action_embed_size=self.args.action_embedding_size,
            state_size=self.args.obs_dim,
            state_embed_size=self.args.state_embedding_size,
            reward_size=1,
            reward_embed_size=self.args.reward_embedding_size,
        ).to(ptu.device)

    def initialize_decoder(self):
        task_embedding_size = self.args.task_embedding_size
        if self.args.disable_stochasticity_in_latent:
            task_embedding_size *= 2

        # initialize model decoders -- self.reward_decoder, self.state_decoder, self.task_decoder
        if self.args.decode_reward:
            # initialise reward decoder for VAE
            self.reward_decoder = RewardDecoder(
                layers=self.args.reward_decoder_layers,
                task_embedding_size=task_embedding_size,
                #
                state_size=self.args.obs_dim,
                state_embed_size=self.args.state_embedding_size,
                action_size=self.args.action_dim,
                action_embed_size=self.args.action_embedding_size,
                num_states=self.args.num_states,
                multi_head=self.args.multihead_for_reward,
                pred_type=self.args.rew_pred_type,
                input_prev_state=self.args.input_prev_state,
                input_action=self.args.input_action,
            ).to(ptu.device)
            # set reward function
            # if self.args.rew_loss_fn == 'BCE':
            #     self.rew_loss_fn = lambda in_, target: F.binary_cross_entropy(in_, target, reduction='none')
            # elif self.args.rew_loss_fn == 'FL':
            #     self.rew_loss_fn = ptu.FocalLoss()
            # else:
            #     raise NotImplementedError
        else:
            self.reward_decoder = None

        if self.args.decode_state:
            # initialise state decoder for VAE
            self.state_decoder = StateTransitionDecoder(
                task_embedding_size=task_embedding_size,
                layers=self.args.state_decoder_layers,
                action_size=self.args.action_dim,
                action_embed_size=self.args.action_embedding_size,
                state_size=self.args.obs_dim,
                state_embed_size=self.args.state_embedding_size,
                pred_type=self.args.state_pred_type,
            ).to(ptu.device)
        else:
            self.state_decoder = None

        if self.args.decode_task:
            env = gym.make(self.args.env_name)
            if self.args.task_pred_type == "task_description":
                task_dim = env.task_dim
            elif self.args.task_pred_type == "task_id":
                task_dim = env.num_tasks
            else:
                raise NotImplementedError
            self.task_decoder = TaskDecoder(
                task_embedding_size=task_embedding_size,
                layers=self.args.task_decoder_layers,
                task_dim=task_dim,
                pred_type=self.args.task_pred_type,
            ).to(ptu.device)
        else:
            self.task_decoder = None

    def initialize_optimizer(self):
        decoder_params = []
        if not self.args.disable_decoder:
            # initialise optimiser for decoder
            if self.args.decode_reward:
                decoder_params.extend(self.reward_decoder.parameters())
            if self.args.decode_state:
                decoder_params.extend(self.state_decoder.parameters())
            if self.args.decode_task:
                decoder_params.extend(self.task_decoder.parameters())
        # initialize optimizer
        self.optimizer = torch.optim.Adam(
            [*self.encoder.parameters(), *decoder_params], lr=self.args.vae_lr
        )

    def compute_task_reconstruction_loss(
        self, dec_embedding, dec_task, return_predictions=False
    ):
        # make some predictions and compute individual losses
        task_pred = self.task_decoder(dec_embedding)

        if self.args.task_pred_type == "task_id":
            env = gym.make(self.args.env_name)
            dec_task = env.task_to_id(dec_task)
            dec_task = dec_task.expand(task_pred.shape[:-1]).view(-1)
            # loss for the data we fed into encoder
            task_pred_shape = task_pred.shape
            loss_task = F.cross_entropy(
                task_pred.view(-1, task_pred.shape[-1]), dec_task, reduction="none"
            ).reshape(task_pred_shape[:-1])
        elif self.args.task_pred_type == "task_description":
            loss_task = (task_pred - dec_task).pow(2).mean(dim=1)

        if return_predictions:
            return loss_task, task_pred
        else:
            return loss_task

    def compute_state_reconstruction_loss(
        self,
        dec_embedding,
        dec_prev_obs,
        dec_next_obs,
        dec_actions,
        return_predictions=False,
    ):
        # make some predictions and compute individual losses
        if self.args.state_pred_type == "deterministic":
            obs_reconstruction = self.state_decoder(
                dec_embedding, dec_prev_obs, dec_actions
            )
            loss_state = (obs_reconstruction - dec_next_obs).pow(2).mean(dim=1)
        elif self.args.state_pred_type == "gaussian":
            state_pred = self.state_decoder(dec_embedding, dec_prev_obs, dec_actions)
            state_pred_mean = state_pred[:, : state_pred.shape[1] // 2]
            state_pred_std = torch.exp(0.5 * state_pred[:, state_pred.shape[1] // 2 :])
            m = torch.distributions.normal.Normal(state_pred_mean, state_pred_std)
            # TODO: check if this is correctly averaged
            loss_state = -m.log_prob(dec_next_obs).mean(dim=1)

        if return_predictions:
            return loss_state, obs_reconstruction
        else:
            return loss_state

    def compute_rew_reconstruction_loss(
        self,
        dec_embedding,
        dec_prev_obs,
        dec_next_obs,
        dec_actions,
        dec_rewards,
        return_predictions=False,
    ):
        """
        Computed the reward reconstruction loss
        (no reduction of loss is done here; sum/avg has to be done outside)
        """
        # make some predictions and compute individual losses
        if self.args.multihead_for_reward:
            if (
                self.args.rew_pred_type == "bernoulli"
                or self.args.rew_pred_type == "categorical"
            ):
                # loss for the data we fed into encoder
                p_rew = self.reward_decoder(dec_embedding, None)
                env = gym.make(self.args.env_name)
                indices = env.task_to_id(dec_next_obs).to(ptu.device)
                if indices.dim() < p_rew.dim():
                    indices = indices.unsqueeze(-1)
                rew_pred = p_rew.gather(dim=-1, index=indices)
                rew_target = (dec_rewards == 1).float()
                loss_rew = F.binary_cross_entropy(
                    rew_pred, rew_target, reduction="none"
                ).mean(dim=-1)
                # loss_rew = self.rew_loss_fn(rew_pred, rew_target).mean(dim=-1)
            elif self.args.rew_pred_type == "deterministic":
                raise NotImplementedError
                # p_rew = self.reward_decoder(dec_embedding, None)
                # env = gym.make(self.args.env_name)
                # indices = env.task_to_id(dec_next_obs)
                # loss_rew = F.mse_loss(p_rew.gather(1, indices.reshape(-1, 1)), dec_rewards, reduction='none').mean(
                #     dim=1)
            else:
                raise NotImplementedError
        else:
            if self.args.rew_pred_type == "bernoulli":
                rew_pred = self.reward_decoder(dec_embedding, dec_next_obs)
                loss_rew = F.binary_cross_entropy(
                    rew_pred, (dec_rewards == 1).float(), reduction="none"
                ).mean(dim=1)
            elif self.args.rew_pred_type == "deterministic":
                rew_pred = self.reward_decoder(
                    dec_embedding, dec_next_obs, dec_prev_obs, dec_actions
                )
                loss_rew = (rew_pred - dec_rewards).pow(2).mean(dim=1)
            elif self.args.rew_pred_type == "gaussian":
                rew_pred = self.reward_decoder(
                    dec_embedding, dec_next_obs, dec_prev_obs, dec_actions
                ).mean(dim=1)
                rew_pred_mean = rew_pred[:, : rew_pred.shape[1] // 2]
                rew_pred_std = torch.exp(0.5 * rew_pred[:, rew_pred.shape[1] // 2 :])
                m = torch.distributions.normal.Normal(rew_pred_mean, rew_pred_std)
                loss_rew = -m.log_prob(dec_rewards)
            else:
                raise NotImplementedError

        if return_predictions:
            return loss_rew, rew_pred
        else:
            return loss_rew

    def compute_kl_loss(self, latent_mean, latent_logvar, len_encoder):

        # -- KL divergence
        if self.args.kl_to_gauss_prior:
            kl_divergences = -0.5 * (
                1 + latent_logvar - latent_mean.pow(2) - latent_logvar.exp()
            ).sum(dim=1)
        else:
            gauss_dim = latent_mean.shape[-1]
            # add the gaussian prior
            all_means = torch.cat(
                (torch.zeros(1, latent_mean.shape[1]).to(ptu.device), latent_mean)
            )
            all_logvars = torch.cat(
                (torch.zeros(1, latent_logvar.shape[1]).to(ptu.device), latent_logvar)
            )
            # https://arxiv.org/pdf/1811.09975.pdf
            # KL(N(mu,E)||N(m,S)) = 0.5 * (log(|S|/|E|) - K + tr(S^-1 E) + (m-mu)^T S^-1 (m-mu)))
            mu = all_means[1:]
            m = all_means[:-1]
            logE = all_logvars[1:]
            logS = all_logvars[:-1]
            kl_divergences = 0.5 * (
                torch.sum(logS, dim=1)
                - torch.sum(logE, dim=1)
                - gauss_dim
                + torch.sum(1 / torch.exp(logS) * torch.exp(logE), dim=1)
                + ((m - mu) / torch.exp(logS) * (m - mu)).sum(dim=1)
            )

        if self.args.learn_prior:
            mask = torch.ones(len(kl_divergences))
            mask[0] = 0
            kl_divergences = kl_divergences * mask

        # returns, for each ELBO_t term, one KL (so H+1 kl's)
        if len_encoder is not None:
            return kl_divergences[len_encoder]
        else:
            return kl_divergences

    def compute_belief_reward(self, task_means, task_logvars, obs, next_obs, actions):
        """
        compute reward in the BAMDP by averaging over sampled latent embeddings - R+ = E[R(b)]
        """
        # sample multiple latent embeddings from posterior - (n_samples, n_processes, latent_dim)
        task_samples = self.encoder._sample_gaussian(
            task_means, task_logvars, self.args.num_belief_samples
        )
        if next_obs.dim() > 2:
            next_obs = next_obs.repeat(self.args.num_belief_samples, 1, 1)
            obs = (
                obs.repeat(self.args.num_belief_samples, 1, 1)
                if obs is not None
                else None
            )
            actions = (
                actions.repeat(self.args.num_belief_samples, 1, 1)
                if actions is not None
                else None
            )
        else:
            next_obs = next_obs.repeat(self.args.num_belief_samples, 1)
            obs = (
                obs.repeat(self.args.num_belief_samples, 1) if obs is not None else None
            )
            actions = (
                actions.repeat(self.args.num_belief_samples, 1)
                if actions is not None
                else None
            )
        # make some predictions and average
        if self.args.multihead_for_reward:
            if (
                self.args.rew_pred_type == "bernoulli"
            ):  # or self.args.rew_pred_type == 'categorical':
                p_rew = self.reward_decoder(task_samples, None).detach()
                # average over samples dimension to get R+
                p_rew = p_rew.mean(dim=0)
                env = gym.make(self.args.env_name)
                indices = env.task_to_id(next_obs).to(ptu.device)
                if indices.dim() < p_rew.dim():
                    indices = indices.unsqueeze(-1)
                rew_pred = p_rew.gather(dim=-1, index=indices)
            else:
                raise NotImplementedError
        else:
            if self.args.rew_pred_type == "deterministic":
                rew_pred = self.reward_decoder(task_samples, next_obs, obs, actions)
                rew_pred = rew_pred.mean(dim=0)
            else:
                raise NotImplementedError
        return rew_pred

    def load_model(self, device="cpu", **kwargs):
        if "encoder_path" in kwargs:
            self.encoder.load_state_dict(
                torch.load(kwargs["encoder_path"], map_location=device)
            )
        if "reward_decoder_path" in kwargs and self.reward_decoder is not None:
            self.reward_decoder.load_state_dict(
                torch.load(kwargs["reward_decoder_path"], map_location=device)
            )
        if "state_decoder_path" in kwargs and self.state_decoder is not None:
            self.state_decoder.load_state_dict(
                torch.load(kwargs["state_decoder_path"], map_location=device)
            )
        if "task_decoder_path" in kwargs and self.task_decoder is not None:
            self.task_decoder.load_state_dict(
                torch.load(kwargs["task_decoder_path"], map_location=device)
            )
