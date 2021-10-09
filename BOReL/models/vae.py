import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from torchkit import pytorch_utils as ptu
from .decoder import StateTransitionDecoder, RewardDecoder
from .encoder import RNNEncoder

from utils import logger


class VAE:
    """
    literally same as varibad's VAE
    https://github.com/lmzintgraf/varibad/blob/master/vae.py
    """

    def __init__(
        self,
        obs_dim,
        act_dim,
        task_embedding_size,
        encoder,
        decoder,
        optim,
    ):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.task_embedding_size = task_embedding_size

        self.initialize_encoder(**encoder)

        self.initialize_decoder(**decoder, **encoder)  # embedding_size

        self.initialize_optimizer(**optim)

    def initialize_encoder(
        self,
        layers_before_aggregator,
        aggregator_hidden_size,
        layers_after_aggregator,
        action_embedding_size,
        state_embedding_size,
        reward_embedding_size,
    ):
        # initialize RNN encoder -- self.encoder

        self.encoder = RNNEncoder(
            layers_before_gru=layers_before_aggregator,
            hidden_size=aggregator_hidden_size,
            layers_after_gru=layers_after_aggregator,
            task_embedding_size=self.task_embedding_size,
            action_size=self.act_dim,
            action_embed_size=action_embedding_size,
            state_size=self.obs_dim,
            state_embed_size=state_embedding_size,
            reward_size=1,
            reward_embed_size=reward_embedding_size,
        ).to(ptu.device)
        logger.log(self.encoder)

    def initialize_decoder(
        self,
        disable_stochasticity_in_latent=False,
        decode_reward=True,
        decode_state=True,
        state_embedding_size=None,
        action_embedding_size=None,
        # reward function
        reward_decoder_layers=[],
        rew_pred_type=None,
        input_prev_state=False,
        input_action=False,
        # state transition function
        state_decoder_layers=[],
        state_pred_type=None,
        **kwargs
    ):

        assert disable_stochasticity_in_latent == False
        self.disable_stochasticity_in_latent = disable_stochasticity_in_latent
        if disable_stochasticity_in_latent:
            decoder_task_embedding_size = 2 * self.task_embedding_size
        else:  # default false: sample z ~ encoder into decoder
            decoder_task_embedding_size = self.task_embedding_size

        # initialize model decoders -- self.reward_decoder, self.state_decoder, self.task_decoder
        self.decode_reward = decode_reward
        self.decode_state = decode_state
        # Occam's razor https://github.com/lmzintgraf/varibad/issues/3#issuecomment-700150431

        if self.decode_reward:
            # initialise reward decoder for VAE
            self.reward_decoder = RewardDecoder(
                layers=reward_decoder_layers,
                task_embedding_size=decoder_task_embedding_size,
                state_size=self.obs_dim,
                state_embed_size=state_embedding_size,
                action_size=self.act_dim,
                action_embed_size=action_embedding_size,
                pred_type=rew_pred_type,  # deterministic
                input_prev_state=input_prev_state,
                input_action=input_action,
            ).to(ptu.device)
            logger.log(self.reward_decoder)
        else:
            self.reward_decoder = None

        if self.decode_state:
            # initialise state decoder for VAE
            self.state_decoder = StateTransitionDecoder(
                task_embedding_size=decoder_task_embedding_size,
                layers=state_decoder_layers,
                action_size=self.act_dim,
                action_embed_size=action_embedding_size,
                state_size=self.obs_dim,
                state_embed_size=state_embedding_size,
                pred_type=state_pred_type,
            ).to(ptu.device)
            logger.log(self.state_decoder)
        else:
            self.state_decoder = None

    def initialize_optimizer(
        self,
        vae_lr,
        rew_loss_coeff,
        state_loss_coeff,
        kl_weight,
        kl_to_gauss_prior=False,
        train_by_batch=True,
    ):
        decoder_params = []
        # initialise optimiser for decoder
        if self.decode_reward:
            decoder_params.extend(self.reward_decoder.parameters())
        if self.decode_state:
            decoder_params.extend(self.state_decoder.parameters())
        # initialize optimizer
        self.optimizer = torch.optim.Adam(
            [*self.encoder.parameters(), *decoder_params], lr=vae_lr
        )

        self.rew_loss_coeff = rew_loss_coeff
        self.state_loss_coeff = state_loss_coeff
        self.kl_weight = kl_weight
        self.kl_to_gauss_prior = kl_to_gauss_prior
        self.train_by_batch = train_by_batch

    def compute_state_reconstruction_loss(
        self,
        dec_embedding,
        dec_prev_obs,
        dec_next_obs,
        dec_actions,
        return_predictions=False,
    ):
        # make some predictions and compute individual losses
        if self.state_decoder.pred_type == "deterministic":
            obs_reconstruction = self.state_decoder(
                dec_embedding, dec_prev_obs, dec_actions
            )
            loss_state = (
                (obs_reconstruction - dec_next_obs).pow(2).mean(dim=1)
            )  # (T+1, T, *, dim) -> (T+1, *, dim)
        elif self.state_decoder.pred_type == "gaussian":
            raise NotImplementedError
            state_pred = self.state_decoder(dec_embedding, dec_prev_obs, dec_actions)
            state_pred_mean = state_pred[:, : state_pred.shape[1] // 2]
            state_pred_std = torch.exp(0.5 * state_pred[:, state_pred.shape[1] // 2 :])
            m = torch.distributions.normal.Normal(state_pred_mean, state_pred_std)
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
        """shape: (T+1, T, *, dim), where * maybe None or B.
        Computed the reward reconstruction loss (avg over timesteps but not over ELBO_t!)
        """
        # make some predictions and compute individual losses
        if self.reward_decoder.pred_type == "deterministic":
            rew_pred = self.reward_decoder(
                dec_embedding, dec_next_obs, dec_prev_obs, dec_actions
            )
            loss_rew = (
                (rew_pred - dec_rewards).pow(2).mean(dim=1)
            )  # (T+1, T, *, 1) -> (T+1, *, 1)
        elif self.reward_decoder.pred_type == "gaussian":
            raise NotImplementedError
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
        """
        latent_mean, latent_logvar: (T+1, *, embed_size), where * maybe None or B.
        For each ELBO_t, KL(q(m_t) || q(m_{t-1})). NOTE: we set prior as PREVIOUS posterior
        for t=0, the prior is N(0,I)
        """
        # prior is N(0, I): KL(N(mu,sigma) || N(0,I)) = 1/2 * (mu**2 + sigma**2 - log(sigma**2) - 1)
        if self.kl_to_gauss_prior:
            kl_divergences = -0.5 * (
                1 + latent_logvar - latent_mean.pow(2) - latent_logvar.exp()
            ).sum(dim=-1)
        else:
            gauss_dim = latent_mean.shape[-1]
            # add the gaussian prior N(0,I): (T+1, *, embed_size) -> (T+2, *, embed_size)
            all_means = torch.cat((ptu.zeros((1, *latent_mean.shape[1:])), latent_mean))
            all_logvars = torch.cat(
                (ptu.zeros((1, *latent_logvar.shape[1:])), latent_logvar)
            )
            # KL(N(mu,E)||N(m,S)) = 0.5 * (log(|S|/|E|) - K + tr(S^-1 E) + (m-mu)^T S^-1 (m-mu)))
            mu = all_means[1:]
            m = all_means[:-1]
            logE = all_logvars[1:]
            logS = all_logvars[:-1]
            kl_divergences = 0.5 * (
                torch.sum(logS, dim=-1)
                - torch.sum(logE, dim=-1)
                - gauss_dim
                + torch.sum(1 / torch.exp(logS) * torch.exp(logE), dim=-1)
                + torch.sum((m - mu) / torch.exp(logS) * (m - mu), dim=-1)
            )

        # returns, for each ELBO_t term, one KL: KL(q(m_t) | p(m))
        if len_encoder is not None:
            return kl_divergences[len_encoder]  # (T+1, *) -> (T+1, *)
        else:
            return kl_divergences

    @torch.no_grad()
    def compute_belief_reward(
        self, num_belief_samples, task_means, task_logvars, obs, next_obs, actions
    ):
        """
        compute reward in the BAMDP by averaging over sampled latent embeddings - R+(s) = E{m~b}[R(s;m)]
        """
        # sample multiple latent embeddings from posterior - (num, embed_size)
        task_samples = self.encoder._sample_gaussian(
            task_means, task_logvars, num_belief_samples
        )
        if next_obs.dim() > 2:  # TODO: when?
            next_obs = next_obs.repeat(num_belief_samples, 1, 1)
            obs = obs.repeat(num_belief_samples, 1, 1) if obs is not None else None
            actions = (
                actions.repeat(num_belief_samples, 1, 1)
                if actions is not None
                else None
            )
        else:
            next_obs = next_obs.repeat(num_belief_samples, 1)  # (num, dim)
            obs = obs.repeat(num_belief_samples, 1) if obs is not None else None
            actions = (
                actions.repeat(num_belief_samples, 1) if actions is not None else None
            )
        # make some predictions and average
        rew_pred = self.reward_decoder(
            task_samples, next_obs, obs, actions  # (num, dim)
        )
        if self.reward_decoder.pred_type == "gaussian":
            rew_pred = rew_pred[:, [0]]  # mean
        rew_pred = rew_pred.mean(dim=0)  # (num,1) -> (1)
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

    def update(self, obs, actions, rewards, next_obs):
        """all the input has 3D shape of (seq_len, num_epsiodes, dim)
        Compute losses, update parameters and return the VAE losses
        Loss objective: \sum_t \mean_t' r(o_t' | m_t) + p(s_{t'+1} | s_t', a_t', m_t)
        for ANY t'=1,...,H, and ANY t=0,...,H. This is sum over ELBO_t and avg over timesteps inside ELBO_t
        """

        # pass through encoder (outputs will be: (max_traj_len+1) x number of rollouts x latent_dim -- includes the prior!)
        latent_samples, latent_mean, latent_logvar, _ = self.encoder(
            actions=actions,
            states=next_obs,
            rewards=rewards,
            hidden_state=None,
            return_prior=True,
        )
        # update_by_batch is ?x faster than split, but ?x larger in gpu memory.
        if self.train_by_batch:
            func = self.update_by_batch
        else:
            func = self.update_by_split
        return func(
            obs, actions, rewards, next_obs, latent_samples, latent_mean, latent_logvar
        )

    def update_by_split(
        self,
        obs,
        actions,
        rewards,
        next_obs,
        latent_samples,
        latent_mean,
        latent_logvar,
    ):
        trajectory_len, num_episodes, _ = obs.shape
        # get time-steps for ELBO computation (B, T+1)
        elbo_timesteps = np.repeat(
            np.arange(0, trajectory_len + 1).reshape(1, -1), num_episodes, axis=0
        )

        rew_recon_losses, state_recon_losses, kl_terms = [], [], []

        # for each episode we have in our batch
        for episode_idx in range(num_episodes):

            # get the embedding values. size: (traj_length+1, latent_dim) = (T+1, M) the +1 is for the prior
            curr_means = latent_mean[:, episode_idx, :]
            curr_logvars = latent_logvar[:, episode_idx, :]
            curr_samples = latent_samples[:, episode_idx, :]

            # select data from current rollout (result is (traj_length, obs_dim) = (T, O))
            curr_obs = obs[:, episode_idx, :]
            curr_next_obs = next_obs[:, episode_idx, :]
            curr_actions = actions[:, episode_idx, :]
            curr_rewards = rewards[:, episode_idx, :]

            num_latents = curr_samples.shape[0]  # includes the prior
            num_decodes = curr_obs.shape[0]

            # expand the latent to match the (x, y) pairs of the decoder -> (T+1, T, M)
            dec_embedding = (
                curr_samples.unsqueeze(0)
                .expand((num_decodes, *curr_samples.shape))
                .transpose(1, 0)
            )

            # expand the (x, y) pair of the encoder -> (T+1, T, dim)
            dec_obs = curr_obs.unsqueeze(0).expand((num_latents, *curr_obs.shape))
            dec_next_obs = curr_next_obs.unsqueeze(0).expand(
                (num_latents, *curr_next_obs.shape)
            )
            dec_actions = curr_actions.unsqueeze(0).expand(
                (num_latents, *curr_actions.shape)
            )
            dec_rewards = curr_rewards.unsqueeze(0).expand(
                (num_latents, *curr_rewards.shape)
            )

            # NOTE: decoder loss \sum_t \mean_t' r(o_t' | m_t) for ANY t'=1,...,H, and ANY t=0,...,H
            # cuz varibad assumes stationary m variable, so it should be consistent for all timestep predictions
            if self.decode_reward:  # all (T+1, T, dim)
                # compute reconstruction loss for this trajectory
                rrl = self.compute_rew_reconstruction_loss(
                    dec_embedding, dec_obs, dec_next_obs, dec_actions, dec_rewards
                )
                # sum along the trajectory which we decoded (sum in ELBO_t)
                rrl = rrl.sum(dim=1)  # (T+1, 1) -> (T+1)
                rew_recon_losses.append(rrl)
            if self.decode_state:
                srl = self.compute_state_reconstruction_loss(
                    dec_embedding, dec_obs, dec_next_obs, dec_actions
                )
                srl = srl.sum(dim=1)  # (T+1, dim) -> (T+1)
                state_recon_losses.append(srl)

            # kl term
            if not self.disable_stochasticity_in_latent:
                # compute the KL term for each ELBO term of the current trajectory
                kl = self.compute_kl_loss(
                    curr_means, curr_logvars, elbo_timesteps[episode_idx]
                )
                kl_terms.append(kl)  # (T+1)

        # sum the ELBO_t terms per task
        if self.decode_reward:
            rew_recon_losses = torch.stack(rew_recon_losses)  # (B, T+1)
            rew_recon_losses = rew_recon_losses.sum(dim=1)  # (B)
        else:
            rew_recon_losses = ptu.zeros(1)  # 0 -- but with option of .mean()

        if self.decode_state:
            state_recon_losses = torch.stack(state_recon_losses)  # (B, T+1)
            state_recon_losses = state_recon_losses.sum(dim=1)  # (B)
        else:
            state_recon_losses = ptu.zeros(1)

        if not self.disable_stochasticity_in_latent:
            kl_terms = torch.stack(kl_terms)  # (B, T+1)
            kl_terms = kl_terms.sum(dim=1)  # (B)
        else:
            kl_terms = ptu.zeros(1)

        # take average (this is the expectation over p(M), i.e. batch)
        loss = (
            self.rew_loss_coeff * rew_recon_losses
            + self.state_loss_coeff * state_recon_losses
            + self.kl_weight * kl_terms
        ).mean()  # (B) -> scalar

        # make sure we can compute gradients
        if not self.disable_stochasticity_in_latent:
            assert kl_terms.requires_grad
        if self.decode_reward:
            assert rew_recon_losses.requires_grad
        if self.decode_state:
            assert state_recon_losses.requires_grad

        # update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "vae_loss": loss.item(),
            "rew_rec_loss": rew_recon_losses.mean().item(),
            "state_rec_loss": state_recon_losses.mean().item(),
            "kl_term": kl_terms.mean().item(),
        }

    def update_by_batch(
        self,
        obs,
        actions,
        rewards,
        next_obs,
        latent_samples,
        latent_mean,
        latent_logvar,
    ):
        trajectory_len = obs.shape[0]  # (T, B, dim)
        num_latents = latent_samples.shape[
            0
        ]  # while latent is (T+1, B, dim) includes prior
        assert trajectory_len + 1 == num_latents

        # get time-steps for ELBO computation (T+1)
        elbo_timesteps = np.arange(0, trajectory_len + 1)

        # expand the latent to match the (x, y) pairs of the decoder -> (T, T+1, B, M) -> (T+1, T, B, M)
        dec_embedding = (
            latent_samples.unsqueeze(0)
            .expand((trajectory_len, *latent_samples.shape))
            .transpose(1, 0)
        )

        # expand the (x, y) pair of the encoder -> (T, B, dim) -> (T+1, T, B, dim)
        dec_obs = obs.unsqueeze(0).expand((num_latents, *obs.shape))
        dec_next_obs = next_obs.unsqueeze(0).expand((num_latents, *next_obs.shape))
        dec_actions = actions.unsqueeze(0).expand((num_latents, *actions.shape))
        dec_rewards = rewards.unsqueeze(0).expand((num_latents, *rewards.shape))

        # NOTE: decoder loss \sum_t \mean_t' r(o_t' | m_t) for ANY t'=1,...,H, and ANY t=0,...,H
        # cuz varibad assumes stationary m variable, so it should be consistent for all timestep predictions
        losses = 0.0
        stats = dict()

        if self.decode_reward:  # all (T+1, T, dim)
            # compute reconstruction loss for this trajectory
            rrl = self.compute_rew_reconstruction_loss(
                dec_embedding, dec_obs, dec_next_obs, dec_actions, dec_rewards
            )
            # sum along the trajectory which we decoded (sum in ELBO_t)
            rew_recon_losses = rrl.sum(-1).mean(-1).sum()  # (T+1, B, 1) -> scalar
            losses += self.rew_loss_coeff * rew_recon_losses
            stats["rew_rec_loss"] = rew_recon_losses.item()

        if self.decode_state:
            srl = self.compute_state_reconstruction_loss(
                dec_embedding, dec_obs, dec_next_obs, dec_actions
            )
            state_recon_losses = srl.sum(-1).mean(-1).sum()  # (T+1, B, dim) -> scalar
            losses += self.state_loss_coeff * state_recon_losses
            stats["state_rec_loss"] = state_recon_losses.item()

        # kl term
        if not self.disable_stochasticity_in_latent:
            # compute the KL term for each ELBO term of the current trajectory
            kl = self.compute_kl_loss(
                latent_mean, latent_logvar, elbo_timesteps
            )  # (T+1, B)
            kl = kl.mean(-1).sum()  # (T+1, B) -> scalar
            losses += self.kl_weight * kl
            stats["kl_term"] = kl.item()

        # update
        self.optimizer.zero_grad()
        losses.backward()
        self.optimizer.step()

        stats["vae_loss"] = losses.item()
        return stats
