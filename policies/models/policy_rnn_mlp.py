import torch
from copy import deepcopy
import torch.nn as nn
from torch.optim import Adam
from utils import helpers as utl
from policies.models import *
from policies.models.policy_rnn import ModelFreeOffPolicy_Separate_RNN
from utils import logger


class ModelFreeOffPolicy_RNN_MLP(ModelFreeOffPolicy_Separate_RNN):
    """
    Markov Actor and Recurrent Critic
    It may be more effective on some special cases of POMDPs,
        where the reward is history-dependent, but Markov actor
        is sufficient to solve the task.
    """

    ARCH = "memory-markov"
    Markov_Actor = True
    Markov_Critic = False

    def __init__(
        self,
        obs_dim,
        action_dim,
        encoder,
        algo_name,
        action_embedding_size,
        observ_embedding_size,
        reward_embedding_size,
        rnn_hidden_size,
        dqn_layers,
        policy_layers,
        rnn_num_layers=1,
        lr=3e-4,
        gamma=0.99,
        tau=5e-3,
        # pixel obs
        image_encoder_fn=lambda: None,
        **kwargs
    ):
        super().__init__(
            obs_dim,
            action_dim,
            encoder,
            algo_name,
            action_embedding_size,
            observ_embedding_size,
            reward_embedding_size,
            rnn_hidden_size,
            dqn_layers,
            policy_layers,
            rnn_num_layers,
            lr,
            gamma,
            tau,
            image_encoder_fn,
            **kwargs,
        )

        # Markov Actor
        self.actor = self.algo.build_actor(
            input_size=obs_dim,
            action_dim=action_dim,
            hidden_sizes=policy_layers,
            image_encoder=image_encoder_fn(),  # separate weight
        )
        self.actor_optimizer = Adam(self.actor.parameters(), lr=lr)
        # target network
        self.actor_target = deepcopy(self.actor)

    @torch.no_grad()
    def act(
        self,
        obs,
        deterministic=False,
        return_log_prob=False,
    ):
        return self.algo.select_action(
            actor=self.actor,
            observ=obs,
            deterministic=deterministic,
            return_log_prob=return_log_prob,
        )

    def report_grad_norm(self):
        # may add qf1, policy, etc.
        return {
            "q_grad_norm": utl.get_grad_norm(self.critic),
            "q_rnn_grad_norm": utl.get_grad_norm(self.critic.rnn),
            "pi_grad_norm": utl.get_grad_norm(self.actor),
        }
