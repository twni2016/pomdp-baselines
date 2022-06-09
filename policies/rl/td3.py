import torch
from .base import RLAlgorithmBase
from policies.models.actor import DeterministicPolicy
from torchkit.networks import FlattenMlp


class TD3(RLAlgorithmBase):
    name = "td3"
    continuous_action = True
    use_target_actor = True

    def __init__(
        self, exploration_noise=0.1, target_noise=0.2, target_noise_clip=0.5, **kwargs
    ):
        self.exploration_noise = exploration_noise
        self.target_noise = target_noise
        self.target_noise_clip = target_noise_clip

    @staticmethod
    def build_actor(input_size, action_dim, hidden_sizes, **kwargs):
        return DeterministicPolicy(
            obs_dim=input_size,
            action_dim=action_dim,
            hidden_sizes=hidden_sizes,
            **kwargs,
        )

    @staticmethod
    def build_critic(hidden_sizes, input_size=None, obs_dim=None, action_dim=None):
        if obs_dim is not None and action_dim is not None:
            input_size = obs_dim + action_dim
        qf1 = FlattenMlp(
            input_size=input_size, output_size=1, hidden_sizes=hidden_sizes
        )
        qf2 = FlattenMlp(
            input_size=input_size, output_size=1, hidden_sizes=hidden_sizes
        )
        return qf1, qf2

    def select_action(self, actor, observ, deterministic: bool, **kwargs):
        mean = actor(observ)
        if deterministic:
            action_tuple = (mean, mean, None, None)
        else:
            action = (mean + torch.randn_like(mean) * self.exploration_noise).clamp(
                -1, 1
            )  # NOTE
            action_tuple = (action, mean, None, None)
        return action_tuple

    @staticmethod
    def forward_actor(actor, observ):
        new_actions = actor(observ)  # (*, B, dim)
        return new_actions, None

    def _inject_noise(self, actions):
        action_noise = (torch.randn_like(actions) * self.target_noise).clamp(
            -self.target_noise_clip, self.target_noise_clip
        )
        new_actions = (actions + action_noise).clamp(-1, 1)  # NOTE
        return new_actions

    def critic_loss(
        self,
        markov_actor: bool,
        markov_critic: bool,
        actor,
        actor_target,
        critic,
        critic_target,
        observs,
        actions,
        rewards,
        dones,
        gamma,
        next_observs=None,  # used in markov_critic
    ):
        with torch.no_grad():
            # first next_actions from target policy,
            # (T+1, B, dim) including reaction to last obs
            if markov_actor:
                new_actions, _ = self.forward_actor(
                    actor_target, next_observs if markov_critic else observs
                )
            else:
                new_actions, _ = actor_target(
                    prev_actions=actions,
                    rewards=rewards,
                    observs=next_observs if markov_critic else observs,
                )
            new_actions = self._inject_noise(new_actions)

            if markov_critic:  # (B, 1)
                next_q1 = critic_target[0](next_observs, new_actions)
                next_q2 = critic_target[1](next_observs, new_actions)
            else:
                next_q1, next_q2 = critic_target(
                    prev_actions=actions,
                    rewards=rewards,
                    observs=observs,
                    current_actions=new_actions,
                )  # (T+1, B, 1)

            min_next_q_target = torch.min(next_q1, next_q2)

            # q_target: (T, B, 1)
            q_target = rewards + (1.0 - dones) * gamma * min_next_q_target  # next q
            if not markov_critic:
                q_target = q_target[1:]  # (T, B, 1)

        if markov_critic:
            q1_pred = critic[0](observs, actions)
            q2_pred = critic[1](observs, actions)
        else:
            # Q(h(t), a(t)) (T, B, 1)
            q1_pred, q2_pred = critic(
                prev_actions=actions,
                rewards=rewards,
                observs=observs,
                current_actions=actions[1:],
            )  # (T, B, 1)

        return (q1_pred, q2_pred), q_target

    def actor_loss(
        self,
        markov_actor: bool,
        markov_critic: bool,
        actor,
        actor_target,
        critic,
        critic_target,
        observs,
        actions=None,
        rewards=None,
    ):

        if markov_actor:
            new_actions, _ = self.forward_actor(actor, observs)
        else:
            new_actions, _ = actor(
                prev_actions=actions, rewards=rewards, observs=observs
            )  # (T+1, B, A)

        if markov_critic:
            q1 = critic[0](observs, new_actions)
            q2 = critic[1](observs, new_actions)
        else:
            q1, q2 = critic(
                prev_actions=actions,
                rewards=rewards,
                observs=observs,
                current_actions=new_actions,
            )  # (T+1, B, 1)
        min_q_new_actions = torch.min(q1, q2)  # (T+1,B,1)

        policy_loss = -min_q_new_actions
        if not markov_critic:
            policy_loss = policy_loss[:-1]  # (T,B,1) remove the last obs
        return policy_loss, None

    #### Below are used in shared RNN setting
    def forward_actor_in_target(self, actor, actor_target, next_observ):
        new_next_actions, _ = self.forward_actor(actor_target, next_observ)
        return self._inject_noise(new_next_actions), None

    def entropy_bonus(self, log_probs):
        return 0.0
