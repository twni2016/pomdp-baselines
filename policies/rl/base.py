from typing import Any, Tuple
from policies.models.actor import MarkovPolicyBase


class RLAlgorithmBase:
    name = "rl"
    continuous_action = True
    use_target_actor = True

    def __init__(self, **kwargs):
        pass

    @staticmethod
    def build_actor(input_size, action_dim, hidden_sizes) -> MarkovPolicyBase:
        raise NotImplementedError

    @staticmethod
    def build_critic(input_size, hidden_sizes, **kwargs) -> Tuple[Any, Any]:
        """
        return two critics
        """
        raise NotImplementedError

    def select_action(
        self, actor, observ, deterministic: bool, **kwargs
    ) -> Tuple[Any, Any, Any, Any]:
        """
        actor: defined by build_actor
        observ: (B, dim), could be history embedding
        return (action, mean*, log_std*, log_prob*) * if exists
        """
        raise NotImplementedError

    @staticmethod
    def forward_actor(actor, observ) -> Tuple[Any, Any]:
        """
        actor: defined by build_actor
        observ: (B, dim), could be history embedding
        return (action, log_prob*)
        """
        raise NotImplementedError

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
        next_observs,
    ) -> Tuple[Tuple[Any, Any], Any]:
        """
        return (q1_pred, q2_pred), q_target
        """
        raise NotImplementedError

    def actor_loss(
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
    ) -> Tuple[Any, Any]:
        """
        return policy_loss, log_probs*
        """
        raise NotImplementedError

    def update_others(self, **kwargs):
        pass
