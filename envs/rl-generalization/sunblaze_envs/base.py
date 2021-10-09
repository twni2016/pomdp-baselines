from abc import ABC, abstractmethod

import gym


class BaseGymEnvironment(gym.Env):
    """Base class for all Gym environments."""

    @property
    def parameters(self):
        """Return environment parameters."""
        return {
            "id": self.spec.id,
        }


class EnvBinarySuccessMixin(ABC):
    """Adds binary success metric to environment."""

    @abstractmethod
    def is_success(self):
        """Returns True is current state indicates success, False otherwise"""
        pass
