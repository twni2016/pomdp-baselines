import gym
from gym.wrappers import Monitor


class AdaptiveVideoMonitor(Monitor):
    """Used for recording, modified reset() to work for Adaptive"""

    def reset(self, reset_params, **kwargs):
        """Only function that calls reset"""
        self._before_reset()
        observation = self.env.reset(reset_params, **kwargs)
        self._after_reset(observation)

        return observation
