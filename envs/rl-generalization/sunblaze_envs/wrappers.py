import collections

import gym
from gym.envs.registration import load
import numpy as np


def ActionDelayWrapper(delay_range_start, delay_range_end):
    """Create an action delay wrapper.

    :param delay_range_start: Minimum delay
    :param delay_range_end: Maximum delay
    """

    class ActionDelayWrapper(gym.Wrapper):
        def _step(self, action):
            self._action_buffer.append(action)
            action = self._action_buffer.popleft()
            return self.env.step(action)

        def _reset(self):
            self._action_delay = np.random.randint(delay_range_start, delay_range_end)
            self._action_buffer = collections.deque(
                [0 for _ in range(self._action_delay)]
            )
            return self.env.reset()

    return ActionDelayWrapper


def wrap_environment(wrapped_class, wrappers=None, **kwargs):
    """Helper for wrapping environment classes."""
    if wrappers is None:
        wrappers = []

    env_class = load(wrapped_class)
    env = env_class(**kwargs)
    for wrapper, wrapper_kwargs in wrappers:
        wrapper_class = load(wrapper)
        wrapper = wrapper_class(**wrapper_kwargs)
        env = wrapper(env)

    return env
