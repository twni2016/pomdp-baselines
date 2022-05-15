# Ignore annoying warnings from imported envs
import warnings

warnings.filterwarnings("ignore", ".*Box bound precision lowered by casting")  # gym

import gym
import numpy as np

from envs.atari.wrappers import *
from envs.atari.atari_env import Atari


def create_env(
    env_id: str,
    no_terminal: bool = False,
    env_time_limit: int = 27000,
    env_action_repeat: int = 4,
    one_hot_actions: bool = False,
    flatten_img: bool = True,
):

    env = Atari(
        env_id.lower(),
        action_repeat=env_action_repeat,
        flatten_img=flatten_img,
    )

    if hasattr(env.action_space, "n") and one_hot_actions:
        env = OneHotActionWrapper(env)
    if env_time_limit > 0:
        env = TimeLimitWrapper(env, env_time_limit)

    # env = ActionRewardResetWrapper(env, no_terminal)
    # env = CollectWrapper(env)

    return env


if __name__ == "__main__":
    env = create_env(env_id="Pong")  # Pong
    print(env.observation_space.shape, env.action_space.n)

    obs = env.reset()
    done = False
    while not done:
        obs, rew, done, info = env.step(env.action_space.sample())
        print(env.step_, obs.max(), obs.min(), rew, done, info)
