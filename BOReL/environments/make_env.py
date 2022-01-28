"""
Based on https://github.com/ikostrikov/pytorch-a2c-ppo-acktr
"""
import os

import gym
import numpy as np
import torch
from gym.spaces.box import Box

from environments.wrappers import VariBadWrapper


def make_env(env_id, episodes_per_task, seed=None, **kwargs):
    env = gym.make(env_id, **kwargs)
    if seed is not None:
        env.seed(seed)
    env = VariBadWrapper(
        env=env,
        episodes_per_task=episodes_per_task,
    )
    return env
