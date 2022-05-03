# Based on
# https://github.com/Stilwell-Git/Randomized-Return-Decomposition/blob/main/envs/normal_mujoco.py
# with major refactor to clean the code

import gym
import numpy as np


class MuJoCoEpisodicRewardEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        # NOTE: policy won't use action_space.low/high, just set [-1,1]
        # NOTE: in this env, no matter early stopping or timing out, **done = True**
        #       in policy training

    def step(self, action):
        # recover the action
        action = np.clip(action, -1, 1)  # first clip into [-1, 1]
        lb = self.env.action_space.low
        ub = self.env.action_space.high
        action = lb + (action + 1.0) * 0.5 * (ub - lb)
        action = np.clip(action, lb, ub)

        obs, raw_reward, done, info = self.env.step(action)

        # add rewards
        self.rewards += raw_reward

        if done:  # episodic reward
            reward = self.rewards
        else:
            reward = 0.0
        return obs, reward, done, info

    def reset(self):
        self.rewards = 0.0
        return self.env.reset()


if __name__ == "__main__":

    raw_env = gym.make("Pendulum-v1")
    # raw_env = gym.make("Hopper-v2")
    # raw_env = gym.make("HalfCheetah-v2")
    env = MuJoCoEpisodicRewardEnv(raw_env)

    obs = env.reset()
    done = False
    step = 0
    while not done:
        next_obs, rew, done, info = env.step(env.action_space.sample())
        step += 1
        print(step, rew, done, info)
