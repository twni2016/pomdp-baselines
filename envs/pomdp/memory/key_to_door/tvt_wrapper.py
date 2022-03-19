import numpy as np
import gym
from envs.pomdp.memory.key_to_door import env


class KeyToDoor(object):
    def __init__(
        self,
        num_apples=10,
        apple_reward=1.0,
        fix_apple_reward_in_episode=True,
        final_reward=10.0,
        default_reward=0,
    ):
        crop = True
        self.pycolab_env = env.PycolabEnvironment(
            "key_to_door",
            num_apples,
            apple_reward,
            fix_apple_reward_in_episode,
            final_reward,
            crop,
            default_reward,
        )
        self.img_size = (3, 5, 5)
        self.action_space = gym.spaces.MultiDiscrete([2 for i in range(4)])
        self.observation_space = gym.spaces.Box(shape=self.img_size, low=0, high=255)

    def step(self, action):
        action = np.argmax(action)
        obs, r = self.pycolab_env.step(action)
        return np.transpose(obs, (-1, 0, 1)), r, False, None

    def reset(self):
        obs, r = self.pycolab_env.reset()
        return np.transpose(obs, (-1, 0, 1))


if __name__ == "__main__":
    env = KeyToDoor()
    import ipdb

    ipdb.set_trace()
