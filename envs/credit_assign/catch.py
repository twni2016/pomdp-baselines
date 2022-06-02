"""
Perceptual catch. Original author: Michel Ma.

Delay = (Number of runs - 1) * 7 + 6
Time_steps = (Number of runs - 1) * 7 + 6
"""


import numpy as np
import gym


class DelayedCatch(gym.Env):
    def __init__(
        self, delay, grid_size=7, flatten_img=True, delayed=True, one_hot_actions=False
    ):
        super().__init__()
        self.grid_size = grid_size
        self.delay = delay
        self.num_catches = (delay + 1) // 7
        self.action_space = gym.spaces.Discrete(3)

        self.image_space = gym.spaces.MultiDiscrete(
            [[[2 for i in range(grid_size)] for i in range(grid_size)]]
        )
        self.flatten_img = flatten_img
        if flatten_img:
            self.observation_space = gym.spaces.MultiDiscrete(
                [2 for i in range(grid_size * grid_size)]
            )
        else:
            self.observation_space = self.image_space

        self.delayed = delayed
        self.one_hot_actions = one_hot_actions

    def _update_state(self, action):
        """
        Input: action and states
        Ouput: new states and reward
        """
        state = self.state
        if self.one_hot_actions:
            action = np.argmax(action)

        if action == 0:  # left
            action = -1
        elif action == 1:  # stay
            action = 0
        elif action == 2:
            action = 1  # right
        else:
            raise ValueError("not valid action")
        f0, f1, basket = state[0]
        new_basket = min(max(0, basket + action), self.grid_size - 1)
        f0 += 1
        out = np.asarray([f0, f1, new_basket])
        out = out[np.newaxis]

        assert len(out.shape) == 2
        self.state = out

    def _draw_state(self):
        im_size = (self.grid_size,) * 2
        state = self.state[0]
        canvas = np.zeros(im_size)
        canvas[state[0], state[1]] = -1  # draw fruit
        canvas[-1, int(state[2])] = 1  # draw basket
        return canvas

    def _get_reward(self):
        fruit_row, fruit_col, basket = self.state[0]
        if fruit_row == self.grid_size - 1:
            if fruit_col == basket:
                return 1
            else:
                return 0
        else:
            return 0

    def _is_over(self):
        if self.state[0, 0] == self.grid_size - 1:
            return True
        else:
            return False

    def observe(self):
        canvas = self._draw_state()
        if self.flatten_img:
            return canvas.reshape(-1)
        else:
            return np.expand_dims(canvas, axis=0)

    def step(self, action):
        self.time_step += 1
        info = {}

        if self._is_over():
            obs = self.soft_reset()

            reward = self._get_reward()
            info["reward"] = reward
            self.accumulated_reward += reward
            return obs, 0, False, info

        self._update_state(action)

        reward = self._get_reward()
        info["reward"] = reward
        self.accumulated_reward += reward

        if not self.delayed:
            return self.observe(), reward, False, info

        if self.time_step >= self.delay:
            return self.observe(), self.accumulated_reward, True, info
        else:
            return self.observe(), 0, False, info

    def reset(self):
        # Pre-generate the initial states
        self.catch_count = 0
        self.ns = np.random.randint(0, self.grid_size - 1, size=self.num_catches)
        self.ms = np.random.randint(1, self.grid_size - 2, size=self.num_catches)
        obs = self.soft_reset()
        self.accumulated_reward = 0
        self.time_step = 0
        return obs

    def soft_reset(self):
        n = self.ns[self.catch_count]
        m = self.ms[self.catch_count]
        self.state = np.asarray([0, n, m])[np.newaxis]
        self.catch_count += 1
        return self.observe()


if __name__ == "__main__":
    runs = 5
    delay = (runs - 1) * 7 + 6
    env = DelayedCatch(delay, flatten_img=True, one_hot_actions=False)

    obs = env.reset()
    done = False
    while not done:
        obs, rew, done, info = env.step(env.action_space.sample())
        print(env.time_step, info)
