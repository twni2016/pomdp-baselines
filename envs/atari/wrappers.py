import gym
import gym.spaces
import numpy as np


class DictWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

    def observation(self, obs_img):
        if len(obs_img.shape) == 1:
            return {"vecobs": obs_img}  # Vector env
        else:
            return {"image": obs_img}  # Image env


class TimeLimitWrapper(gym.Wrapper):
    def __init__(self, env, time_limit):
        super().__init__(env)
        self._max_episode_steps = time_limit

    def step(self, action):
        obs, reward, done, info = self.env.step(action)  # type: ignore
        self.step_ += 1
        if self.step_ >= self._max_episode_steps:
            done = True
            info["TimeLimit.truncated"] = True
        return obs, reward, done, info

    def reset(self):
        self.step_ = 0
        return self.env.reset()  # type: ignore


class ActionRewardResetWrapper(gym.Wrapper):
    def __init__(self, env, no_terminal):
        super().__init__(env)
        self.env = env
        self.no_terminal = no_terminal
        # Handle environments with one-hot or discrete action, but collect always as one-hot
        self.action_size = (
            env.action_space.n
            if hasattr(env.action_space, "n")
            else env.action_space.shape[0]
        )

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if isinstance(action, int):
            action_vec = np.zeros(self.action_size)
            action_vec[action] = 1.0
        else:
            assert isinstance(action, np.ndarray) and action.shape == (
                self.action_size,
            ), "Wrong one-hot action shape"
            action_vec = action
        obs["action"] = action_vec
        obs["reward"] = np.array(reward)
        obs["terminal"] = np.array(
            False if self.no_terminal or info.get("time_limit") else done
        )
        obs["reset"] = np.array(False)
        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        obs["action"] = np.zeros(self.action_size)
        obs["reward"] = np.array(0.0)
        obs["terminal"] = np.array(False)
        obs["reset"] = np.array(True)
        return obs


class CollectWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.episode = []

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.episode.append(obs.copy())
        if done:
            episode = {
                k: np.array([t[k] for t in self.episode]) for k in self.episode[0]
            }
            info["episode"] = episode
        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        self.episode = [obs.copy()]
        return obs


class OneHotActionWrapper(gym.Wrapper):
    """Allow to use one-hot action on a discrete action environment."""

    def __init__(self, env):
        super().__init__(env)
        self.env = env
        # Note: we don't want to change env.action_space to Box(0., 1., (n,)) here,
        # because then e.g. RandomPolicy starts generating continuous actions.

    def step(self, action):
        if not isinstance(action, int):
            action = action.argmax()
        return self.env.step(action)

    def reset(self):
        return self.env.reset()
