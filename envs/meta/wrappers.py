from gym.envs.registration import load
import gym
import numpy as np
from gym import spaces


def mujoco_wrapper(entry_point, **kwargs):
    # Load the environment from its entry point
    env_cls = load(entry_point)
    env = env_cls(**kwargs)
    return env


class VariBadWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        episodes_per_task: int,
        oracle: bool = False,  # default no
    ):
        """
        Wrapper, creates a multi-episode (BA)MDP around a one-episode MDP. Automatically deals with
        - horizons H in the MDP vs horizons H+ in the BAMDP,
        - resetting the tasks
        - normalized actions in case of continuous action space
        - adding the timestep / done info to the state (might be needed to make states markov)
        """

        super().__init__(env)

        # if continuous actions, make sure in [-1, 1]
        # NOTE: policy won't use action_space.low/high, just set [-1,1]
        # this is a bad practice...
        if isinstance(self.env.action_space, gym.spaces.Box):
            self._normalize_actions = True
        else:
            self._normalize_actions = False

        self.oracle = oracle
        if self.oracle == True:
            print("WARNING: YOU ARE RUNNING MDP, NOT POMDP!\n")
            tmp_task = self.env.get_current_task()
            self.observation_space = spaces.Box(
                low=np.array(
                    [*self.observation_space.low, *([0] * len(tmp_task))]
                ),  # shape will be deduced from this
                high=np.array([*self.observation_space.high, *([1] * len(tmp_task))]),
                dtype=np.float32,
            )

        if episodes_per_task > 1:
            self.add_done_info = True
        else:
            self.add_done_info = False
        if self.add_done_info:
            self.observation_space = spaces.Box(
                low=np.array(
                    [*self.observation_space.low, 0]
                ),  # shape will be deduced from this
                high=np.array([*self.observation_space.high, 1]),
                dtype=np.float32,
            )

        # calculate horizon length H^+
        self.episodes_per_task = episodes_per_task
        # counts the number of episodes
        self.episode_count = 0

        # count timesteps in BAMDP
        self.step_count_bamdp = 0.0
        # the horizon in the BAMDP is the one in the MDP times the number of episodes per task,
        # and if we train a policy that maximises the return over all episodes
        # we add transitions to the reset start in-between episodes
        try:
            self.horizon_bamdp = self.episodes_per_task * self.env._max_episode_steps
        except AttributeError:
            self.horizon_bamdp = (
                self.episodes_per_task * self.env.unwrapped._max_episode_steps
            )

        # this tells us if we have reached the horizon in the underlying MDP
        self.done_mdp = True

    def _get_obs(self, state):
        if self.oracle:
            tmp_task = self.env.get_current_task().copy()
            state = np.concatenate([state, tmp_task])
        if self.add_done_info:
            state = np.concatenate((state, [float(self.done_mdp)]))
        return state

    def reset(self, task=None):

        # reset task -- this sets goal and state -- sets self.env._goal and self.env._state
        self.env.reset_task(task)

        self.episode_count = 0
        self.step_count_bamdp = 0

        # normal reset
        try:
            state = self.env.reset()
        except AttributeError:
            state = self.env.unwrapped.reset()

        self.done_mdp = False

        return self._get_obs(state)

    def wrap_state_with_done(self, state):
        # for some custom evaluation like semicircle
        if self.add_done_info:
            state = np.concatenate((state, [float(self.done_mdp)]))
        return state

    def reset_mdp(self):
        state = self.env.reset()
        self.done_mdp = False

        return self._get_obs(state)

    def step(self, action):

        if self._normalize_actions:  # from [-1, 1] to [lb, ub]
            action = np.clip(action, -1, 1)  # first clip into [-1, 1]
            lb = self.env.action_space.low
            ub = self.env.action_space.high
            action = lb + (action + 1.0) * 0.5 * (ub - lb)
            action = np.clip(action, lb, ub)

        # do normal environment step in MDP
        state, reward, self.done_mdp, info = self.env.step(action)

        info["done_mdp"] = self.done_mdp
        state = self._get_obs(state)

        self.step_count_bamdp += 1
        # if we want to maximise performance over multiple episodes,
        # only say "done" when we collected enough episodes in this task
        done_bamdp = False
        if self.done_mdp:
            self.episode_count += 1
            if self.episode_count == self.episodes_per_task:
                done_bamdp = True

        if self.done_mdp and not done_bamdp:
            info["start_state"] = self.reset_mdp()

        return state, reward, done_bamdp, info


class TimeLimitMask(gym.Wrapper):
    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        if done and self.env._max_episode_steps == self.env._elapsed_steps:
            info["bad_transition"] = True

        return obs, rew, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
