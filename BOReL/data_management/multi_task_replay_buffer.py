import numpy as np
from utils import helpers as utl
from data_management.simple_replay_buffer import SimpleReplayBuffer
from gym.spaces import Discrete


class MultiTaskReplayBuffer(object):
    def __init__(
        self,
        max_replay_buffer_size,
        obs_dim,
        action_space,
        tasks,
        trajectory_len,
        num_reward_arrays=1,
        **kwargs
    ):
        """
        :param max_replay_buffer_size:
        :param obs_dim
        :param action_space
        :param tasks: for multi-task setting
        """
        self._obs_dim = obs_dim
        self._action_space = action_space
        self.trajectory_len = trajectory_len
        self.task_buffers = dict(
            [
                (
                    idx,
                    SimpleReplayBuffer(
                        max_replay_buffer_size=max_replay_buffer_size,
                        observation_dim=self._obs_dim,
                        action_dim=utl.get_dim(self._action_space),
                        trajectory_len=trajectory_len,
                        num_reward_arrays=num_reward_arrays,
                        **kwargs,
                    ),
                )
                for idx in tasks
            ]
        )

    def add_sample(
        self, task, observation, action, reward, terminal, next_observation, **kwargs
    ):

        if isinstance(self._action_space, Discrete):
            action = np.eye(self._action_space.n)[action]
        self.task_buffers[task].add_sample(
            observation, action, reward, terminal, next_observation, **kwargs
        )

    def add_samples(
        self,
        task,
        observations,
        actions,
        rewards,
        terminals,
        next_observations,
        **kwargs
    ):
        self.task_buffers[task].add_samples(
            observations, actions, rewards, terminals, next_observations, **kwargs
        )

    def terminate_episode(self, task):
        self.task_buffers[task].terminate_episode()

    def random_batch(self, task, batch_size, sequence=False):
        if sequence:
            batch = self.task_buffers[task].random_sequence(batch_size)
        else:
            batch = self.task_buffers[task].random_batch(batch_size)
        return batch

    def can_sample_batch(self, task, batch_size):
        return self.task_buffers[task].can_sample_batch(batch_size)

    def random_episodes(self, task, num_episodes):
        return self.task_buffers[task].random_episodes(num_episodes)

    def can_sample_episodes(self, task, num_episodes=None):
        return self.task_buffers[task].can_sample_episodes(num_episodes)

    def num_steps_can_sample(self, task):
        return self.task_buffers[task].num_steps_can_sample()

    def add_path(self, task, path):
        self.task_buffers[task].add_path(path)

    def add_paths(self, task, paths):
        for path in paths:
            self.task_buffers[task].add_path(path)

    def clear_buffer(self, task):
        self.task_buffers[task].clear()

    def get_running_episode(self, task):
        return self.task_buffers[task].get_running_episode()

    def reset_running_episode(self, task):
        self.task_buffers[task].reset_running_episode()

    def num_complete_episodes(self, task):
        return self.task_buffers[task].num_complete_episodes()
