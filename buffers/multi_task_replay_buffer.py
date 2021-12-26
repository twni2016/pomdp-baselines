import numpy as np
from utils import helpers as utl
from buffers.simple_replay_buffer import SimpleReplayBuffer
from utils import logger


class MultiTaskReplayBuffer(object):
    def __init__(
        self,
        max_replay_buffer_size,
        obs_dim,
        action_space,
        tasks,
        trajectory_len,
        **kwargs
    ):
        """
        For each training task, maintain a replay buffer
        Used for both Policy and VAE

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
                        **kwargs,
                    ),
                )
                for idx in tasks
            ]
        )
        logger.log(
            "buffer size",
            max_replay_buffer_size,
            "task size",
            len(tasks),
            "traj len",
            trajectory_len,
        )

    def add_sample(
        self,
        task: int,
        observation,
        action,
        reward,
        terminal,
        next_observation,
        **kwargs
    ):
        self.task_buffers[task].add_sample(
            observation, action, reward, terminal, next_observation, **kwargs
        )

    def add_samples(
        self,
        task: int,
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

    def terminate_episode(self, task: int):
        self.task_buffers[task].terminate_episode()

    def random_batch(self, task: int, batch_size):
        return self.task_buffers[task].random_batch(batch_size)

    def can_sample_batch(self, task, batch_size):
        return self.task_buffers[task].can_sample_batch(batch_size)

    def random_episodes(self, task, num_episodes):
        return self.task_buffers[task].random_episodes(num_episodes)

    def can_sample_episodes(self, task, num_episodes=None):
        return self.task_buffers[task].can_sample_episodes(num_episodes)

    def num_steps_can_sample(self, task):
        return self.task_buffers[task].num_steps_can_sample()

    def clear_buffer(self, task):
        self.task_buffers[task].clear()

    def num_complete_episodes(self, task):
        return self.task_buffers[task].num_complete_episodes()
