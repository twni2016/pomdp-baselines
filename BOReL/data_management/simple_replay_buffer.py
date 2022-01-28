import numpy as np
import torch
from data_management.replay_buffer import ReplayBuffer
from torchkit import pytorch_utils as ptu


class SimpleReplayBuffer(ReplayBuffer):
    def __init__(
        self,
        max_replay_buffer_size,
        observation_dim,
        action_dim,
        trajectory_len,
        num_reward_arrays=1,
        **kwargs
    ):
        """

        :param max_replay_buffer_size:
        :param observation_dim:
        :param action_dim:
        :param trajectory_len:
        :param num_reward_arrays: if want to save multiple reward terms (say r = r1 + r2 and want to save both)
        :param kwargs: reward_types list [goal_reward, contact_reward, ...]
        """
        self._max_replay_buffer_size = max_replay_buffer_size
        self._observation_dim = observation_dim
        self._action_dim = action_dim
        self.trajectory_len = trajectory_len
        self.multiple_rewards = num_reward_arrays > 1

        self._observations = np.zeros((max_replay_buffer_size, observation_dim))
        # It's a bit memory inefficient to save the observations twice,
        # but it makes the code *much* easier since you no longer have to
        # worry about termination conditions.
        self._next_obs = np.zeros((max_replay_buffer_size, observation_dim))
        self._actions = np.zeros((max_replay_buffer_size, action_dim))
        # Make everything a 2D np array to make it easier for other code to
        # reason about the shape of the data
        if self.multiple_rewards:
            self._rewards = {
                reward_type: np.zeros((max_replay_buffer_size, 1))
                for reward_type in kwargs["reward_types"]
            }
        else:
            self._rewards = np.zeros((max_replay_buffer_size, 1))
        # self._sparse_rewards = np.zeros((max_replay_buffer_size, 1))
        # self._terminals[i] = a terminal was received at time i
        self._terminals = np.zeros((max_replay_buffer_size, 1), dtype="uint8")
        self.clear()

    def add_sample(
        self, observation, action, reward, terminal, next_observation, **kwargs
    ):
        self._observations[self._top] = observation
        self._actions[self._top] = action
        if self.multiple_rewards:
            for reward_type in reward:
                self._rewards[reward_type][self._top] = reward[reward_type]
        else:
            self._rewards[self._top] = reward
        self._terminals[self._top] = terminal
        self._next_obs[self._top] = next_observation
        self._advance()
        # if terminal - start new episode/rollout
        if terminal:
            self.terminate_episode()

    def add_samples(
        self, observations, actions, rewards, terminals, next_observations, **kwargs
    ):
        """inputs are of size (n_samples, dim)"""
        # Assumes no overhead in buffer (there is place for n_samples on top of buffer)
        n_samples = observations.shape[0]
        self._observations[self._top : self._top + n_samples] = observations
        self._actions[self._top : self._top + n_samples] = actions
        self._rewards[self._top : self._top + n_samples] = rewards
        self._terminals[self._top : self._top + n_samples] = terminals
        self._next_obs[self._top : self._top + n_samples] = next_observations
        for _ in range(n_samples):
            self._advance()

    def terminate_episode(self):
        self._episode_starts.append(self._curr_episode_start)
        if len(self._episode_starts) > int(
            self._max_replay_buffer_size / self.trajectory_len
        ):
            del self._episode_starts[0]
        self._curr_episode_start = self._top

    def size(self):
        return self._size

    def clear(self):
        self._top = 0
        self._size = 0
        self._episode_starts = []
        self._curr_episode_start = 0
        self._running_episode_len = 0

    def _advance(self):
        self._top = (self._top + 1) % self._max_replay_buffer_size
        if self._size < self._max_replay_buffer_size:
            self._size += 1
        self._running_episode_len += 1

    def sample_data(self, indices):
        return dict(
            observations=self._observations[indices],
            actions=self._actions[indices],
            rewards=self._rewards[indices]
            if not self.multiple_rewards
            else np.sum(
                [self._rewards[reward_type][indices] for reward_type in self._rewards],
                axis=0,
            ),
            terminals=self._terminals[indices],
            next_observations=self._next_obs[indices],
        )

    def random_batch(self, batch_size):
        """batch of unordered transitions"""
        indices = np.random.randint(0, self._size, batch_size)
        return self.sample_data(indices)

    def can_sample_batch(self, batch_size):
        return self._size >= batch_size

    def random_sequence(self, batch_size):
        """batch of trajectories"""
        # take random trajectories until we have enough
        i = 0
        indices = []
        while len(indices) < batch_size:
            start = np.random.choice(self._episode_starts[:-1])
            pos_idx = self._episode_starts.index(start)
            indices += list(range(start, self._episode_starts[pos_idx + 1]))
            i += 1
        # cut off the last traj if needed to respect batch size
        indices = indices[:batch_size]
        return self.sample_data(indices)

    def random_episodes(self, num_episodes):
        episode_indices = np.random.choice(
            range(self.num_complete_episodes()),
            min(self.num_complete_episodes(), num_episodes),
        )
        indices = []
        for idx in episode_indices:
            start = self._episode_starts[idx]
            end = self._episode_starts[idx] + self.trajectory_len
            indices += list(np.arange(start, end) % self._max_replay_buffer_size)
        return self.sample_data(indices)

    def can_sample_episodes(self, num_episodes=None):
        if num_episodes is None:
            num_episodes = 1
        return self.num_complete_episodes() >= num_episodes

    def num_steps_can_sample(self):
        return self._size

    def get_running_episode(self, zero_pad=True):
        """
        Returns the batch of data from the current episode
        (zero-padded to trajectory length)
        :return:
        """
        length = self._running_episode_len
        ep_start = self._curr_episode_start
        pad_len = self.trajectory_len - length

        if length > 0:
            if pad_len > 0 and zero_pad:
                obs = np.concatenate(
                    (
                        self._observations[ep_start : ep_start + length],
                        np.zeros((pad_len, self._observation_dim)),
                    ),
                    axis=0,
                )
                next_obs = np.concatenate(
                    (
                        self._next_obs[ep_start : ep_start + length],
                        np.zeros((pad_len, self._observation_dim)),
                    ),
                    axis=0,
                )
                actions = np.concatenate(
                    (
                        self._actions[ep_start : ep_start + length],
                        np.zeros((pad_len, self._action_dim)),
                    ),
                    axis=0,
                )
                rewards = np.concatenate(
                    (
                        self._rewards[ep_start : ep_start + length],
                        np.zeros((pad_len, 1)),
                    ),
                    axis=0,
                )
            else:
                obs = self._observations[ep_start : ep_start + length]
                next_obs = self._next_obs[ep_start : ep_start + length]
                actions = self._actions[ep_start : ep_start + length]
                rewards = self._rewards[ep_start : ep_start + length]
        else:
            obs = np.zeros((pad_len, self._observation_dim))
            next_obs = np.zeros((pad_len, self._observation_dim))
            actions = np.zeros((pad_len, self._action_dim))
            rewards = np.zeros((pad_len, 1))

        return obs, next_obs, actions, rewards, length

    def reset_running_episode(self):
        self._running_episode_len = 0

    def num_complete_episodes(self):
        return len(self._episode_starts)


# def unpack_batch(batch):
#     ''' unpack a batch and return individual elements '''
#     obs = batch['observations'][None, ...]
#     actions = batch['actions'][None, ...]
#     rewards = batch['rewards'][None, ...]
#     next_obs = batch['next_observations'][None, ...]
#     terms = batch['terminals'][None, ...]
#     return obs, actions, rewards, next_obs, terms
