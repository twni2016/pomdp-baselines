import numpy as np
from .replay_buffer import ReplayBuffer


class SimpleReplayBuffer(ReplayBuffer):
    buffer_type = "markov"

    def __init__(
        self,
        max_replay_buffer_size,
        observation_dim,
        action_dim,
        max_trajectory_len: int,
        add_timeout: bool = False,
        **kwargs
    ):
        """
        :param max_replay_buffer_size:
        :param observation_dim:
        :param action_dim:
        :param trajectory_len:
        :param kwargs: reward_types list [goal_reward, contact_reward, ...]

        NOTE: difference from terminal and timeout:
        - for model-based methods (e.g. varibad): add_timeout = False, we only have terminals
                - for VAE, terminal is whether time out, so we use terminal to sample episodes
                - for policy, terminal is whether reach goal, so we use terminal to bootstrap or not in sampled transitions
        - for model-free methods (e.g. RNN policy): add_timeout = True, we have both terminals and dones
                - we only have policy buffer
                - terminal is whether reach goal for bootstrap
                - timeout is whether time out for sampling episodes
        """
        self._max_replay_buffer_size = max_replay_buffer_size
        self._observation_dim = observation_dim
        self._action_dim = action_dim
        self.trajectory_len = max_trajectory_len

        self._observations = np.zeros((max_replay_buffer_size, observation_dim))
        # It's a bit memory inefficient to save the observations twice,
        # but it makes the code *much* easier since you no longer have to
        # worry about termination conditions.
        self._next_obs = np.zeros((max_replay_buffer_size, observation_dim))
        self._actions = np.zeros((max_replay_buffer_size, action_dim))
        # Make everything a 2D np array to make it easier for other code to
        # reason about the shape of the data
        self._rewards = np.zeros((max_replay_buffer_size, 1))
        self._terminals = np.zeros((max_replay_buffer_size, 1), dtype="uint8")

        self.add_timeout = add_timeout
        if add_timeout:
            self._timeouts = np.zeros((max_replay_buffer_size, 1), dtype="uint8")
        self.clear()

    def add_sample(
        self,
        observation,
        action,
        reward,
        terminal,
        next_observation,
        timeout=None,
        **kwargs
    ):
        self._observations[self._top] = observation
        self._actions[self._top] = action
        self._rewards[self._top] = reward
        self._terminals[self._top] = terminal
        self._next_obs[self._top] = next_observation

        if self.add_timeout:
            self._timeouts[self._top] = timeout

        self._advance()

        # if terminal - start new episode/rollout
        if (self.add_timeout and timeout) or (not self.add_timeout and terminal):
            self.terminate_episode()

    def terminate_episode(self):
        # NOTE: one can also use self.terminal == True to find the starts
        # but this requires more complicated condition checking
        # here we assume fixed episode length
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
        # ensure that each element is the start index of a entire traj
        self._episode_starts = []
        self._curr_episode_start = 0

    def _advance(self, step=1):
        self._top = (self._top + step) % self._max_replay_buffer_size
        self._size = min(self._size + step, self._max_replay_buffer_size)

    def sample_data(self, indices):
        return dict(
            obs=self._observations[indices],
            act=self._actions[indices],
            rew=self._rewards[indices],
            term=self._terminals[indices],
            obs2=self._next_obs[indices],
        )

    def random_batch(self, batch_size):
        """batch of unordered transitions"""
        # assert self.can_sample_batch(batch_size)
        indices = np.random.randint(0, self._size, batch_size)
        return self.sample_data(indices)

    def can_sample_batch(self, batch_size):
        return self._size >= batch_size

    def random_episodes(self, num_episodes, sub_traj_len=-1, replace=False):
        """NOTE: return each item has 3D shape (sub_traj_len, num_episodes, dim)"""
        # assert self.can_sample_episodes(num_episodes) # to make sure no replacement
        episode_indices = np.random.choice(
            range(self.num_complete_episodes()), num_episodes, replace=replace
        )
        assert sub_traj_len <= self.trajectory_len
        if sub_traj_len == -1:
            sub_traj_len = self.trajectory_len
        assert sub_traj_len >= 1

        sub_traj_starts = np.random.randint(
            0, self.trajectory_len - sub_traj_len + 1, num_episodes
        )  # for whole traj, just zeros

        indices = []
        for idx, sub_traj_start in zip(episode_indices, sub_traj_starts):  # small loop
            start = self._episode_starts[idx] + sub_traj_start  # + 0
            end = start + sub_traj_len  # + T
            indices += list(np.arange(start, end) % self._max_replay_buffer_size)

        raw_batch = self.sample_data(indices)
        # each item has 2D shape (num_episodes * sub_traj_len, dim)
        batch = dict()
        for k in raw_batch.keys():
            batch[k] = (
                raw_batch[k].reshape(num_episodes, sub_traj_len, -1).transpose(1, 0, 2)
            )

        return batch

    def can_sample_episodes(self, num_episodes=None):
        if num_episodes is None:
            num_episodes = 1
        return self.num_complete_episodes() >= num_episodes

    def num_steps_can_sample(self):
        return self._size

    def num_complete_episodes(self):
        return len(self._episode_starts)
