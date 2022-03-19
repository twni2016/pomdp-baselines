import numpy as np


class SeqReplayBuffer:
    buffer_type = "seq_vanilla"

    def __init__(
        self,
        max_replay_buffer_size,
        observation_dim,
        action_dim,
        sampled_seq_len: int,
        sample_weight_baseline: float,
        **kwargs
    ):
        """
        this buffer is used for sequence/trajectory/episode:
                it stored the whole sequence
                into the buffer (not transition), and can sample (sub)sequences
                that has 3D shape (sampled_seq_len, batch_size, dim)
                based on some rules below.
        it still uses 2D size as normal (max_replay_buffer_size, dim)
                but tracks the sequences

        NOTE: it save observations twice, so it is vanilla version of seq replay buffer,
            sufficient to vector-based observations, as RAM is not the bottleneck
        """
        self._max_replay_buffer_size = max_replay_buffer_size
        self._observation_dim = observation_dim
        self._action_dim = action_dim

        self._observations = np.zeros(
            (max_replay_buffer_size, observation_dim), dtype=np.float32
        )
        self._next_observations = np.zeros(
            (max_replay_buffer_size, observation_dim), dtype=np.float32
        )

        self._actions = np.zeros((max_replay_buffer_size, action_dim), dtype=np.float32)
        self._rewards = np.zeros((max_replay_buffer_size, 1), dtype=np.float32)

        # terminals are "done" signals, useful for policy training
        # for each trajectory, it has single 1 like 0000001000 for reaching goal or early stopping
        # 	or simply 0s for timing out.
        # NOTE: so we cannot use terminals to determine the trajectory boundary!
        self._terminals = np.zeros((max_replay_buffer_size, 1), dtype=np.uint8)

        # NOTE: valid_starts are (internal) masks which is 1 (or postive number as weight)
        # 	if we can SAMPLE the (sub)sequence FROM this index else 0.
        # For each trajectory, the first index has valid_start as 1 (or postive number),
        # 	the LAST sampled_seq_len indices are 0s, and the middle ones are 1s (or positive numbers)
        # 	That is to say, if its length <= sampled_seq_len, then the valid_starts looks like 100000000
        # 	else looks like 11111000000 (have sampled_seq_len - 1 zeros)
        # See _compute_valid_starts function for details
        self._valid_starts = np.zeros((max_replay_buffer_size), dtype=np.float32)

        assert sampled_seq_len >= 2
        assert sample_weight_baseline >= 0.0
        self._sampled_seq_len = sampled_seq_len
        self._sample_weight_baseline = sample_weight_baseline

        self.clear()

        RAM = 0.0
        for name, var in vars(self).items():
            if isinstance(var, np.ndarray):
                RAM += var.nbytes
        print(f"buffer RAM usage: {RAM / 1024 ** 3 :.2f} GB")

    def size(self):
        return self._size

    def clear(self):
        self._top = 0  # trajectory level (first dim in 3D buffer)
        self._size = 0  # trajectory level (first dim in 3D buffer)

    def add_episode(self, observations, actions, rewards, terminals, next_observations):
        """
        NOTE: must add one whole episode/sequence/trajectory,
                        not some partial transitions
        the length of different episode can vary, but must be greater than 2
                so that the end of valid_starts is 0.

        all the inputs have 2D shape of (L, dim)
        """
        assert (
            observations.shape[0]
            == actions.shape[0]
            == rewards.shape[0]
            == terminals.shape[0]
            == next_observations.shape[0]
            >= 2
        )

        seq_len = observations.shape[0]  # L
        indices = list(
            np.arange(self._top, self._top + seq_len) % self._max_replay_buffer_size
        )

        self._observations[indices] = observations
        self._actions[indices] = actions
        self._rewards[indices] = rewards
        self._terminals[indices] = terminals
        self._next_observations[indices] = next_observations

        self._valid_starts[indices] = self._compute_valid_starts(seq_len)

        self._top = (self._top + seq_len) % self._max_replay_buffer_size
        self._size = min(self._size + seq_len, self._max_replay_buffer_size)

    def _compute_valid_starts(self, seq_len):
        valid_starts = np.ones((seq_len), dtype=float)

        num_valid_starts = float(max(1.0, seq_len - self._sampled_seq_len + 1.0))

        # compute weights: baseline + num_of_can_sampled_indices
        total_weights = self._sample_weight_baseline + num_valid_starts

        # now each item has even weight, if baseline is 0.0, then it's 1s
        valid_starts *= total_weights / num_valid_starts

        # set the num_valid_starts: indices are zeros
        valid_starts[int(num_valid_starts) :] = 0.0

        return valid_starts

    def random_episodes(self, batch_size):
        """
        return each item has 3D shape (sampled_seq_len, batch_size, dim)
        """
        sampled_episode_starts = self._sample_indices(batch_size)  # (B,)

        # get sequential indices
        indices = []
        for start in sampled_episode_starts:  # small loop
            end = start + self._sampled_seq_len  # continuous + T
            indices += list(np.arange(start, end) % self._max_replay_buffer_size)

        # extract data
        batch = self._sample_data(indices)
        # each item has 2D shape (num_episodes * sampled_seq_len, dim)

        # generate masks (B, T)
        masks = self._generate_masks(indices, batch_size)
        batch["mask"] = masks

        for k in batch.keys():
            batch[k] = (
                batch[k]
                .reshape(batch_size, self._sampled_seq_len, -1)
                .transpose(1, 0, 2)
            )

        return batch

    def _sample_indices(self, batch_size):
        # self._top points at the start of a new sequence
        # self._top - 1 is the end of the recently stored sequence
        valid_starts_indices = np.where(self._valid_starts > 0.0)[0]

        sample_weights = np.copy(self._valid_starts[valid_starts_indices])
        # normalize to probability distribution
        sample_weights /= sample_weights.sum()

        return np.random.choice(valid_starts_indices, size=batch_size, p=sample_weights)

    def _sample_data(self, indices):
        return dict(
            obs=self._observations[indices],
            act=self._actions[indices],
            rew=self._rewards[indices],
            term=self._terminals[indices],
            obs2=self._next_observations[indices],
        )

    def _generate_masks(self, indices, batch_size):
        """
        input: sampled_indices list of len B*T
        output: masks (B, T)
        """

        # get valid_starts of sampled sequences (B, T)
        # each row starts with a postive number, like 11111000, or 10000011, or 1s
        sampled_seq_valids = np.copy(self._valid_starts[indices]).reshape(
            batch_size, self._sampled_seq_len
        )
        sampled_seq_valids[sampled_seq_valids > 0.0] = 1.0  # binarize

        # build masks
        masks = np.ones_like(sampled_seq_valids, dtype=float)  # (B, T), default is 1

        # we want to find the boundary (ending) of sampled sequences
        # 	i.e. the FIRST 1 (positive number) after 0
        # 	this is important for varying length episodes
        # the boundary (ending) appears at the FIRST -1 in diff
        diff = sampled_seq_valids[:, :-1] - sampled_seq_valids[:, 1:]  # (B, T-1)
        # add 1s into the first column
        diff = np.concatenate([np.ones((batch_size, 1)), diff], axis=1)  # (B, T)

        # special case: the sampled sequence cannot cross self._top
        indices_array = np.array(indices).reshape(
            batch_size, self._sampled_seq_len
        )  # (B,T)
        # set the top as -1.0 as invalid starts
        diff[indices_array == self._top] = -1.0

        # now the start of next episode appears at the FIRST -1 in diff
        invalid_starts_b, invalid_starts_t = np.where(
            diff == -1.0
        )  # (1D array in batch dim, 1D array in seq dim)
        invalid_indices_b = []
        invalid_indices_t = []
        last_batch_index = -1

        for batch_index, start_index in zip(invalid_starts_b, invalid_starts_t):
            if batch_index == last_batch_index:
                # for same batch_idx, we only care the first appearance of -1
                continue
            last_batch_index = batch_index

            invalid_indices = list(
                np.arange(start_index, self._sampled_seq_len)
            )  # to the end
            # extend to the list
            invalid_indices_b += [batch_index] * len(invalid_indices)
            invalid_indices_t += invalid_indices

        # set invalids in the masks
        masks[invalid_indices_b, invalid_indices_t] = 0.0

        return masks


if __name__ == "__main__":
    seq_len = 10
    baseline = 1.0
    buffer = SeqReplayBuffer(100, 2, 2, seq_len, baseline)
    for l in range(seq_len - 2, seq_len + 2):
        print(l)
        assert buffer._compute_valid_starts(l)[0] > 0.0
        print(buffer._compute_valid_starts(l))
    for e in range(10):
        data = np.zeros((10, 2))
        buffer.add_episode(
            np.zeros((11, 2)),
            np.zeros((11, 2)),
            np.zeros((11, 1)),
            np.zeros((11, 1)),
            np.zeros((11, 2)),
        )
    print(buffer._size, buffer._valid_starts)
