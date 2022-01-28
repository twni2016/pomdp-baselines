import numpy as np
from data_management.multi_task_replay_buffer import MultiTaskReplayBuffer


class MultiTaskVAEStorage(MultiTaskReplayBuffer):
    def __init__(
        self,
        max_replay_buffer_size,
        obs_dim,
        action_space,
        tasks,
        trajectory_len,
    ):
        super().__init__(
            max_replay_buffer_size, obs_dim, action_space, tasks, trajectory_len
        )
