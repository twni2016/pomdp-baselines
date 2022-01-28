import numpy as np

from .ant import AntEnv

# from gym.envs.mujoco.ant import AntEnv


class MultitaskAntEnv(AntEnv):
    def __init__(self, task={}, n_tasks=2, **kwargs):
        self._task = task
        self.n_tasks = n_tasks
        if n_tasks is None:
            self._goal = self._sample_raw_task()["goal"]
        else:
            self.tasks = self.sample_tasks(n_tasks)
            self._goal = self.tasks[0]["goal"]
        super(MultitaskAntEnv, self).__init__()

    def get_current_task(self):
        # for multi-task MDP
        return np.array([self._goal])

    def get_all_task_idx(self):
        return range(len(self.tasks))

    def reset_task(self, task_info):
        if self.n_tasks is None:  # unlimited tasks
            assert task_info is None
            self._task = self._sample_raw_task()  # sample here
        else:  # limited tasks
            self._task = self.tasks[task_info]  # as idx
        self._goal = self._task[
            "goal"
        ]  # assume parameterization of task by single vector
        self.reset()
