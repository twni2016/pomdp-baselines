from environments.mujoco.rand_param_envs.gym.core import Env
from environments.mujoco.rand_param_envs.gym.envs.mujoco import MujocoEnv
import numpy as np
import random


class MetaEnv(Env):
    def step(self, *args, **kwargs):
        return self._step(*args, **kwargs)

    def sample_tasks(self, n_tasks):
        """
        Samples task of the meta-environment

        Args:
            n_tasks (int) : number of different meta-tasks needed

        Returns:
            tasks (list) : an (n_tasks) length list of tasks
        """
        raise NotImplementedError

    def set_task(self, task):
        """
        Sets the specified task to the current environment

        Args:
            task: task of the meta-learning environment
        """
        raise NotImplementedError

    def get_task(self):
        """
        Gets the task that the agent is performing in the current environment

        Returns:
            task: task of the meta-learning environment
        """
        raise NotImplementedError

    def reset_task(self, task):
        if task is None:
            task = self.sample_tasks(1)[0]
        self.set_task(task)
        self.reset()

    def log_diagnostics(self, paths, prefix):
        """
        Logs env-specific diagnostic information

        Args:
            paths (list) : list of all paths collected with this env during this iteration
            prefix (str) : prefix for logger
        """
        pass


class RandomEnv(MetaEnv, MujocoEnv):
    """
    This class provides functionality for randomizing the physical parameters of a mujoco model
    The following parameters are changed:
        - body_mass
        - body_inertia
        - damping coeff at the joints
    """

    RAND_PARAMS = ["body_mass", "dof_damping", "body_inertia", "geom_friction"]
    RAND_PARAMS_EXTENDED = RAND_PARAMS + ["geom_size"]

    def __init__(
        self, log_scale_limit, file_name, *args, rand_params=RAND_PARAMS, **kwargs
    ):
        self.log_scale_limit = log_scale_limit
        self.rand_params = rand_params
        MujocoEnv.__init__(self, file_name, 4)
        assert set(rand_params) <= set(
            self.RAND_PARAMS_EXTENDED
        ), "rand_params must be a subset of " + str(self.RAND_PARAMS_EXTENDED)
        self.save_parameters()

    def sample_tasks(self, n_tasks):
        """
        Generates randomized parameter sets for the mujoco env

        Args:
            n_tasks (int) : number of different meta-tasks needed

        Returns:
            tasks (list) : an (n_tasks) length list of tasks
        """
        param_sets = []

        for _ in range(n_tasks):
            # body mass -> one multiplier for all body parts

            new_params = {}

            if "body_mass" in self.rand_params:
                rand_params = [
                    random.uniform(-self.log_scale_limit, self.log_scale_limit)
                    for _ in range(np.prod(self.model.body_mass.shape))
                ]
                body_mass_multiplyers = np.array(1.5) ** np.array(rand_params).reshape(
                    self.model.body_mass.shape
                )
                new_params["body_mass"] = (
                    self.init_params["body_mass"] * body_mass_multiplyers
                )

            # body_inertia
            if "body_inertia" in self.rand_params:
                rand_params = [
                    random.uniform(-self.log_scale_limit, self.log_scale_limit)
                    for _ in range(np.prod(self.model.body_inertia.shape))
                ]
                body_inertia_multiplyers = np.array(1.5) ** np.array(
                    rand_params
                ).reshape(self.model.body_inertia.shape)
                new_params["body_inertia"] = (
                    body_inertia_multiplyers * self.init_params["body_inertia"]
                )

            # damping -> different multiplier for different dofs/joints
            if "dof_damping" in self.rand_params:
                rand_params = [
                    random.uniform(-self.log_scale_limit, self.log_scale_limit)
                    for _ in range(np.prod(self.model.dof_damping.shape))
                ]
                dof_damping_multipliers = np.array(1.3) ** np.array(
                    rand_params
                ).reshape(self.model.dof_damping.shape)
                new_params["dof_damping"] = np.multiply(
                    self.init_params["dof_damping"], dof_damping_multipliers
                )

            # friction at the body components
            if "geom_friction" in self.rand_params:
                rand_params = [
                    random.uniform(-self.log_scale_limit, self.log_scale_limit)
                    for _ in range(np.prod(self.model.geom_friction.shape))
                ]
                dof_damping_multipliers = np.array(1.5) ** np.array(
                    rand_params
                ).reshape(self.model.geom_friction.shape)
                new_params["geom_friction"] = np.multiply(
                    self.init_params["geom_friction"], dof_damping_multipliers
                )

            param_sets.append(new_params)

        return param_sets

    def set_task(self, task):
        for param, param_val in task.items():
            param_variable = getattr(self.model, param)
            assert (
                param_variable.shape == param_val.shape
            ), "shapes of new parameter value and old one must match"
            setattr(self.model, param, param_val)
        self.curr_params = task

    def get_task(self):
        try:
            task = self.curr_params
            task = np.concatenate([task[k].reshape(-1) for k in task.keys()])[
                :, np.newaxis
            ]
            return task
        except AttributeError:
            return None

    def save_parameters(self):
        self.init_params = {}
        if "body_mass" in self.rand_params:
            self.init_params["body_mass"] = self.model.body_mass

        # body_inertia
        if "body_inertia" in self.rand_params:
            self.init_params["body_inertia"] = self.model.body_inertia

        # damping -> different multiplier for different dofs/joints
        if "dof_damping" in self.rand_params:
            self.init_params["dof_damping"] = self.model.dof_damping

        # friction at the body components
        if "geom_friction" in self.rand_params:
            self.init_params["geom_friction"] = self.model.geom_friction
        self.curr_params = self.init_params
