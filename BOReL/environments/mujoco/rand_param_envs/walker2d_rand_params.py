import numpy as np
from environments.mujoco.rand_param_envs.base import RandomEnv
from environments.mujoco.rand_param_envs.gym import utils


class Walker2DRandParamsEnv(RandomEnv, utils.EzPickle):
    def __init__(self, log_scale_limit=3.0):
        self._max_episode_steps = 200
        self._elapsed_steps = -1  # the thing below takes one step
        RandomEnv.__init__(self, log_scale_limit, "walker2d.xml", 5)
        utils.EzPickle.__init__(self)

    def _step(self, a):
        posbefore = self.model.data.qpos[0, 0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.model.data.qpos[0:3, 0]
        alive_bonus = 1.0
        reward = (posafter - posbefore) / self.dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        done = not (height > 0.8 and height < 2.0 and ang > -1.0 and ang < 1.0)
        ob = self._get_obs()
        self._elapsed_steps += 1
        info = {"task": self.get_task()}
        if self._elapsed_steps == self._max_episode_steps:
            done = True
            info["bad_transition"] = True
        return ob, reward, done, info

    def _get_obs(self):
        qpos = self.model.data.qpos
        qvel = self.model.data.qvel
        return np.concatenate([qpos[1:], np.clip(qvel, -10, 10)]).ravel()

    def reset_model(self):
        self.set_state(
            self.init_qpos
            + self.np_random.uniform(low=-0.005, high=0.005, size=self.model.nq),
            self.init_qvel
            + self.np_random.uniform(low=-0.005, high=0.005, size=self.model.nv),
        )
        return self._get_obs()

    def _reset(self):
        ob = super()._reset()
        self._elapsed_steps = 0
        return ob

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.5
        self.viewer.cam.lookat[2] += 0.8
        self.viewer.cam.elevation = -20


class Walker2DRandParamsOracleEnv(Walker2DRandParamsEnv):
    def _get_obs(self):
        if hasattr(self, "cur_params"):
            task = self.get_task()
            task = np.concatenate([task[k].reshape(-1) for k in task.keys()])[
                :, np.newaxis
            ]
        else:
            task = np.zeros((self.rand_param_dim, 1))
        qpos = self.model.data.qpos
        qvel = self.model.data.qvel
        return np.concatenate([qpos[1:], np.clip(qvel, -10, 10), task]).ravel()
