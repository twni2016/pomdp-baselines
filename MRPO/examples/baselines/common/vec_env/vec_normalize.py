from . import VecEnvWrapper
import numpy as np
import copy


class VecNormalize(VecEnvWrapper):
    """
    A vectorized wrapper that normalizes the observations
    and returns from an environment.
    """

    def __init__(
        self,
        venv,
        ob=True,
        ret=True,
        clipob=10.0,
        cliprew=10.0,
        gamma=0.99,
        epsilon=1e-8,
        use_tf=False,
    ):
        VecEnvWrapper.__init__(self, venv)
        if use_tf:
            from ..running_mean_std import TfRunningMeanStd

            self.ob_rms = (
                TfRunningMeanStd(shape=self.observation_space.shape, scope="ob_rms")
                if ob
                else None
            )
            self.ret_rms = TfRunningMeanStd(shape=(), scope="ret_rms") if ret else None
        else:
            from ..running_mean_std import RunningMeanStd

            self.ob_rms = (
                RunningMeanStd(shape=self.observation_space.shape) if ob else None
            )
            # self.ob_rms = None
            self.ret_rms = RunningMeanStd(shape=()) if ret else None
        self.clipob = clipob
        self.cliprew = cliprew
        self.ret = np.zeros(self.num_envs)
        self.gamma = gamma
        self.epsilon = epsilon

    def step_wait(self):
        obs_bn, rews, news, infos = self.venv.step_wait()

        self.ret = self.ret * self.gamma + rews  # return
        obs = self._obfilt(copy.deepcopy(obs_bn))
        if self.ret_rms:
            self.ret_rms.update(self.ret)
            rews = np.clip(
                rews / np.sqrt(self.ret_rms.var + self.epsilon),
                -self.cliprew,
                self.cliprew,
            )
        self.ret[news] = 0.0
        # print('obs:{}'.format(obs))
        # print('obs_bn:{}'.format(obs_bn))
        return obs, rews, news, infos, obs_bn

    def _obfilt(self, obs):
        if self.ob_rms:
            self.ob_rms.update(obs)
            obs = np.clip(
                (obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon),
                -self.clipob,
                self.clipob,
            )
            return obs
        else:
            return obs

    def reset(self):
        # print('VEC_NOR reset')
        self.ret = np.zeros(self.num_envs)
        obs_bn = self.venv.reset()
        return self._obfilt(obs_bn), obs_bn
