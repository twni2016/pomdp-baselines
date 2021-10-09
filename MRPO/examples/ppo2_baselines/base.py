import gym
import os

# from baselines.ppo2.policies import CnnPolicy
# from baselines.ppo2.policies import MlpPolicy
# Use the implementation from a2c instead since it supports both cont. and disc.
from ..a2c_baselines import base as a2c_base

# from ..wrappers import wrap_dqn, ScaledFloatFrame


"""
def make_env(env_id, process_idx=0, outdir=None, test=False):
    import sunblaze_envs

    env = gym.make(env_id)
    if outdir:
        env = sunblaze_envs.MonitorParameters(
            env,
            output_filename=os.path.join(outdir, 'env-parameters-{}.json'.format(process_idx))
        )

    if 'Breakout' in env_id or 'SpaceInvaders' in env_id:
        env = ScaledFloatFrame(wrap_dqn(env, clip_rewards=not test, episodic=not test))

    return env
"""

make_env = a2c_base.make_env
mlp_policy = a2c_base.mlp_policy
lstm_policy = a2c_base.lstm_policy
