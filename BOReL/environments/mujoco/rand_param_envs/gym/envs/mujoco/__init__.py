from environments.mujoco.rand_param_envs.gym.envs.mujoco.mujoco_env import MujocoEnv

# ^^^^^ so that user gets the correct error
# message if mujoco is not installed correctly
from environments.mujoco.rand_param_envs.gym.envs.mujoco.ant import AntEnv
from environments.mujoco.rand_param_envs.gym.envs.mujoco.half_cheetah import (
    HalfCheetahEnv,
)
from environments.mujoco.rand_param_envs.gym.envs.mujoco.hopper import HopperEnv
from environments.mujoco.rand_param_envs.gym.envs.mujoco.walker2d import Walker2dEnv
from environments.mujoco.rand_param_envs.gym.envs.mujoco.humanoid import HumanoidEnv
from environments.mujoco.rand_param_envs.gym.envs.mujoco.inverted_pendulum import (
    InvertedPendulumEnv,
)
from environments.mujoco.rand_param_envs.gym.envs.mujoco.inverted_double_pendulum import (
    InvertedDoublePendulumEnv,
)
from environments.mujoco.rand_param_envs.gym.envs.mujoco.reacher import ReacherEnv
from environments.mujoco.rand_param_envs.gym.envs.mujoco.swimmer import SwimmerEnv
from environments.mujoco.rand_param_envs.gym.envs.mujoco.humanoidstandup import (
    HumanoidStandupEnv,
)
