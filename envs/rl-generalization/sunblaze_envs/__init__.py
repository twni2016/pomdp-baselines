from .registration import register, registry, make
from .monitor import MonitorParameters


# Classic control environments.

register(
    id="SunblazeCartPole-v0",
    entry_point="sunblaze_envs.classic_control:ModifiableCartPoleEnv",
    max_episode_steps=200,
    reward_threshold=195.0,
)

register(
    id="SunblazeCartPoleRandomNormal-v0",
    entry_point="sunblaze_envs.classic_control:RandomNormalCartPole",
    max_episode_steps=200,
    reward_threshold=195.0,
)

register(
    id="SunblazeCartPoleRandomExtreme-v0",
    entry_point="sunblaze_envs.classic_control:RandomExtremeCartPole",
    max_episode_steps=200,
    reward_threshold=195.0,
)

register(
    id="SunblazeMountainCar-v0",
    entry_point="sunblaze_envs.classic_control:ModifiableMountainCarEnv",
    max_episode_steps=200,
    reward_threshold=-110.0,
)

register(
    id="SunblazeMountainCarRandomNormal-v0",
    entry_point="sunblaze_envs.classic_control:RandomNormalMountainCar",
    max_episode_steps=200,
    reward_threshold=-110.0,
)

register(
    id="SunblazeMountainCarRandomExtreme-v0",
    entry_point="sunblaze_envs.classic_control:RandomExtremeMountainCar",
    max_episode_steps=200,
    reward_threshold=-110.0,
)

register(
    id="SunblazePendulum-v0",
    entry_point="sunblaze_envs.classic_control:ModifiablePendulumEnv",
    max_episode_steps=200,
)

register(
    id="SunblazePendulumRandomNormal-v0",
    entry_point="sunblaze_envs.classic_control:RandomNormalPendulum",
    max_episode_steps=200,
)

register(
    id="SunblazePendulumRandomExtreme-v0",
    entry_point="sunblaze_envs.classic_control:RandomExtremePendulum",
    max_episode_steps=200,
)

register(
    id="SunblazeAcrobot-v0",
    entry_point="sunblaze_envs.classic_control:ModifiableAcrobotEnv",
    max_episode_steps=500,
)

register(
    id="SunblazeAcrobotRandomNormal-v0",
    entry_point="sunblaze_envs.classic_control:RandomNormalAcrobot",
    max_episode_steps=500,
)

register(
    id="SunblazeAcrobotRandomExtreme-v0",
    entry_point="sunblaze_envs.classic_control:RandomExtremeAcrobot",
    max_episode_steps=500,
)

# Mujoco environments

register(
    id="SunblazeHopper-v0",
    entry_point="sunblaze_envs.mujoco:ModifiableRoboschoolHopper",
    max_episode_steps=1000,
    reward_threshold=3800.0,
)

register(
    id="SunblazeHopperRandomNormal-v0",
    entry_point="sunblaze_envs.mujoco:RandomNormalHopper",
    max_episode_steps=1000,
    reward_threshold=3800.0,
)

register(
    id="SunblazeHopperRandomExtreme-v0",
    entry_point="sunblaze_envs.mujoco:RandomExtremeHopper",
    max_episode_steps=1000,
    reward_threshold=3800.0,
)

register(
    id="SunblazeHalfCheetah-v0",
    entry_point="sunblaze_envs.mujoco:ModifiableRoboschoolHalfCheetah",
    max_episode_steps=1000,
    reward_threshold=4800.0,
)

register(
    id="SunblazeHalfCheetahRandomNormal-v0",
    entry_point="sunblaze_envs.mujoco:RandomNormalHalfCheetah",
    max_episode_steps=1000,
    reward_threshold=4800.0,
)

register(
    id="SunblazeHalfCheetahRandomExtreme-v0",
    entry_point="sunblaze_envs.mujoco:RandomExtremeHalfCheetah",
    max_episode_steps=1000,
    reward_threshold=4800.0,
)

# MRPO mujoco for robust RL benchmarks (Jiang 2021)

register(
    id="MRPOWalker2dRandomNormal-v0",
    entry_point="sunblaze_envs.mujoco:RandomNormalWalker2d_MRPO",
    max_episode_steps=1000,
    reward_threshold=4800.0,
)

register(
    id="MRPOHopperRandomNormal-v0",
    entry_point="sunblaze_envs.mujoco:RandomNormalHopper_MRPO",
    max_episode_steps=1000,
    reward_threshold=4800.0,
)

register(
    id="MRPOHalfCheetahRandomNormal-v0",
    entry_point="sunblaze_envs.mujoco:RandomNormalHalfCheetah_MRPO",
    max_episode_steps=1000,
    reward_threshold=4800.0,
)
