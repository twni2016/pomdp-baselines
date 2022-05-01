import gym
from gym.envs.registration import register
from envs.credit_assign.key_to_door import key_to_door
from envs.credit_assign.episodic_reward import MuJoCoEpisodicRewardEnv

delay_fn = lambda runs: (runs - 1) * 7 + 6

for runs in [1, 2, 5, 10, 20, 40]:
    register(
        f"Catch-{runs}-v0",
        entry_point="envs.credit_assign.catch:DelayedCatch",
        kwargs=dict(
            delay=delay_fn(runs),
            flatten_img=True,
            one_hot_actions=False,
        ),
        max_episode_steps=delay_fn(runs),
    )

# optimal expected return: 1.0 * (~23) + 5.0 = 28. due to unknown number of respawned apples
register(
    "KeytoDoor-SR-v0",
    entry_point="envs.credit_assign.key_to_door.tvt_wrapper:KeyToDoor",
    kwargs=dict(
        flatten_img=True,
        one_hot_actions=False,
        apple_reward=1.0,
        final_reward=5.0,
        respawn_every=20,  # apple respawn after 20 steps
        REWARD_GRID=key_to_door.REWARD_GRID_SR,
        max_frames=key_to_door.MAX_FRAMES_PER_PHASE_SR,
    ),
    max_episode_steps=sum(key_to_door.MAX_FRAMES_PER_PHASE_SR.values()),
)

# optimal expected return: 1.0 * 10 + 1.0 = 11
register(
    "KeytoDoor-LowVar-v0",
    entry_point="envs.credit_assign.key_to_door.tvt_wrapper:KeyToDoor",
    kwargs=dict(
        flatten_img=True,
        one_hot_actions=False,
        apple_reward=1.0,
        final_reward=1.0,
        respawn_every=0,  # apple never respawn
        REWARD_GRID=key_to_door.REWARD_GRID_CCA,
        max_frames=key_to_door.MAX_FRAMES_PER_PHASE_CCA,
    ),
    max_episode_steps=sum(key_to_door.MAX_FRAMES_PER_PHASE_CCA.values()),
)

# optimal expected return: 1.0 * 10 + 5.0 = 15
register(
    "KeytoDoor-LowVar5-v0",
    entry_point="envs.credit_assign.key_to_door.tvt_wrapper:KeyToDoor",
    kwargs=dict(
        flatten_img=True,
        one_hot_actions=False,
        apple_reward=1.0,
        final_reward=5.0,
        respawn_every=0,  # apple never respawn
        REWARD_GRID=key_to_door.REWARD_GRID_CCA,
        max_frames=key_to_door.MAX_FRAMES_PER_PHASE_CCA,
    ),
    max_episode_steps=sum(key_to_door.MAX_FRAMES_PER_PHASE_CCA.values()),
)


# optimal expected return: (1.0+10.0)/2 * 10 + 1.0 = 56
register(
    "KeytoDoor-HighVar-v0",
    entry_point="envs.credit_assign.key_to_door.tvt_wrapper:KeyToDoor",
    kwargs=dict(
        flatten_img=True,
        one_hot_actions=False,
        apple_reward=(1.0, 10.0),  # random pick one as reward
        final_reward=1.0,
        respawn_every=0,  # apple never respawn
        REWARD_GRID=key_to_door.REWARD_GRID_CCA,
        max_frames=key_to_door.MAX_FRAMES_PER_PHASE_CCA,
    ),
    max_episode_steps=sum(key_to_door.MAX_FRAMES_PER_PHASE_CCA.values()),
)

# optimal expected return: (1.0+10.0)/2 * 10 + 5.5 = 60.5
register(
    "KeytoDoor-HighVar5-v0",
    entry_point="envs.credit_assign.key_to_door.tvt_wrapper:KeyToDoor",
    kwargs=dict(
        flatten_img=True,
        one_hot_actions=False,
        apple_reward=(1.0, 10.0),  # random pick one as reward
        final_reward=5.5,
        respawn_every=0,  # apple never respawn
        REWARD_GRID=key_to_door.REWARD_GRID_CCA,
        max_frames=key_to_door.MAX_FRAMES_PER_PHASE_CCA,
    ),
    max_episode_steps=sum(key_to_door.MAX_FRAMES_PER_PHASE_CCA.values()),
)

# optimal expected return: (1.0+10.0)/2 * 10 + 10 = 65
register(
    "KeytoDoor-HighVar10-v0",
    entry_point="envs.credit_assign.key_to_door.tvt_wrapper:KeyToDoor",
    kwargs=dict(
        flatten_img=True,
        one_hot_actions=False,
        apple_reward=(1.0, 10.0),  # random pick one as reward
        final_reward=10.0,
        respawn_every=0,  # apple never respawn
        REWARD_GRID=key_to_door.REWARD_GRID_CCA,
        max_frames=key_to_door.MAX_FRAMES_PER_PHASE_CCA,
    ),
    max_episode_steps=sum(key_to_door.MAX_FRAMES_PER_PHASE_CCA.values()),
)

mujoco_list = [
    "Ant-v2",
    "HalfCheetah-v2",
    "Walker2d-v2",
    "Humanoid-v2",
    "Reacher-v2",
    "Swimmer-v2",
    "Hopper-v2",
    "HumanoidStandup-v2",
]

for env_name in mujoco_list:
    register(
        env_name.replace("-v2", "Ep-v2"),
        entry_point="envs.credit_assign.episodic_reward:MuJoCoEpisodicRewardEnv",
        kwargs=dict(
            env=gym.make(env_name),
        ),
        max_episode_steps=1000,
    )
