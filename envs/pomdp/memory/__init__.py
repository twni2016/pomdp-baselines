from gym.envs.registration import register
from envs.pomdp.memory.key_to_door import key_to_door

delay_fn = lambda runs: (runs - 1) * 7 + 6

for runs in [1, 2, 5, 10, 20, 40]:
    register(
        f"Catch-{runs}-v0",
        entry_point="envs.pomdp.memory.catch:DelayedCatch",
        kwargs=dict(
            delay=delay_fn(runs),
            flatten_img=True,
            one_hot_actions=False,
        ),
        max_episode_steps=delay_fn(runs),
    )

# optimal expected return: 1.0 * (~23) + 10.0 = 33. due to unknown number of respawned apples
register(
    "KeytoDoor-SR-v0",
    entry_point="envs.pomdp.memory.key_to_door.tvt_wrapper:KeyToDoor",
    kwargs=dict(
        flatten_img=True,
        one_hot_actions=False,
        apple_reward=1.0,
        final_reward=10.0,
        respawn_every=20,  # apple respawn after 20 steps
        REWARD_GRID=key_to_door.REWARD_GRID_SR,
    ),
    max_episode_steps=90,
)

# optimal expected return: 1.0 * 10 + 1.0 = 11
register(
    "KeytoDoor-LowVar-v0",
    entry_point="envs.pomdp.memory.key_to_door.tvt_wrapper:KeyToDoor",
    kwargs=dict(
        flatten_img=True,
        one_hot_actions=False,
        apple_reward=1.0,
        final_reward=1.0,
        respawn_every=0,  # apple never respawn
        REWARD_GRID=key_to_door.REWARD_GRID_CCA,
    ),
    max_episode_steps=90,
)

# optimal expected return: (1.0+10.0)/2 * 10 + 1.0 = 56
register(
    "KeytoDoor-HighVar-v0",
    entry_point="envs.pomdp.memory.key_to_door.tvt_wrapper:KeyToDoor",
    kwargs=dict(
        flatten_img=True,
        one_hot_actions=False,
        apple_reward=(1.0, 10.0),  # random pick one as reward
        final_reward=1.0,
        respawn_every=0,  # apple never respawn
        REWARD_GRID=key_to_door.REWARD_GRID_CCA,
    ),
    max_episode_steps=90,
)
