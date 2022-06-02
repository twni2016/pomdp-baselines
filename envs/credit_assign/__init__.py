from gym.envs.registration import register
from envs.credit_assign.key_to_door import key_to_door

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
