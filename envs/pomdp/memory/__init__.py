from gym.envs.registration import register

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
