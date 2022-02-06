import gym

from envs.meta.wrappers import VariBadWrapper

# In VariBAD, they use on-policy PPO by vectorized env.
# In BOReL, they use off-policy SAC by single env.


def make_env(env_id, episodes_per_task, seed=None, oracle=False, **kwargs):
    """
    kwargs: include n_tasks=num_tasks
    """
    env = gym.make(env_id, **kwargs)
    if seed is not None:
        env.seed(seed)
        env.action_space.np_random.seed(seed)
    env = VariBadWrapper(
        env=env,
        episodes_per_task=episodes_per_task,
        oracle=oracle,
    )
    return env
