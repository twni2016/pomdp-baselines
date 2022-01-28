import sunblaze_envs

env = sunblaze_envs.make("SunblazeHalfCheetahRandomNormal-v0")  # (26, 6)
# env = sunblaze_envs.make('SunblazeHopperRandomNormal-v0') # (15, 3)
print(env.observation_space, env.action_space)
print(env.action_space.high, env.action_space.low)  # [-1,1]
obs = env.reset()
print(obs)
print(env._max_episode_steps)
done = False
step = 0
while not done:
    step += 1
    obs, rew, done, info = env.step(env.action_space.sample())
    # there exists early failure done=True
    # env.unwrapped.is_success() measures the z >= 20m
    print(step, obs, rew, done, info, env.unwrapped.is_success())
