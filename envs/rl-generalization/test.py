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

# train on D, eval on D, R, E
# train on R, eval on R, E
# (DD), (RR), (DR, DR, RE)

# train 15000 epsiodes (<15M, but I don't know the exact number, since it may early stop)
# and eval on the last checkpoint on the average of 1000 episodes (1M)
# oring on success

# they try context len [5, 10, 15] (for A2C) and [128, 256, 512] for PPO (we can try 5, 64, 512)
#   which is also proportional to training freq: they SGD once per context len
#   therefore, we can try freq 1/5 or 1/10
# they also try RL2 (N=1), so we need to try inputs [o, oar]
# they use LSTM, thus we also use LSTM
