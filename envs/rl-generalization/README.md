# POMDP Environments for Evaluating Generalization and Robustness in RL
Based on the code https://github.com/sunblaze-ucb/rl-generalization

## Using the Generalization Environments
From original SunBlaze repo,

```python
import gym
import sunblaze_envs

# Deterministic: the default version with fixed parameters
fixed_env = sunblaze_envs.make('SunblazeCartPole-v0')

# Random: parameters are sampled from a range nearby the default settings
random_env = sunblaze_envs.make('SunblazeCartPoleRandomNormal-v0')

# Extreme: parameters are sampled from an `extreme' range
extreme_env = sunblaze_envs.make('SunblazeCartPoleRandomExtreme-v0')
```
In the case of CartPole, RandomNormal and RandomExtreme will vary the strength of each actions, the mass of the pole, and the length of the pole:

Specific ranges for each environment setting are listed [here](sunblaze_envs#environment-details). See the code in [examples](/examples) for usage with example algorithms from OpenAI Baselines.

## Using the Robust RL Environments
Similarly, we adopt the environments from [MRPO paper](https://proceedings.mlr.press/v139/jiang21c.html), for example

```python
import gym
import sunblaze_envs

MRPO_walker_env = sunblaze_envs.make('MRPOWalker2dRandomNormal-v0')

```
