# POMDP Environments

## Overview
- `meta/`: Meta RL environments
- `pomdp/`: "standard" POMDP environments
- `rl-generalization`: Generalization in RL and Robust RL environments

## Normalized Action Space
We make sure every environment has continuous action space [-1, 1]^|A|, exposed to the policy. Policy should not use `self.action_space.high` or `self.action_space.low`.

In Meta RL and "standard" POMDP, we use the snipplet for normalizing the action space:
```python 
class EnvWrapper(gym.Wrapper):
    def step(self, action):
        action = np.clip(action, -1, 1) # first clip into [-1, 1]
        lb = self.env.action_space.low
        ub = self.env.action_space.high
        action = lb + (action + 1.) * 0.5 * (ub - lb) # recover the original action space
        action = np.clip(action, lb, ub)
        ...
```
