# Env Wrappers
We make sure every env wrapper has continuous action space [-1, 1]^|A|, 
and policy should not use `self.action_space.high` or `self.action_space.low`.

The snipplet for normalizing action space is: 
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
