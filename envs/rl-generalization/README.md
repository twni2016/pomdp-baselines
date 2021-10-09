# RL environments for evaluating generalization

This repo contains a set of environments (based on [OpenAI Gym](https://gym.openai.com) and [Roboschool](https://github.com/openai/roboschool)), designed for evaluating generalization in reinforcement learning. We also include implementations of several deep reinforcement learning algorithms (based on [OpenAI Baselines](https://github.com/openai/baselines)), which we have evaluated on these environments.

All environments tested using Python 3.

## Installation

### Using virtualenv

We recommend that you install inside a [virtualenv](https://packaging.python.org/guides/installing-using-pip-and-virtualenv/#creating-a-virtualenv).

Install the environments by checking out this repository and running:
```sh
pip3 install --process-dependency-links -e .
```

Some examples of agents using the environments are provided in the `examples`
directory. They require some additional dependencies, which can be installed by
running:
```sh
pip3 install --process-dependency-links -e .[examples]
```

To get a list of all provided environments, you can run:
```sh
python3 -m examples.list_environments
```

Install Roboschool separately following the instructions [here](https://github.com/openai/roboschool#installation).

### Using Docker	
 You can use Docker to avoid issues while installing dependencies such as Roboschool. You can clone the following Docker image which has all of the dependencies installed:	
```sh
# download docker image
docker pull cpacker/rl-generalization

# start an interactive bash session
docker run -v /path/to/your/copy/of/rl-generalization:/rl-generalization -it cpacker/rl-generalization /bin/bash

# (inside container)
cd /rl-generalization	
python3 -m examples.list_environments
```

## Using the modified environments

You can substitute our environments anywhere you would use an OpenAI Gym environment. For example, instead of:
```python
import gym
env = gym.make('CartPole-v0')
```
You can use one of the following environments:
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

<img src="data/cartpole_combined.png" alt="CartPole D/R/E" width="400"/>

Specific ranges for each environment setting are listed [here](sunblaze_envs#environment-details). See the code in [examples](/examples) for usage with example algorithms from OpenAI Baselines.


## Citations

To cite this repository in your research, you can reference the following [paper](https://arxiv.org/abs/1810.12282):

> Charles Packer, Katelyn Gao, Jernej Kos, Philipp Kr&auml;henb&uuml;hl, Vladlen Koltun, and Dawn Song. Assessing Generalization in Deep Reinforcement Learning. *arXiv preprint arXiv:1810.12282* (2018).

```TeX
@misc{PackerGao:1810.12282,
  Author = {Charles Packer and Katelyn Gao and Jernej Kos and Philipp Kr\"ahenb\"uhl and Vladlen Koltun and Dawn Song},
  Title = {Assessing Generalization in Deep Reinforcement Learning},
  Year = {2018},
  Eprint = {arXiv:1810.12282},
}
```

