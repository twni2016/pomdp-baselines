# Recurrent Model-Free RL Can Be a Strong Baseline for Many POMDPs
Welcome to the POMDP world! 

This repository provides some simple baselines for POMDPs, specifically the **recurrent model-free RL**, on the benchmarks in **several subareas of POMDPs** (including meta RL, robust RL, generalization in RL, temporal credit assignment) for the following paper accepted to **ICML 2022**: 

*Recurrent Model-Free RL Can Be a Strong Baseline for Many POMDPs.* By [Tianwei Ni](https://twni2016.github.io/) (Mila, CMU), [Benjamin Eysenbach](https://ben-eysenbach.github.io/) (CMU) and [Ruslan Salakhutdinov](http://www.cs.cmu.edu/~rsalakhu/) (CMU).

[[arXiv]](https://arxiv.org/abs/2110.05038) [[project site]](https://sites.google.com/view/pomdp-baselines) [[numeric results]](https://drive.google.com/file/d/1dfulN8acol-qaNR2h4PDpIaWBg9Ck4pY/view?usp=sharing)

## Interested in Transformer Model-Free RL?
Check out our recent work on [When Do Transformers Shine in RL? Decoupling Memory from Credit Assignment](https://arxiv.org/abs/2307.03864) with the [code](https://github.com/twni2016/Memory-RL) (**NeurIPS 2023 oral**) based on this repository! 

## Motivation

### RL mostly studies on MDPs, why POMDPs?
While MDPs prevail in RL research, POMDPs prevail in the real world and life. In many real problems (robotics, healthcare, finance, human interaction), we inevitably face partial observability, e.g. noisy sensors and lack of sensors. Can we observe "states"? Where do "states" come from? 

Moreover, in RL research, many problems can be cast as POMDPs: meta RL, robust RL, generalization in RL, and temporal credit assignment. Within a more suitable framework, we can develop better RL algorithms. 

### Why use recurrent model-free RL for POMDP? What about other methods? 
It is an open research area on deep RL algorithms for POMDPs. Among them, recurrent model-free RL, developed with a long history, is simple to implement, easy to understand, and trained end-to-end. Nonetheless, there is a popular belief that it performs poorly in practice. This work revisits it and provides some guidelines on the design of its key components, to make it stronger. 

There are many other (more complicated or specialized) methods for POMDPs and their subareas. We show recurrent model-free RL, if well designed, can _often_ outperform _some_ of these methods in their benchmarks. It could be served as a strong baseline to incentivize future work. 


## CHANGE LOG

* Jul 2022: Move the code for the compared methods to [a new branch](https://github.com/twni2016/pomdp-baselines/tree/all-methods)
* Jun 2022: Cleaned and refactored the code for camera ready.
* May 2022: this work has been accepted to **ICML 2022**! 
* Mar 2022: introduce recurrent [SAC-discrete](https://arxiv.org/abs/1910.07207) for **discrete action** space and see [this PR for instructions](https://github.com/twni2016/pomdp-baselines/pull/1). As a baseline, it [greatly improves sample efficiency](https://github.com/twni2016/pomdp-baselines/pull/2), compared to a specialized method IMPALA+SR, on their long-term credit assignment benchmark.

## A Minimal Example to Run Our Implementation
Here we provide a stand-alone minimal example with the least dependencies to run our implementation of recurrent model-free RL! 
> Only requires PyTorch and PyBullet, no need to install MuJoCo or roboschool, no external configuration file.

Simply open the Jupyter Notebook [example.ipynb](example.ipynb) and it contains the training and evaluation procedure on a toy POMDP environment (Pendulum-V). It only costs < 20 min to run the whole process on a GPU.

## Installation
First download this repository into your local directory (preferably on a cluster or a server) to <local_path>. Then we recommend using a virtual env to install all the dependencies. We provide the yaml file to install using miniconda:

```bash
conda env create -f environments.yml
conda activate pomdp
```

The `environments.yml` file includes all the dependencies (e.g. MuJoCo, PyTorch, PyBullet) used in our experiments (including the compared methods), where we use `mujoco-py=2.1` as [it is free to use without license](https://github.com/openai/mujoco-py/releases/tag/v2.1.2.14).

However, to run robust RL and generalization in RL experiments, you have to install [Roboschool](https://github.com/openai/roboschool). We found it hard to install Roboschool from scratch, therefore we provide a docker file `roboschool.sif` in [google drive](https://drive.google.com/file/d/1KpTpVwoU02AI7uQrk2T9hQ6s15EISRTa/view?usp=sharing) that contains Roboschool and the other necessary libraries, adapted from [SunBlaze repo](https://github.com/sunblaze-ucb/rl-generalization). 
  - To download and activate the docker file by singularity (tested in v3.7) on a cluster (on a single server should be similar):
    ```bash
    # download roboschool.sif from the google drive to envs/rl-generalization/roboschool.sif
    # then run singularity shell
    singularity shell --nv -H <local_path>:/home envs/rl-generalization/roboschool.sif
    ```
  - Then you can test it by `import roboschool` in a `python3` shell.

## Run Our Implementation of Recurrent Model-Free RL and the Compared Methods

### Benchmarks / Environments

We support several benchmarks in different subareas of POMDPs (see `envs/` for details), including

* "Standard" POMDPs: occlusion benchmark in PyBullet
* Meta RL: gridworld and MuJoCo benchmark
* Robust RL: SunBlaze benchmark in Roboschool
* Generalization in RL: SunBlaze benchmark in Roboschool
* Temporal credit assignment: delayed rewards with pixel observation and discrete control

Before starting running any experiments, we suggest having a good plan of *environment series* based on difficulty level. As it is hard to analyze and varies from algorithm to algorithm, we provide some rough estimates:

1. Extremely Simple as a Sanity Check: Pendulum-V (also shown in our minimal example jupyter notebook) and CartPole-V (for discrete action space)
2. Simple, Fast, yet Non-trivial: Wind (require precise inference and control), Semi-Circle (sparse reward). Both are continuous gridworlds, thus very fast.
3. Medium: Cheetah-Vel (1-dim stationary hidden state), `*`-Robust (2-dim stationary hidden state), `*`-P (could be roughly inferred by 2nd order MDP)
4. Hard: `*`-Dir (relatively complicated dynamics), `*`-V (long-term inference), `*`-Generalize (extrapolation)

### General Form of Commands
**We use `.yml` file in `configs/` folder for training, and then we can overwrite the config file with command-line arguments for our implementation.**

To run our implementation, Markovian, and oracle, in <local_path> simply
```
export PYTHONPATH=${PWD}:$PYTHONPATH
python3 policies/main.py --cfg configs/<subarea>/<env_name>/<algo_name>.yml \
  [--env <env_name>  --oracle
   --algo {td3,sac,sacd} --(no)automatic_entropy_tuning --target_entropy <float> --entropy_alpha <float>
   --debug --seed <int> --cuda <int>
  ]
```
where `algo` specifies the algorithm name:
- `mlp` correspond to **Markovian** policies
- `rnn` correspond to **our implementation** of recurrent model-free RL

We have merged the prior methods above into our repository: please see [the `all-methods` branch](https://github.com/twni2016/pomdp-baselines/tree/all-methods).
> For the compared methods, we use their open-sourced implementation with their default hyperparameters.


### "Standard" POMDP

{Ant,Cheetah,Hopper,Walker}-{P,V} in the paper, corresponding to `configs/pomdp/<ant|cheetah|hopper|walker>_blt/<p|v>`, which requires PyBullet. We also provide Pendulum environments for sanity check.

Take Ant-P as an example:
```bash
# Run our implementation
python policies/main.py --cfg configs/pomdp/ant_blt/p/rnn.yml --algo sac
# Run Markovian
python policies/main.py --cfg configs/pomdp/ant_blt/p/mlp.yml --algo sac
# Oracle: we directly use Table 1 results (SAC w/ unstructured row) in https://arxiv.org/abs/2005.05719 as it is well-tuned
``` 

We also support recurrent SAC-discrete for POMDPs with **discrete action space**. Take CartPole-V as an example:
```
python policies/main.py --cfg configs/pomdp/cartpole/v/rnn.yml --target_entropy 0.7
```
See [this PR for detailed instructions](https://github.com/twni2016/pomdp-baselines/pull/1) and [this PR for results on a long-term credit assignment benchmark](https://github.com/twni2016/pomdp-baselines/pull/2).

### Meta RL 

{Semi-Circle, Wind, Cheetah-Vel} in the paper, corresponding to `configs/meta/<point_robot|wind|cheetah_vel|ant_dir>`. Among them, Cheetah-Vel requires MuJoCo, and Semi-Circle can serve as a sanity check. Wind looks simple but is not very easy to solve.

Take Semi-Circle as an example:
```bash
# Run our implementation
python policies/main.py --cfg configs/meta/point_robot/rnn.yml --algo td3
# Run Markovian
python policies/main.py --cfg configs/meta/point_robot/mlp.yml --algo sac
# Run Oracle
python policies/main.py --cfg configs/meta/point_robot/mlp.yml --algo sac --oracle
```

{Ant, Cheetah, Humanoid}-Dir in the paper, corresponding to `configs/meta/<ant_dir|cheetah_dir|humanoid_dir>`. They require MuJoCo and are hard to solve.
Take Ant-Dir as an example:
```bash
# Run our implementation
python policies/main.py --cfg configs/meta/ant_dir/rnn.yml --algo sac
# Run Markovian
python policies/main.py --cfg configs/meta/ant_dir/mlp.yml --algo sac
# Run Oracle
python policies/main.py --cfg configs/meta/ant_dir/mlp.yml --algo sac --oracle
```

### Robust RL
Use roboschool. {Hopper,Walker,Cheetah}-Robust in the paper, corresponding to `configs/rmdp/<hopper|walker|cheetah>`. First, activate the roboschool docker env as introduced in the installation section. 

Take Cheetah-Robust as an example:
```bash
## In the docker environment:
# Run our implementation
python3 policies/main.py --cfg configs/rmdp/cheetah/rnn.yml --algo td3
# Run Markovian
python3 policies/main.py --cfg configs/rmdp/cheetah/mlp.yml --algo sac
# Run Oracle
python3 policies/main.py --cfg configs/rmdp/cheetah/mlp.yml --algo sac --oracle
```

### Generalization in RL
Use roboschool. {Hopper|Cheetah}-Generalize in the paper, corresponding to `configs/generalize/Sunblaze<Hopper|HalfCheetah>/<DD-DR-DE|RD-RR-RE>`. 
First, activate the roboschool docker env as introduced in the installation section. 

To train on Default environment and test on the all environments, use `*DD-DR-DE*.yml`; to train on Random environment and test on the all environments, use use `*RD-RR-RE*.yml`. Please see the [SunBlaze paper](https://arxiv.org/abs/1810.12282) for details. 

Take running on `SunblazeHalfCheetahRandomNormal-v0` as an example:
```bash
## In the docker environment:
# Run our implementation
python3 policies/main.py --cfg configs/generalize/SunblazeHalfCheetah/RD-RR-RE/rnn.yml --algo td3
# Run Markovian
python3 policies/main.py --cfg configs/generalize/SunblazeHalfCheetah/RD-RR-RE/mlp.yml --algo sac
# Run Oracle
python3 policies/main.py --cfg configs/generalize/SunblazeHalfCheetah/RD-RR-RE/mlp.yml --algo sac --oracle
```

### Temporal Credit Assignment
{Delayed-Catch, Key-to-Door} in the paper, corresponding to `configs/credit/<catch|keytodoor>`. Note that this is discrete control on pixel inputs, so the architecture is a bit different from the default one.

To reproduce our results, please run:
```bash
python3 policies/main.py --cfg configs/credit/catch/rnn.yml
python3 policies/main.py --cfg configs/credit/keytodoor/rnn.yml
```

### Atari
Although Atari environments are **not** this paper's focus, we provide an implementation to train on a game, following the Dreamerv2 setting. The hyperparameters are **not** well-tuned, so the results are not expected to be good.

```bash
# train on Pong (confirmed it can work on Pong)
python3 policies/main.py --cfg configs/atari/rnn.yml --env Pong
```

## Misc

### Draw and Download the Learning Curves 
Please see [plot_curves.md](docs/plot_curves.md) for details on plotting. 

### Details of Our Implementation of Recurrent Model-Free RL: Decision Factors, Best Variants, Code Features
Please see [our_details.md](docs/our_details.md) for more information on:
- How to tune the decision factors discussed in the paper in the configuration files
- How to tune the other hyperparameters that are also important to training
- Our best variants in each subarea and numeric results of learning curves

## Acknowledgement
Please see [acknowledge.md](docs/acknowledge.md) for details.

## Citation
If you find our code useful to your work, please consider citing our paper:
```
@inproceedings{ni2022recurrent,
  title={Recurrent Model-Free {RL} Can Be a Strong Baseline for Many {POMDP}s},
  author={Ni, Tianwei and Eysenbach, Benjamin and Salakhutdinov, Ruslan},
  booktitle={International Conference on Machine Learning},
  pages={16691--16723},
  year={2022},
  organization={PMLR}
}
```

## Contributing
Before pull request, please reformat your code:
```bash
# avoid trailing commas issue after kwargs
black . -t py35
```

## Other Implementations
You may find other PyTorch implementations on recurrent model-free RL useful:
- [Recurrent Off-policy Baselines for Memory-based Continuous Control](https://github.com/zhihanyang2022/off-policy-continuous-control) has recurrent TD3/SAC
- [Task-Agnostic Continual RL: In Praise of a Simple Baseline](https://github.com/amazon-research/replay-based-recurrent-rl) has recurrent SAC for task-agnostic continual RL
- [Tianshou](https://github.com/thu-ml/tianshou) has recurrent support
- [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3) has recurrent PPO
- [RLlib](https://docs.ray.io/en/master/rllib/rllib-algorithms.html) has recurrent PPO
- [CleanRL](https://github.com/vwxyzjn/cleanrl) has recurrent PPO

## Contact
If you have any questions, please create an issue in this repository or contact Tianwei Ni (tianwei.ni@mila.quebec)

