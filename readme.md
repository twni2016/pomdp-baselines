# Recurrent Model-Free RL can be a Strong Baseline for Many POMDPs
Welcome to the POMDP world! 

This repo provides some simple baselines for POMDPs, specifically the **recurrent model-free RL**, on the benchmarks in **several subareas of POMDPs** (including meta RL, robust RL, generalization in RL, temporal credit assignment) for the following paper:

[[arXiv]](https://arxiv.org/abs/2110.05038) [[project site]](https://sites.google.com/view/pomdp-baselines) [[numeric results]](https://drive.google.com/file/d/1dfulN8acol-qaNR2h4PDpIaWBg9Ck4pY/view?usp=sharing)

by [Tianwei Ni](https://twni2016.github.io/), [Benjamin Eysenbach](https://ben-eysenbach.github.io/) and [Ruslan Salakhutdinov](http://www.cs.cmu.edu/~rsalakhu/). **To show in ICML 2022.**

## Motivation

### RL mostly studies on MDPs, why POMDPs?
While MDPs prevail in RL research, POMDPs prevail in real world and life. In many real problems (robotics, healthcare, finance, human interaction), we inevitably face with partial observability, e.g. noisy sensors and lack of sensors. Can we really observe the "states"? Where do "states" come from? 

Moreover, in RL research, there are many problems that can be cast as POMDPs: meta RL, robust RL, and generalization in RL. Within a more suitable framework, we can develop better RL algorithms. 

### Why using recurrent model-free RL for POMDP? What about other methods? 
It is an open research area on deep RL algorithms for POMDPs. Among them, recurrent model-free RL, developed with a long history, is simple to implement, easy to understand, and trained end-to-end. Nonetheless, there is a popular belief that it performs poorly in practice. This repo revisits it and provides some guildlines on the design of its key components, to make it stronger. 

There are many other (more complicated or specialized) methods for POMDPs and its subareas. We show recurrent model-free RL, if well designed, can _often_ outperform _some_ of these methods in their benchmarks. It could be served as a strong baseline to incentivize future work. 


## CHANGE LOG
Note that current repo should be run smoothly.

DONE:
* May 2022: this work has been accepted to **ICML 2022**! 
* Mar 2022: introduce recurrent [SAC-discrete](https://arxiv.org/abs/1910.07207) for **discrete action** space and see [this PR for instructions](https://github.com/twni2016/pomdp-baselines/pull/1). As a baseline, it [greatly improves sample efficiency](https://github.com/twni2016/pomdp-baselines/pull/2), compared to a specialized method IMPALA+SR, on their long-term credit assignment benchmark.
* Feb 2022: simplify `--oracle` commands, and upload the plotting scripts
* Jan 2022: introduce new meta RL environments (*-Dir), and replace re-implementation of off-policy variBAD with original implementation
* Dec 2021: add some command-line arguments to overwrite the config file and save the updated one
* Dec 2021: fix [seed reproducibility issue](envs/readme.md#reproducibilty-issue-in-gym-environments) for gym v0.21 (but not for SunBlaze)
* Nov 2021: add Markovian and Oracle policies training

TODO:
- Add documentation on our main code and log csv files 

## A Minimal Example to Run Our Implementation
Here we provide a stand-alone minimal example with the least dependencies to run our implementation of recurrent model-free RL! 
> Only requires PyTorch and PyBullet, no need to install MuJoCo or roboschool, no external configuration file.

Simply open the Jupyter Notebook [example.ipynb](example.ipynb) and it contains the training and evaluation procedure on a toy POMDP environment (Pendulum-V). It only costs < 20 min to run the whole process on a GPU.

## Installation
First download this repo into your local directory (preferably on a cluster or a server) to <local_path>. Then we recommend to use a virtual env to install all the dependencies. We provide the yaml file to install using miniconda:

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

See [run_commands.md](docs/run_commands.md) for our estimated difficulty levels of these environments.

### General Form of Commands
**We use `.yml` file in `configs/` folder for training, and then we can overwrite the config file by command-line arguments for our implementation.**

To run our implementation, Markovian, and oracle, in <local_path> simply
```
export PYTHONPATH=${PWD}:$PYTHONPATH
python3 policies/main.py --cfg configs/<subarea>/<env_name>/<algo_name>.yml \
  [--env <env_name> --algo {td3,sac,sacd} --seed <int> --cuda <int> --oracle
   --(no)automatic_entropy_tuning --target_entropy <float> --entropy_alpha <float>]
```
where `algo_name` specifies the algorithm name:
- `mlp` correspond to **Markovian** policies
- `rnn` correspond to **our implementation** of recurrent model-free RL
- `ppo_rnn` and `a2c_rnn` correspond to [(Kostrikov, 2018)](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail) implementation of recurrent model-free RL
- `vrm` corresponds to [VRM](https://github.com/oist-cnru/Variational-Recurrent-Models) compared in "standard" POMDPs
- `MRPO` correspond to [MRPO](https://proceedings.mlr.press/v139/jiang21c.html) compared in robust RL

> We have merged the prior methods above into our repository (there is no need to install other repositories), so that future work can use this single repository to run a number of baselines besides ours: A2C-GRU, PPO-GRU, VRM, off-policy variBAD, MRPO. 
>
> Since our code is heavily drawn from those prior works, we encourage authors to [cite those prior papers or implementations](docs/acknowledge.md).
>
> For the compared methods, we use their open-sourced implementation with their default hyperparameters.

### Specific Running Commands for Each Subarea
Please see [run_commands.md](docs/run_commands.md) for details on running our implementation of recurrent model-free RL and also all the compared methods.

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
@article{ni2021recurrent,
  title={Recurrent Model-Free RL can be a Strong Baseline for Many POMDPs},
  author={Ni, Tianwei and Eysenbach, Benjamin and Salakhutdinov, Ruslan},
  journal={arXiv preprint arXiv:2110.05038},
  year={2021}
}
```

## Contribution
Before pull request, please reformat your code:
```bash
black . -t py35 # avoid trailing commas issue after kwargs
```

## Contact
If you have any questions, please create an issue in this repo or contact Tianwei Ni (tianwei.ni@mila.quebec)

