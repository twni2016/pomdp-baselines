# Recurrent Model-Free RL is a Strong Baseline for Many POMDPs
Welcome to the POMDP world! 
This repo provides some simple baselines for POMDPs, specifically the recurrent model-free RL, for the following paper

Paper: [arXiv](https://arxiv.org/abs/2110.05038) Numeric Results: [google drive](https://drive.google.com/file/d/18l9Y4N8zPRdGBnx8oSELiQcoReF7V4wP/view?usp=sharing) Web: [Site](https://sites.google.com/view/pomdp-baselines)

by [Tianwei Ni](https://twni2016.github.io/), [Benjamin Eysenbach](https://ben-eysenbach.github.io/) and [Ruslan Salakhutdinov](http://www.cs.cmu.edu/~rsalakhu/).

## Installation
First download this repo into your local directory (preferably on a cluster or a server) <local_path>. Then we recommend to use a virtual env to install all the dependencies. For example, we install using miniconda and pip:

```bash
conda create -n pomdp python==3.8
conda activate pomdp
pip install -r requirements.txt
```

The `requirements.txt` file includes all the dependencies (e.g. PyTorch, PyBullet) used in our experiments (including compared methods), but there are two exceptions:
- To run Cheetah-Vel in meta RL, you have to install [MuJoCo](https://github.com/openai/mujoco-py) with a license on your own
- To run robust RL and generalization in RL experiments, you have to install [roboschool](https://github.com/openai/roboschool). 
    - We found it hard to install roboschool from scratch, therefore we provide a docker file `roboschool.sif` in [google drive](https://drive.google.com/file/d/1KpTpVwoU02AI7uQrk2T9hQ6s15EISRTa/view?usp=sharing) that contains roboschool and the other necessary libraries, adapted from [SunBlaze repo](https://github.com/sunblaze-ucb/rl-generalization). 
    - To download and activate the docker file by singularity (tested in v3.7) on a cluster (on a single server should be similar):
    ```bash
    # download roboschool.sif from the google drive to envs/rl-generalization/roboschool.sif
    # then run singularity shell
    singularity shell --nv -H <local_path>:/home envs/rl-generalization/roboschool.sif
    ```
    - Then you can test it by `import roboschool` in a `python3` shell.

## General Form to Run Our Implementation of Recurrent Model-Free RL and Compared Methods

Basically, we use `.yml` file in `configs/` folder for each subarea of POMDPs. 
To run our implementation, in <local_path> simply use
```
export PYTHONPATH=${PWD}:$PYTHONPATH
python3 policies/main.py configs/<subarea>/<env_name>/<algo_name>.yml
```
where `algo_name` specifies the algorithm name:
- `sac_rnn` and `td3_rnn` correspond to our implementation of recurrent model-free RL
- `ppo_rnn` and `a2c_rnn` correspond to [(Kostrikov, 2018)](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail) implementation of recurrent model-free RL
- `vrm` corresponds to [VRM](https://github.com/oist-cnru/Variational-Recurrent-Models) compared in "standard" POMDPs
- `varibad` corresponds the [off-policy version](https://github.com/Rondorf/BOReL) of original [VariBAD](https://arxiv.org/abs/1910.08348) compared in meta RL
- `MRPO` correspond to [MRPO](https://proceedings.mlr.press/v139/jiang21c.html) compared in robust RL

We have merged the prior methods above into our repository (there is no need to install other repositories), so that future work can use this single repository to run a number of baselines besides ours: A2C-GRU, PPO-GRU, VRM, VariBAD, MRPO. 
Since our code is heavily drawn from those prior works, we encourage authors to [cite those prior papers or implementations](acknowledge.md).
For the compared methods, we use their open-sourced implementation with their default hyperparameters.

## Specific Running Commands for Each Subarea
Please see [run_commands.md](run_commands.md) for details on running our implementation of recurrent model-free RL and also all the compared methods.

## A Minimal Example to Run Our Implementation
Here we provide a stand-alone minimal example with the least dependencies to run our implementation of recurrent model-free RL! 
> Only requires PyTorch and PyBullet, no need to install MuJoCo or roboschool, no external configuration file.

Simply open the Jupyter Notebook [example.ipynb](example.ipynb) and it contains the training and evaluation procedure on a toy POMDP environment (Pendulum-V). It only costs < 20 min to run the whole process.

## Details of Our Implementation of Recurrent Model-Free RL: Decision Factors, Best Variants, Code Features
Please see [our_details.md](our_details.md) for more information on:
- How to tune the decision factors discussed in the paper in the configuration files
- How to tune the other hyperparameters that are also important to training
- Where is the core class of our recurrent model-free RL and the RAM-efficient replay buffer
- Our best variants in subarea and numeric results on all the bar charts and learning curves

## TODO List
- Add documentation on our main code
- Simplify the code (merge the functions of collection and evaluation, add command-line arguments to change the configuration file)

## Acknowledgement
Please see [acknowledge.md](acknowledge.md) for details.

## Citation
If you find our code useful to your work, please consider citing our paper:
```
@article{ni2021recurrent,
  title={Recurrent Model-Free RL is a Strong Baseline for Many POMDPs},
  author={Ni, Tianwei and Eysenbach, Benjamin and Salakhutdinov, Ruslan},
  journal={arXiv preprint arXiv:2110.05038},
  year={2021}
}
```
## Contact
If you have any questions, please create an issue in this repo or contact Tianwei Ni (twni2016@gmail.com)

