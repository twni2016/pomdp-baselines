# Run Experiments

## Difficulty Levels of Environments

Before start running any experiments, we suggest to have a good plan of *environment series* based on difficulty level. As it is hard to analyze and varies from algorithm to algorithm, we provide some rough estimates:

1. Extremely Simple as a Sanity Check: Pendulum-V (also shown in our minimal example jupyter notebook)
2. Simple, Fast, yet Non-trivial: Wind (require precise inference and control), Semi-Circle (sparse reward). Both are continuous gridworlds, thus very fast.
3. Medium: Cheetah-Vel (1-dim stationary hidden state), `*`-Robust (2-dim stationary hidden state), `*`-P (could be roughly inferred by 2nd order MDP)
4. Hard: `*`-Dir (relatively complicated dynamics), `*`-V (long-term inference), `*`-Generalize (extrapolation)

## Best Configs / Variants
To run the best variant of our implemention, please refer to [our_details.md](our_details.md), and then change the corresponding hyperparameters in the config files.


## Specific Running Commands for Each Subarea
General form:
```
export PYTHONPATH=${PWD}:$PYTHONPATH
python3 policies/main.py --cfg configs/<subarea>/<env_name>/<algo_name>.yml \
  [--algo {td3,sac} --seed <int> --cuda <int> --oracle]
```

### "Standard" POMDP
{Ant,Cheetah,Hopper,Walker}-{P,V} in the paper, corresponding to `configs/pomdp/<ant|cheetah|hopper|walker>_blt/<p|v>`, which requires PyBullet. We also provide Pendulum environments for sanity check.

Take Ant-P as example:
```bash
# Run our implemention
python policies/main.py --cfg configs/pomdp/ant_blt/p/rnn.yml --algo sac
# Run Markovian
python policies/main.py --cfg configs/pomdp/ant_blt/p/mlp.yml --algo sac
# Oracle: we directly use Table 1 results (SAC w/ unstructured row) in https://arxiv.org/abs/2005.05719 as it is well-tuned

# Run A2C-GRU from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail
python PPO/main.py --config configs/pomdp/ant_blt/p/a2c_rnn.yml \
    --algo a2c --lr 7e-4 --gae-lambda 1.0 --entropy-coef 0.0 \
    --num-steps 5 --recurrent-policy
# Run PPO-GRU from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail
python PPO/main.py --config configs/pomdp/ant_blt/p/ppo_rnn.yml \
    --algo ppo --lr 2.5e-4 --use-gae --entropy-coef 0.01 \
    --num-steps 128 --recurrent-policy
# Run VRM from https://github.com/oist-cnru/Variational-Recurrent-Models
python VRM/run_experiment.py configs/pomdp/ant_blt/p/vrm.yml
``` 

### Meta RL 

{Semi-Circle, Wind, Cheetah-Vel} in the paper, corresponding to `configs/meta/<point_robot|wind|cheetah_vel|ant_dir>`. Among them, Cheetah-Vel requires MuJoCo, and Semi-Circle can serve as a sanity check. Wind looks simple but not very easy to solve.

Take Semi-Circle as example:
```bash
# Run our implemention
python policies/main.py --cfg configs/meta/point_robot/rnn.yml --algo td3
# Run Markovian
python policies/main.py --cfg configs/meta/point_robot/mlp.yml --algo sac
# Run Oracle
python policies/main.py --cfg configs/meta/point_robot/mlp.yml --algo sac --oracle

# Run off-policy variBAD from https://github.com/Rondorf/BOReL
cd BOReL; python online_training.py --env-type point_robot_sparse
```

{Ant, Cheetah, Humanoid}-Dir in the paper, corresponding to `configs/meta/<ant_dir|cheetah_dir|humanoid_dir>`. They require MuJoCo and are hard to solve.
Take Ant-Dir as example:
```bash
# Run our implemention
python policies/main.py --cfg configs/meta/ant_dir/rnn.yml --algo sac
# Run Markovian
python policies/main.py --cfg configs/meta/ant_dir/mlp.yml --algo sac
# Run Oracle
python policies/main.py --cfg configs/meta/ant_dir/mlp.yml --algo sac --oracle

# For on-policy variBAD and RL2, we use the data from https://github.com/lmzintgraf/varibad
```

### Robust RL
Use roboschool. {Hopper,Walker,Cheetah}-Robust in the paper, corresponding to `configs/rmdp/<hopper|walker|cheetah>`. First, activate the roboschool docker env as introduced in the installation section. 

Take Cheetah-Robust as example:
```bash
## In the docker environment:
# Run our implemention
python3 policies/main.py --cfg configs/rmdp/cheetah/rnn.yml --algo td3
# Run Markovian
python3 policies/main.py --cfg configs/rmdp/cheetah/mlp.yml --algo sac
# Run Oracle
python3 policies/main.py --cfg configs/rmdp/cheetah/mlp.yml --algo sac --oracle

# Run PPO-GRU from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail
python PPO/main.py --config configs/rmdp/cheetah/ppo_rnn.yml \
    --algo ppo --lr 2.5e-4 --use-gae --entropy-coef 0.01 \
    --num-steps 128 --recurrent-policy
# Run MRPO from http://proceedings.mlr.press/v139/jiang21c/jiang21c-supp.zip
python3 MRPO/examples/MRPO/train.py configs/rmdp/cheetah/MRPO.yml
```

### Generalization in RL
Use roboschool. {Hopper|Cheetah}-Generalize in the paper, corresponding to `configs/generalize/Sunblaze<Hopper|HalfCheetah>/<DD-DR-DE|RD-RR-RE>`. 
First, activate the roboschool docker env as introduced in the installation section. 

To train on Default environment and test on the all environments, use `*DD-DR-DE*.yml`; to train on Random environment and test on the all environments, use use `*RD-RR-RE*.yml`. Please see the [SunBlaze paper](https://arxiv.org/abs/1810.12282) for details. 

Take running on `SunblazeHalfCheetahRandomNormal-v0` as example:
```bash
## In the docker environment:
# Run our implemention
python3 policies/main.py --cfg configs/generalize/SunblazeHalfCheetah/RD-RR-RE/rnn.yml --algo td3
# Run Markovian
python3 policies/main.py --cfg configs/generalize/SunblazeHalfCheetah/RD-RR-RE/mlp.yml --algo sac
# Run Oracle
python3 policies/main.py --cfg configs/generalize/SunblazeHalfCheetah/RD-RR-RE/mlp.yml --algo sac --oracle

# For PPO, A2C, EPOpt-PPO-FF, we use the figures from SunBlaze paper
```
